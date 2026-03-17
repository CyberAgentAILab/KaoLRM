import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.

    Bugfix reference: https://github.com/NVlabs/eg3d/issues/67
    """
    return torch.tensor(
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]],
        dtype=torch.float32,
    )


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N * n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N * n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(plane_axes, plane_features, coordinates, mode="bilinear", padding_mode="zeros", box_warp=2):
    """
    Sample triplane features at given 3D coordinates via bilinear grid sampling.

    Each triplane covers a 3D bounding cube of side `box_warp`.  Coordinates are
    first rescaled to [-1, 1] (the range expected by F.grid_sample), then projected
    onto each of the three orthogonal planes, and finally sampled with bilinear
    interpolation.  Points outside the cube produce zero features (padding_mode="zeros").

    Args:
        plane_axes:     (n_planes, 3, 3) — axis definitions for each plane.
        plane_features: (N, n_planes, C, H, W) — triplane feature maps.
        coordinates:    (N, M, 3) — query 3D points in world/canonical space.
        box_warp:       Side length of the bounding cube the triplane covers.

    Returns:
        (N, n_planes, M, C) — per-plane features for each query point.
    """
    assert padding_mode == "zeros"
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N * n_planes, C, H, W)

    # Rescale coordinates from [-box_warp/2, box_warp/2] to [-1, 1].
    coordinates = (2 / box_warp) * coordinates

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = (
        torch.nn.functional.grid_sample(
            plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False
        )
        .permute(0, 3, 2, 1)
        .reshape(N, n_planes, M, C)
    )
    return output_features


class _FeatureConditioner(nn.Module):
    def __init__(self, dim_condition, dim_feature):
        super().__init__()
        self.film_layer = nn.Linear(dim_condition, dim_feature * 2)  # outputs: [scale | shift]

    def forward(self, feature, index_embedding):
        # feature: [B, 5023, triplane_dim*3]
        # index_embedding: [B, N, dim_condition]

        film_params = self.film_layer(index_embedding)  # [B, N, dim_feature * 2]
        scale, shift = film_params.chunk(2, dim=-1)  # Each: [B, N, dim_feature]

        conditioned_feature = feature * scale + shift  # Element-wise
        return conditioned_feature


class OSGDecoder(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.

    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(
                *[
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        activation(),
                    ]
                    for _ in range(num_layers - 2)
                ]
            ),
            nn.Linear(hidden_dim, 11),
        )

        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @torch.compile
    def forward(self, sampled_features):
        # Aggregate features by mean
        # sampled_features = sampled_features.mean(1)
        # Aggregate features by concatenation
        _N, n_planes, _M, _C = sampled_features.shape
        # Concatenate features from all three planes: (N, M, n_planes*C)
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes * _C)

        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N * M, C)

        y = self.net(x)
        y = y.view(N, M, -1)
        return y


class GaussianDecoder(nn.Module):
    def __init__(self, dim_triplane):
        super().__init__()
        self.plane_axes = generate_planes()
        self.decoder = OSGDecoder(dim_triplane)
        self.scale_act = lambda x: 0.5 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x) * (1 + 2 * 0.001) - 0.001  # from Mip-NeRF, known as saturated

    def forward(self, coordinates, planes, scale_coords):
        B, num_pts = coordinates.shape[0], coordinates.shape[1]

        # Note: the coord system of the pretrained LRM's triplane is wrongly converted
        #       so to ensure the alignment, the following transform is performed
        LLFF2OPENGL = torch.tensor(
            [[[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]], dtype=torch.float32, device=coordinates.device
        ).repeat(B, 1, 1)

        # Note: the triplane is pretrained with a focal of 0.75 and with a size of [-1,+1]^3
        #       when applying a different focal for training (while images remain unchanged)
        #       it's necessary to scale the coords accordingly by new_focal/0.75
        scale = scale_coords / 0.75
        points_h = torch.cat(
            [scale * coordinates.clone(), torch.ones(B, num_pts, 1, device=coordinates.device)], dim=-1
        )  # Shape (B, num_pts, 4)
        points_transformed_h = torch.bmm(points_h, LLFF2OPENGL.transpose(1, 2))
        points_transformed = points_transformed_h[..., :3]

        sampled_features = sample_from_planes(self.plane_axes.to(planes.device), planes, points_transformed)  # [B, N, 32]

        # out_gaussians layout per point (11 values total):
        #   [0]     opacity   — scalar, passed through sigmoid
        #   [1:4]   scales    — (sx, sy, sz), passed through 0.5*softplus for positivity
        #   [4:8]   rotation  — quaternion (w, x, y, z), L2-normalized
        #   [8:11]  RGB color — (r, g, b), passed through saturated sigmoid (Mip-NeRF style)
        out_gaussians = self.decoder(sampled_features)
        opacity = self.opacity_act(out_gaussians[..., 0:1])
        scales = self.scale_act(out_gaussians[..., 1:4])
        rotations = self.rot_act(out_gaussians[..., 4:8])
        rgbs = self.rgb_act(out_gaussians[..., 8:])

        gaussians = torch.cat([opacity, scales, rotations, rgbs], dim=-1)

        return gaussians


if __name__ == "__main__":
    coordinates = torch.rand([16, 5023, 3])
    planes = torch.rand([16, 3, 32, 64, 64])
    model = GaussianDecoder(32)
    gaussians = model(coordinates, planes)
    print(gaussians.shape)
