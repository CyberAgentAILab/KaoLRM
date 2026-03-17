import numpy as np
import torch
from diff_surfel_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData


def inverse_sigmoid(x, eps=1e-6):
    """inversion of sigmoid function.

    Args:
        x (Tensor): x
        eps (float, optional): eps. Defaults to 1e-6.

    Returns:
        Tensor: log(x / (1 - x))
    """
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


class GaussianRenderer:
    def __init__(self): ...

    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color, fov_x, fov_y, render_size):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        device = gaussians.device
        B, V = cam_view.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        normals = []
        depths = []
        for b in range(B):
            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:6].contiguous().float()
            gaussians[b, :, 6] = 0.0  # in case of regularization
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float()  # [N, 3]

            for v in range(V):
                # render novel views
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                tanHalfFovY = torch.tan(fov_y[b, v] / 2)
                tanHalfFovX = torch.tan(fov_x[b, v] / 2)

                image_wise_bg_color = bg_color[b, v]

                raster_settings = GaussianRasterizationSettings(
                    image_height=render_size,
                    image_width=render_size,
                    tanfovx=tanHalfFovX,
                    tanfovy=tanHalfFovY,
                    bg=image_wise_bg_color,
                    scale_modifier=1,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                rendered_image, radii, allmap = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )
                rendered_alpha = allmap[1:2]

                rendered_normal = allmap[2:5]

                rendered_depth = allmap[0:1]  # expected, instead of median
                rendered_depth = torch.nan_to_num(rendered_depth / rendered_alpha, 0, 0)

                rendered_image = rendered_image.clamp(0, 1)
                rendered_depth = rendered_depth.clamp(0, 5)
                rendered_normal = torch.nn.functional.normalize(rendered_normal, dim=0)

                images.append(rendered_image)
                alphas.append(rendered_alpha)
                normals.append(rendered_normal)
                depths.append(rendered_depth)

        images = torch.stack(images, dim=0).view(B, V, 3, render_size, render_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, render_size, render_size)

        normals = torch.stack(normals, dim=0).view(B, V, 3, render_size, render_size)
        depths = torch.stack(depths, dim=0).view(B, V, 1, render_size, render_size)
        normals[:, :, [1, 2], :, :] *= -1.0  # revert axis

        return {
            "image": images,  # [B, V, 3, H, W]
            "alpha": alphas,  # [B, V, 1, H, W]
            "normal": normals,  # [B, V, 3, H, W]
            "depth": depths,  # [B, V, 1, H, W]
        }

    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, "only support batch size 1"

        from plyfile import PlyData, PlyElement

        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float()  # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        fields = ["x", "y", "z"]
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            fields.append(f"f_dc_{i}")
        fields.append("opacity")
        for i in range(scales.shape[1]):
            fields.append(f"scale_{i}")
        for i in range(rotations.shape[1]):
            fields.append(f"rot_{i}")

        dtype_full = [(attribute, "f4") for attribute in fields]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        PlyData([el]).write(path)

    def load_ply(self, path, compatible=True):

        plydata = PlyData.read(path)

        xyz = np.stack(
            (np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),
            axis=1,
        )
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float()  # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
