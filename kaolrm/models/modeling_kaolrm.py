# Copyright (c) 2023-2024, Zexin He
# Copyright (c) 2025, Qingtian Zhu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from accelerate.logging import get_logger

from .diffrenderer import DiffRenderer
from .embedder import CameraEmbedder
from .flame import FLAME, sample_points_from_meshes
from .flame_decoder import FLAMEDecoder
from .gaussian_decoder import GaussianDecoder
from .gs import GaussianRenderer
from .transformer import TransformerDecoder

logger = get_logger(__name__)


class KaoLRM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """

    def __init__(
        self,
        camera_embed_dim: int,
        transformer_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        triplane_low_res: int,
        triplane_high_res: int,
        triplane_dim: int,
        encoder_freeze: bool = True,
        encoder_type: str = "dino",
        encoder_model_name: str = "facebook/dino-vitb16",
        encoder_feat_dim: int = 768,
        **kwargs,
    ):
        super().__init__()

        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.camera_embed_dim = camera_embed_dim
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # modules
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )
        self.camera_embedder = CameraEmbedder(
            raw_dim=12 + 4,
            embed_dim=camera_embed_dim,
        )
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3 * triplane_low_res**2, transformer_dim) * (1.0 / transformer_dim) ** 0.5)
        self.transformer = TransformerDecoder(
            block_type="cond_mod",
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            mod_dim=camera_embed_dim,
        )
        self.upsampler = nn.ConvTranspose2d(transformer_dim, triplane_dim, kernel_size=2, stride=2, padding=0)

        self.flame_decoder = FLAMEDecoder(triplane_dim, triplane_high_res)

        self.flame_model = FLAME()

        self.diffrenderer = DiffRenderer()  # for normal and depth of FLAME mesh

        self.gaussian_decoder = GaussianDecoder(self.triplane_dim)

        self.synthesizer = GaussianRenderer()

    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        assert encoder_type in ["dino", "dinov2"], "Unsupported encoder type"
        if encoder_type == "dino":
            from .encoders.dino_wrapper import DinoWrapper

            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == "dinov2":
            from .encoders.dinov2_wrapper import Dinov2Wrapper

            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper

    def forward_transformer(self, image_feats, camera_embeddings):
        assert image_feats.shape[0] == camera_embeddings.shape[0], "Batch size mismatch for image_feats and camera_embeddings!"
        N = image_feats.shape[0]
        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )
        return x

    def reshape_upsample(self, tokens):
        """
        Reshape flat transformer output tokens into a (N, 3, D, H', W') triplane
        tensor, running a 2D upsampler on each plane independently.

        Token layout from the transformer:
            tokens: (N, 3*H*W, D)  — H=W=triplane_low_res

        Step-by-step dimension trace:
            (N, 3*H*W, D)
            → view  → (N, 3, H, W, D)
            → einsum "nihwd→indhw"  → (3, N, D, H, W)   [reorder for batched conv]
            → view  → (3*N, D, H, W)                    [treat planes as batch]
            → upsampler → (3*N, D', H', W')              [2× spatial upscale]
            → view  → (3, N, D', H', W')
            → einsum "indhw→nidhw" → (N, 3, D', H', W') [restore batch-first order]
        """
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum("nihwd->indhw", x)  # [3, N, D, H, W]
        x = x.contiguous().view(3 * N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum("indhw->nidhw", x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    @torch.compile
    def forward_planes(self, image, camera):
        # image: [N, C_img, H_img, W_img]
        # camera: [N, D_cam_raw]
        N = image.shape[0]

        # encode image
        image_feats = self.encoder(image)
        assert image_feats.shape[-1] == self.encoder_feat_dim, (
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"
        )

        # embed camera
        camera_embeddings = self.camera_embedder(camera)
        assert camera_embeddings.shape[-1] == self.camera_embed_dim, (
            f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"
        )

        # transformer generating planes
        tokens = self.forward_transformer(image_feats, camera_embeddings)
        planes = self.reshape_upsample(tokens)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"

        return planes

    def flame2mesh(self, decoded_params, num_sampling, fix_z_trans):
        shape_params = decoded_params["shape"]
        expression_params = decoded_params["expression"]
        pose_params = decoded_params["pose"]
        scale = decoded_params["scale"]
        translation = decoded_params["translation"]

        if fix_z_trans:
            translation[:, -1] = 0.0  # set z-axis translation as 0 to reduce DoF

        vertices, _, lmk3d = self.flame_model(shape_params, expression_params, pose_params)

        B = vertices.shape[0]

        vertices, lmk3d = (
            vertices * scale.unsqueeze(2) + translation.unsqueeze(1),
            lmk3d * scale.unsqueeze(2) + translation.unsqueeze(1),
        )

        # barycentric interpolation here, if not FLAME vertices
        if num_sampling != 5023:
            vertices_render = sample_points_from_meshes(vertices, self.flame_model.faces_tensor.repeat(B, 1, 1), num_sampling)
        else:
            vertices_render = vertices

        return vertices.float(), lmk3d, vertices_render

    def backface_culling(self, vertices_render, gaussians):
        # Approximate back-face culling: keep Gaussians whose anchor vertices have
        # the largest Z values (i.e. closest to the camera in camera space).
        # The divisor 1.8 retains ~56% of points, empirically balancing between
        # removing occluded geometry and preserving sufficient surface coverage.
        zvals = vertices_render[..., 2]  # (B, N)
        k = int(zvals.shape[1] / 1.8)
        _, topk_indices = torch.topk(zvals, k, dim=1)  # topk_indices: (B, k)
        gaussians_topk = torch.gather(
            gaussians,  # (B, N, D)
            dim=1,
            index=topk_indices.unsqueeze(2).expand(-1, -1, gaussians.shape[2]),  # (B, k, D)
        )
        return gaussians_topk

    def forward(
        self,
        image,
        source_camera,
        world_view_matrix,
        projection_matrix,
        optical_center,
        render_bg_colors,
        fov_x,
        fov_y,
        K,
        R,
        T,
        num_sampling,
        focal,
        render_size,
    ):
        B = source_camera.shape[0]

        planes = self.forward_planes(image, source_camera)

        decoded_params = self.flame_decoder(planes)

        fix_z = True if focal > 1.0 else False
        vertices, lmk3d, vertices_render = self.flame2mesh(decoded_params, num_sampling=num_sampling, fix_z_trans=fix_z)

        faces = self.flame_model.faces_tensor.repeat(B, 1, 1)
        flame_rendering = self.diffrenderer.render_normal_wFM(vertices, faces, K, R, T, render_size)

        gaussians = self.gaussian_decoder(vertices_render, planes, scale_coords=focal)
        gaussians = torch.cat((vertices_render, gaussians), dim=-1)

        if fix_z:  # for monocular only
            gaussians = self.backface_culling(vertices_render, gaussians)

        gs_rendering = self.synthesizer.render(
            gaussians,
            cam_view=world_view_matrix,
            cam_view_proj=projection_matrix,
            cam_pos=optical_center,
            bg_color=render_bg_colors,
            fov_x=fov_x,
            fov_y=fov_y,
            render_size=render_size,
        )

        return {
            "rgb": gs_rendering["image"],
            "alpha": gs_rendering["alpha"],
            "gs_normal": gs_rendering["normal"],  # to differentiate with flame_normal/flame_depth
            "gs_depth": gs_rendering["depth"],
            "gaussians": gaussians,  # contains vertices_render
            "planes": planes,
            "vertices": vertices,
            "lmk3d": lmk3d,
            "beta": decoded_params["shape"],
            "psi": decoded_params["expression"],
            "theta": decoded_params["pose"],
            "flame_normal": flame_rendering["normal"],
            "flame_depth": flame_rendering["depth"],
            "flame_mask": flame_rendering["facial_mask"],
        }
