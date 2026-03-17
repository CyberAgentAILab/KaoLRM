import pickle

import numpy as np
import torch
import trimesh
from einops import rearrange
from pytorch3d.renderer import (
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.structures import Meshes

with open("./data/FLAME_masks.pkl", "rb") as f:
    flame_masks = pickle.load(f, encoding="latin1")
mesh = trimesh.load_mesh("./data/template.ply")
n_faces = len(mesh.faces)
selected_indices = np.concatenate(
    (
        flame_masks["right_ear"],
        flame_masks["left_ear"],
        flame_masks["face"],
        flame_masks["left_eyeball"],
        flame_masks["right_eyeball"],
        flame_masks["nose"],
    )
)
selected_set = set(selected_indices.tolist()) - set(flame_masks["forehead"].tolist())
# Find faces where all 3 vertices belong to selected_indices
selected_face_indices = np.where(np.all(np.isin(mesh.faces, list(selected_set)), axis=1))[0]


class NormalShader(ShaderBase):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", None)

        faces = meshes.faces_packed()
        vertex_normals = meshes.verts_normals_packed()
        face_normals = vertex_normals[faces]

        pix_to_face = fragments.pix_to_face
        bary_coords = fragments.bary_coords
        face_normals_at_pixels = face_normals[pix_to_face]  # (N, H, W, K, 3, 3)

        interpolated_normals = (bary_coords.unsqueeze(-1) * face_normals_at_pixels).sum(dim=-2)
        interpolated_normals = torch.nn.functional.normalize(interpolated_normals, dim=-1)

        # === Transform normals to camera space
        # Note: perform world2camera transform (R.T)
        R = cameras.R
        R = R[:, None, None, None, :, :]
        normals_cam = torch.matmul(R.transpose(-2, -1), interpolated_normals.unsqueeze(-1)).squeeze(-1)

        normals_cam[..., [0, 2]] *= -1

        depth = fragments.zbuf[..., 0]  # (N, H, W)
        mask = fragments.pix_to_face[..., 0] >= 0  # Foreground mask (N, H, W)
        depth[~mask] = 0  # or np.nan, or whatever background value you prefer
        normals = normals_cam[..., 0, :]
        normals[~mask, :] = 0

        return normals, depth


# with facial mask
class NormalShader_wFM(ShaderBase):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", None)

        # Get packed faces and normals
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        face_normals = vertex_normals[faces]  # (F, 3, 3)

        pix_to_face = fragments.pix_to_face[..., 0]  # (N, H, W), face index per pixel
        bary_coords = fragments.bary_coords[..., 0, :]  # (N, H, W, 3)
        mask_fg = pix_to_face >= 0  # Foreground mask

        # Interpolate normals at pixel centers
        face_normals_at_pixels = face_normals[pix_to_face.clamp_min(0)]  # (N, H, W, 3, 3)
        # (Clamp -1s to 0, will mask out with mask_fg later)
        interpolated_normals = (bary_coords.unsqueeze(-1) * face_normals_at_pixels).sum(dim=-2)
        interpolated_normals = torch.nn.functional.normalize(interpolated_normals, dim=-1)

        R = cameras.R
        R = R[:, None, None, :, :]  # (N, 1, 1, 3, 3)
        normals_cam = torch.matmul(R.transpose(-2, -1), interpolated_normals.unsqueeze(-1)).squeeze(-1)
        # Flip convention, for PyTorch3D coordinate alignment if needed:
        normals_cam[..., [0, 2]] *= -1

        normals = normals_cam  # (N, H, W, 3)
        normals[~mask_fg] = 0

        # Depth
        depth = fragments.zbuf[..., 0]  # (N, H, W)
        depth[~mask_fg] = 0

        # =========================
        # FACIAL REGION MASK
        # =========================
        # Build a mask mapping from face index to [True if facial triangle, else False]
        n_faces = meshes.faces_packed().shape[0]

        facial_flag = torch.zeros(n_faces, dtype=torch.bool, device=pix_to_face.device)
        Nf = 9976
        num_meshes = len(meshes)
        for i in range(num_meshes):
            facial_flag[i * Nf + selected_face_indices] = True

        pix_to_face_safe = pix_to_face.clone()
        pix_to_face_safe[~mask_fg] = 0  # to avoid indexing at -1
        facial_mask = mask_fg & facial_flag[pix_to_face_safe]  # (N, H, W). True only for facial triangles

        return normals, depth, facial_mask


class DiffRenderer:
    def __init__(self): ...

    def render_normal(self, vertices, faces, K, R, T, render_size):
        """
        vertices: B num_vertices 3
        faces: B num_faces 3
        K: B V 4 4
        R: B V 3 3
        T: B V 3 3
        """
        # print(f"{vertices.shape=},{faces.shape=},{K.shape=},{R.shape=},{T.shape=}")
        raster_settings = RasterizationSettings(
            image_size=(render_size, render_size),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        renderer_normal = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings), shader=NormalShader())

        # in order to work with accelerate
        with torch.amp.autocast("cuda", enabled=False):
            B, V = K.shape[0], K.shape[1]
            mesh = Meshes(
                verts=vertices,  # (B, V, 3)
                faces=faces,  # (B, F, 3)
            )
            meshes = mesh.extend(V)

            R = R.reshape(-1, 3, 3)
            T = T.reshape(-1, 3)
            K = K.reshape(-1, 4, 4)

            image_size = torch.tensor([render_size, render_size], device=K.device)
            cams = PerspectiveCameras(K=K, R=R, T=T, device=K.device, in_ndc=False, image_size=image_size.repeat(B * V, 1))
            normal_map, depth_map = renderer_normal(meshes, cameras=cams)

        # (BV, H, W, 3) (BV, H, W) -> (B, V, 3, H, W) (B, V, 1, H, W)
        normal = rearrange(normal_map, "(B V) H W C -> B V C H W", B=B, V=V, C=3)
        depth = rearrange(depth_map, "(B V) H W -> B V H W", B=B, V=V).unsqueeze(2)
        return {
            "normal": normal,
            "depth": depth,
        }

    # with facial mask
    def render_normal_wFM(self, vertices, faces, K, R, T, render_size):
        """
        vertices: B num_vertices 3
        faces: B num_faces 3
        K: B V 4 4
        R: B V 3 3
        T: B V 3 3
        """
        raster_settings = RasterizationSettings(
            image_size=(render_size, render_size),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        renderer_normal = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings), shader=NormalShader_wFM())

        # in order to work with accelerate
        with torch.amp.autocast("cuda", enabled=False):
            B, V = K.shape[0], K.shape[1]
            mesh = Meshes(
                verts=vertices,  # (B, V, 3)
                faces=faces,  # (B, F, 3)
            )
            meshes = mesh.extend(V)

            R = R.reshape(-1, 3, 3)
            T = T.reshape(-1, 3)
            K = K.reshape(-1, 4, 4)

            image_size = torch.tensor([render_size, render_size], device=K.device)
            cams = PerspectiveCameras(K=K, R=R, T=T, device=K.device, in_ndc=False, image_size=image_size.repeat(B * V, 1))
            normal_map, depth_map, facial_mask = renderer_normal(meshes, cameras=cams)

        # (BV, H, W, 3) (BV, H, W) -> (B, V, 3, H, W) (B, V, 1, H, W)
        normal = rearrange(normal_map, "(B V) H W C -> B V C H W", B=B, V=V, C=3)
        depth = rearrange(depth_map, "(B V) H W -> B V H W", B=B, V=V).unsqueeze(2)
        facial_mask = rearrange(facial_mask, "(B V) H W -> B V H W", B=B, V=V)

        return {
            "normal": normal,
            "depth": depth,
            "facial_mask": facial_mask,
        }

    def render_phong(self, vertices, faces, K, R, T, render_size):
        """
        vertices: B num_vertices 3
        faces: B num_faces 3
        K: B V 4 4
        R: B V 3 3
        T: B V 3 3
        """
        raster_settings = RasterizationSettings(
            image_size=(render_size, render_size),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        renderer_phong = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings), shader=SoftPhongShader())
        # in order to work with accelerate
        with torch.amp.autocast("cuda", enabled=False):
            B, V = K.shape[0], K.shape[1]
            textures = TexturesVertex(verts_features=torch.ones_like(vertices, device=vertices.device) * 0.7)
            mesh = Meshes(
                verts=vertices,  # (B, V, 3)
                faces=faces,  # (B, F, 3)
                textures=textures,
            )
            meshes = mesh.extend(V)

            R = R.reshape(-1, 3, 3)
            T = T.reshape(-1, 3)
            K = K.reshape(-1, 4, 4)

            image_size = torch.tensor([render_size, render_size], device=K.device)
            cams = PerspectiveCameras(K=K, R=R, T=T, device=K.device, in_ndc=False, image_size=image_size.repeat(B * V, 1))

            lights = PointLights(
                device=K.device, location=((0.0, 0.0, 2.0),), ambient_color=((0.5, 0.5, 0.5),), diffuse_color=((0.5, 0.5, 0.5),)
            )
            materials = Materials(
                ambient_color=((1.0, 1.0, 1.0),),
                diffuse_color=((1.0, 1.0, 1.0),),
                specular_color=((0.0, 0.0, 0.0),),
                device=K.device,
            )
            images = renderer_phong(meshes, cameras=cams, lights=lights, materials=materials)

        # (BV, H, W, 4) -> (B, V, 4, H, W)
        rgba = rearrange(images, "(B V) H W C -> B V C H W", B=B, V=V, C=4)
        return {
            "rgba": rgba,
        }
