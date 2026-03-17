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


import math

import torch

"""
R: (N, 3, 3)
T: (N, 3)
E: (N, 4, 4)
vector: (N, 3)
"""


def compose_extrinsic_R_T(R: torch.Tensor, T: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from R and T.
    Batched I/O.
    """
    RT = torch.cat((R, T.unsqueeze(-1)), dim=-1)
    return compose_extrinsic_RT(RT)


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([RT, torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)], dim=1)


def decompose_extrinsic_R_T(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into R and T.
    Batched I/O.
    """
    RT = decompose_extrinsic_RT(E)
    return RT[:, :, :3], RT[:, :, 3]


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def camera_normalization_objaverse(
    normed_dist_to_center, poses: torch.Tensor, lmks: torch.Tensor = None, ret_transform: bool = False
):
    """
    Normalize all camera poses so that the first (pivotal) camera aligns with a
    fixed canonical extrinsic frame.  This ensures the 3D reconstruction is
    expressed in a consistent coordinate system regardless of the raw capture rig.

    Canonical frame convention (LLFF-style, Z-up world):
        - Camera looks toward the -Y axis in world space
        - World +Z is "up"
        - Camera sits at distance `dist_to_center` along the -Y axis

    Steps:
        1. Compute the transformation T that maps the pivotal pose → canonical.
        2. Apply T uniformly to every pose in the batch.

    Args:
        normed_dist_to_center: Either a float (explicit distance) or "auto"
            (infer distance from the pivotal camera translation).
        poses:  (N, 3, 4) RT matrices for all views.
        lmks:   Optional landmarks; unused here but kept for API consistency.
        ret_transform: If True, also return the normalization matrix.
    """
    assert normed_dist_to_center is not None
    pivotal_pose = compose_extrinsic_RT(poses[:1])
    dist_to_center = (
        pivotal_pose[:, :3, 3].norm(dim=-1, keepdim=True).item() if normed_dist_to_center == "auto" else normed_dist_to_center
    )

    # Desired canonical extrinsic: camera placed at (0, dist, 0) looking at origin.
    # Row layout: [x_cam | y_cam | z_cam | t] in world coordinates.
    canonical_camera_extrinsics = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 0, -1, -dist_to_center],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ],
        dtype=torch.float32,
    )
    # camera_norm_matrix = canonical * pivotal^{-1}
    # Applying it to any pose P gives: canonical * pivotal^{-1} * P,
    # which maps pivotal → canonical and shifts every other pose accordingly.
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # Apply the normalization transform to all views.
    poses = compose_extrinsic_RT(poses)
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)
    poses = decompose_extrinsic_RT(poses)

    if ret_transform:
        return poses, camera_norm_matrix.squeeze(dim=0)
    return poses


def get_normalized_camera_intrinsics(intrinsics: torch.Tensor):
    """
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    Return batched fx, fy, cx, cy
    If the intrinsics have already been normalized, they will remain the same
    """
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy


def build_camera_principle(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    return torch.cat(
        [
            RT.reshape(-1, 12),
            fx.unsqueeze(-1),
            fy.unsqueeze(-1),
            cx.unsqueeze(-1),
            cy.unsqueeze(-1),
        ],
        dim=-1,
    )


def build_camera_standard(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    E = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    intrinsic_matrix = torch.stack(
        [
            torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
            torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
            torch.tensor([[0, 0, 1]], dtype=torch.float32, device=RT.device).repeat(RT.shape[0], 1),
        ],
        dim=1,
    )
    return torch.cat(
        [
            E.reshape(-1, 16),
            intrinsic_matrix.reshape(-1, 9),
        ],
        dim=-1,
    )


def center_looking_at_camera_pose(
    camera_position: torch.Tensor,
    look_at: torch.Tensor = None,
    up_world: torch.Tensor = None,
    device: torch.device = torch.device("cpu"),
):
    """
    camera_position: (M, 3)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4)
    """
    # Default: look at the world origin, world-up is +Z.
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
    up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # Build a right-handed orthonormal camera frame via Gram-Schmidt:
    #   z_axis: points from the look-at target toward the camera (camera "back")
    #   x_axis: right axis, perpendicular to z and world-up
    #   y_axis: up axis, perpendicular to x and z (reorthogonalized)
    z_axis = camera_position - look_at
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    x_axis = torch.linalg.cross(up_world, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = torch.linalg.cross(z_axis, x_axis)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
    # Stack as camera-to-world (c2w) RT matrix: columns are [x, y, z, position]
    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    return extrinsics


def frontal_views_linspace(n_views: int, radius: float = 2.0, height: float = 0.0, device: torch.device = torch.device("cpu")):
    """
    n_views: number of surrounding views
    radius: camera dist to center
    height: height of the camera
    return: (M, 3, 4)
    """
    assert n_views > 0
    assert radius > 0

    theta = torch.linspace(-torch.pi, 0, n_views, device=device)
    projected_radius = math.sqrt(radius**2 - height**2)
    x = torch.cos(theta) * projected_radius
    y = torch.sin(theta) * projected_radius
    z = torch.full((n_views,), height, device=device)

    camera_positions = torch.stack([x, y, z], dim=1)
    extrinsics = center_looking_at_camera_pose(camera_positions, device=device)

    return extrinsics


def surrounding_views_linspace(
    n_views: int, radius: float = 2.0, height: float = 0.0, device: torch.device = torch.device("cpu")
):
    """
    n_views: number of surrounding views
    radius: camera dist to center
    height: height of the camera
    return: (M, 3, 4)
    """
    assert n_views > 0
    assert radius > 0

    theta = torch.linspace(-torch.pi / 2, 3 * torch.pi / 2, n_views, device=device)
    projected_radius = math.sqrt(radius**2 - height**2)
    x = torch.cos(theta) * projected_radius
    y = torch.sin(theta) * projected_radius
    z = torch.full((n_views,), height, device=device)

    camera_positions = torch.stack([x, y, z], dim=1)
    extrinsics = center_looking_at_camera_pose(camera_positions, device=device)

    return extrinsics


def create_intrinsics(
    f: float,
    c: float = None,
    cx: float = None,
    cy: float = None,
    w: float = 1.0,
    h: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
):
    """
    return: (3, 2)
    """
    fx = fy = f
    if c is not None:
        assert cx is None and cy is None, "c and cx/cy cannot be used together"
        cx = cy = c
    else:
        assert cx is not None and cy is not None, "cx/cy must be provided when c is not provided"
    fx, fy, cx, cy, w, h = fx / w, fy / h, cx / w, cy / h, 1.0, 1.0
    intrinsics = torch.tensor(
        [
            [fx, fy],
            [cx, cy],
            [w, h],
        ],
        dtype=dtype,
        device=device,
    )
    return intrinsics


def focal2fov(focal, pixels):
    return 2 * torch.atan(pixels / (2 * focal))


def get_fov(intrinsics):
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)
    return FovX, FovY


def get_projection_matrix(intrinsics):
    """
    Build an OpenGL-style perspective projection matrix from camera intrinsics.

    The resulting matrix maps camera-space points into clip space (NDC [-1, 1]).
    Formula derived from the standard OpenGL projection:

        P[0,0] = 2*fx/W,          P[0,2] = 2*(cx/W) - 1   (horizontal shift)
        P[1,1] = 2*fy/H,          P[1,2] = 2*(cy/H) - 1   (vertical shift)
        P[2,2] = (far+near)/(far-near),  P[2,3] = -2*far*near/(far-near)
        P[3,2] = 1                (perspective divide via w = z)

    Returns the *transposed* matrix because downstream rasterizers (e.g. 3DGS)
    expect row-vector conventions (point @ P).

    Args:
        intrinsics: (B, 3, 2) — [[fx, fy], [cx, cy], [width, height]]

    Returns:
        (B, 4, 4) projection matrices (transposed for row-vector use).
    """
    B = intrinsics.shape[0]
    znear, zfar = 0.01, 1000.0
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]

    P = torch.zeros(B, 4, 4)

    P[:, 0, 0] = 2 * fx / width
    P[:, 0, 2] = -1 + 2 * (cx / width)

    P[:, 1, 1] = 2 * fy / height
    P[:, 1, 2] = -1 + 2 * (cy / height)

    P[:, 2, 2] = (zfar + znear) / (zfar - znear)
    P[:, 2, 3] = -1 * 2 * zfar * znear / (zfar - znear)
    P[:, 3, 2] = 1.0

    return P.transpose(1, 2)


def get_world_view_transform_optical_center(poses):
    """
    Convert LLFF-convention camera-to-world (c2w) poses into the world-to-camera
    (w2c) transform required by the 3D Gaussian Splatting rasterizer, together
    with the camera optical center in world space.

    Coordinate-system chain:
        LLFF  →  OpenGL  →  OpenCV
    - LLFF2OPENGL swaps Y/Z axes (LLFF uses Y-down/Z-forward; OpenGL uses Y-up/Z-back).
    - OPENGL2OPENCV flips Y and Z (OpenCV uses Y-down/Z-forward, opposite to OpenGL).

    The rasterizer expects a *transposed* w2c matrix (row-vector convention).

    Args:
        poses: (B, 3, 4) RT matrices in LLFF convention.

    Returns:
        w2c_T:  (B, 4, 4) transposed world-to-camera matrix (OpenCV convention).
        C:      (B, 3)    camera optical center in world space.
    """
    B = poses.shape[0]
    c2w = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    c2w[:, :3, :] = poses

    # LLFF (right/up/back) → OpenGL (right/up/back, different axis sign)
    LLFF2OPENGL = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]], dtype=torch.float32).repeat(B, 1, 1)

    # OpenGL (right/up/−forward) → OpenCV (right/down/forward)
    OPENGL2OPENCV = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        ],
        dtype=torch.float32,
    ).repeat(B, 1, 1)

    c2w_opencv = torch.bmm(torch.bmm(LLFF2OPENGL, c2w), OPENGL2OPENCV)  # (B, 4, 4)
    C = c2w_opencv[:, :3, -1]  # optical center = camera position in world space
    w2c_opencv = torch.inverse(c2w_opencv)

    return w2c_opencv.transpose(1, 2), C


def get_side_view_projection(poses, intrinsics):
    # Batched IO
    B = poses.shape[0]
    c2w = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    c2w[:, :3, :] = poses

    LLFF2OPENGL = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]], dtype=torch.float32).repeat(B, 1, 1)

    OPENGL2OPENCV = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        ],
        dtype=torch.float32,
    ).repeat(B, 1, 1)

    c2w_opencv = torch.bmm(torch.bmm(LLFF2OPENGL, c2w), OPENGL2OPENCV)  # (B, 4, 4)
    Rt_opencv = torch.inverse(c2w_opencv)[:, :3, :]

    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]

    K = torch.zeros(B, 3, 3)
    K[:, 0, 0], K[:, 1, 1] = fx, fy
    K[:, 0, 2], K[:, 1, 2] = cx, cy
    K[:, 2, 2] = 1.0

    return torch.bmm(K, Rt_opencv)


def get_gs_inputs(render_intrinsics, poses):
    projection_matrix = get_projection_matrix(render_intrinsics.repeat(poses.shape[0], 1, 1))
    world_view_matrix, optical_center = get_world_view_transform_optical_center(poses)
    full_proj_matrix = torch.bmm(world_view_matrix, projection_matrix)
    fov_x, fov_y = get_fov(render_intrinsics.repeat(poses.shape[0], 1, 1))

    render_projection = get_side_view_projection(poses, render_intrinsics.repeat(poses.shape[0], 1, 1))
    return {
        "fov_x": fov_x,
        "fov_y": fov_y,
        "world_view_matrix": world_view_matrix,
        "optical_center": optical_center,
        "full_proj_matrix": full_proj_matrix,
        "render_projection": render_projection,
    }


def get_pt3d_K(intrinsics):
    B = intrinsics.shape[0]
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    _width, _height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    K = torch.zeros(B, 4, 4)
    K[:, 0, 0], K[:, 1, 1] = fx, fy
    K[:, 0, 2], K[:, 1, 2] = cx, cy
    K[:, 2, 2] = 1
    K[:, 3, 2] = 1
    return K


def get_pt3d_RT(poses):
    """
    Convert LLFF poses to the R, T convention expected by PyTorch3D cameras.

    PyTorch3D uses a left-handed coordinate system where X points left, Y points
    up, and Z points into the scene.  The sign-flip matrix S = diag(-1, -1, 1)
    converts from OpenCV's right-handed system (X right, Y down, Z forward) to
    PyTorch3D's convention.

    Args:
        poses: (B, 3, 4) RT matrices in LLFF convention.

    Returns:
        R: (B, 3, 3) rotation matrices in PyTorch3D convention.
        T: (B, 3)    translation vectors in PyTorch3D convention.
    """
    B = poses.shape[0]
    c2w = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    c2w[:, :3, :] = poses

    LLFF2OPENGL = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]], dtype=torch.float32).repeat(B, 1, 1)

    OPENGL2OPENCV = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        ],
        dtype=torch.float32,
    ).repeat(B, 1, 1)

    c2w_opencv = torch.bmm(torch.bmm(LLFF2OPENGL, c2w), OPENGL2OPENCV)  # (B, 4, 4)

    # S flips X and Y to convert OpenCV → PyTorch3D handedness.
    S = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    R = c2w_opencv[:, :3, :3]  # (B, 3, 3)
    # T = -R^T @ t  converts the world-space translation to camera-space.
    T = -torch.matmul(R.transpose(1, 2), c2w_opencv[:, :3, 3:4]).squeeze(-1)  # (B, 3)
    T = torch.matmul(S, T.t()).t()  # (B, 3)
    R = torch.matmul(R, S)  # (B, 3, 3)
    return R, T


def get_pt3d_inputs(render_intrinsics, poses):
    K_batched = get_pt3d_K(render_intrinsics.repeat(poses.shape[0], 1, 1))
    R_batched, T_batched = get_pt3d_RT(poses)

    return {
        "K": K_batched,
        "R": R_batched,
        "T": T_batched,
    }
