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

import os
import random
from typing import Union

import numpy as np
import torch
from megfile import smart_open, smart_path_join

from ..utils.proxy import no_proxy
from .base import BaseDataset
from .cam_utils import build_camera_principle, camera_normalization_objaverse, get_gs_inputs, get_pt3d_inputs

__all__ = ["ObjaverseDataset"]


class ObjaverseDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        meta_path: str,
        sample_side_views: int,
        render_image_res: int,
        source_image_res: int,
        normalize_camera: bool,
        normed_dist_to_center: Union[float, str] = None,
        num_all_views: int = 32,
        **kwargs,
    ):
        super().__init__(root_dir, meta_path)
        self.sample_side_views = sample_side_views
        self.render_image_res = render_image_res
        self.source_image_res = source_image_res
        self.normalize_camera = normalize_camera
        self.normed_dist_to_center = normed_dist_to_center
        self.num_all_views = num_all_views

    @staticmethod
    def _load_pose(file_path):
        pose = np.load(smart_open(file_path, "rb"))
        pose = torch.from_numpy(pose).float()
        return pose

    @no_proxy
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        uid = self.uids[idx]

        root_dir = self.root_dir

        pose_dir = os.path.join(root_dir, uid, "pose")
        rgba_dir = os.path.join(root_dir, uid, "rgba")
        intrinsics_path = os.path.join(root_dir, uid, "intrinsics.npy")

        # load intrinsics
        intrinsics = np.load(smart_open(intrinsics_path, "rb"))
        intrinsics = torch.from_numpy(intrinsics).float()

        # sample views (incl. source view and side views)
        sample_views = np.random.choice(range(self.num_all_views), self.sample_side_views + 1, replace=False)

        lmk_path = os.path.join(root_dir, uid, "lmks2d.npy")
        lmks2d_dict = np.load(lmk_path, allow_pickle=True)

        poses, rgbs, bg_colors, alphas, lmks2d = [], [], [], [], []
        source_image = None

        for view in sample_views:
            pose_path = smart_path_join(pose_dir, f"{view:03d}.npy")
            rgba_path = smart_path_join(rgba_dir, f"{view:03d}.png")

            pose = self._load_pose(pose_path)
            bg_color = random.choice([0.0, 0.5, 1.0])
            rgb, alpha = self._load_rgba_image(rgba_path, bg_color=bg_color)

            lmks2d_view = torch.from_numpy(lmks2d_dict[view]).float()
            poses.append(pose)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            alphas.append(alpha)
            lmks2d.append(lmks2d_view)

            if source_image is None:
                source_image, alpha = self._load_rgba_image(rgba_path, bg_color=1.0)

        poses = torch.stack(poses, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        alphas = torch.cat(alphas, dim=0)
        lmks2d = torch.stack(lmks2d, dim=0)

        if self.normalize_camera:
            poses = camera_normalization_objaverse(self.normed_dist_to_center, poses)

        source_camera = build_camera_principle(poses[:1], intrinsics.unsqueeze(0)).squeeze(0)

        raw_source_res = source_image.shape[2]
        source_image = torch.nn.functional.interpolate(
            source_image, size=(self.source_image_res, self.source_image_res), mode="bicubic", align_corners=True
        ).squeeze(0)
        source_image = torch.clamp(source_image, 0, 1)

        render_image_res = self.render_image_res
        render_image = torch.nn.functional.interpolate(
            rgbs, size=(render_image_res, render_image_res), mode="bicubic", align_corners=True
        )
        render_image = torch.clamp(render_image, 0, 1)

        render_alpha = torch.nn.functional.interpolate(
            alphas, size=(render_image_res, render_image_res), mode="bicubic", align_corners=True
        )
        render_alpha = torch.clamp(render_alpha, 0, 1)

        bg_colors = torch.tensor(bg_colors, dtype=torch.float32).repeat(3, 1).T

        render_intrinsics = intrinsics * (render_image_res / raw_source_res)
        lmks2d = lmks2d * (render_image_res / raw_source_res)

        pt3d_inputs = get_pt3d_inputs(render_intrinsics, poses)
        gs_inputs = get_gs_inputs(render_intrinsics, poses)

        render_projection, world_view_matrix = gs_inputs["render_projection"], gs_inputs["world_view_matrix"]
        full_proj_matrix, optical_center = gs_inputs["full_proj_matrix"], gs_inputs["optical_center"]
        fov_x, fov_y = gs_inputs["fov_x"], gs_inputs["fov_y"]

        K_pt3d, R_pt3d, T_pt3d = pt3d_inputs["K"], pt3d_inputs["R"], pt3d_inputs["T"]

        return {
            "uid": uid,
            "source_camera": source_camera,
            "source_image": source_image,
            "render_image": render_image,
            "render_bg_colors": bg_colors,
            "render_alpha": render_alpha,
            "lmks2d": lmks2d,
            "render_projection": render_projection,
            "world_view_matrix": world_view_matrix,
            "projection_matrix": full_proj_matrix,
            "optical_center": optical_center,
            "fov_x": fov_x,
            "fov_y": fov_y,
            "K": K_pt3d,
            "R": R_pt3d,
            "T": T_pt3d,
            "mask": None,
        }
