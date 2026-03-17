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

import argparse
import os

import numpy as np
import torch
import trimesh
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from kaolrm.datasets.cam_utils import (
    build_camera_principle,
    create_intrinsics,
    get_gs_inputs,
    get_pt3d_inputs,
    surrounding_views_linspace,
)
from kaolrm.runners import REGISTRY_RUNNERS
from kaolrm.utils.hf_hub import wrap_model_hub
from kaolrm.utils.logging import configure_logger
from kaolrm.utils.video import images_to_video

from .base_inferrer import Inferrer

logger = get_logger(__name__)


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image_res
        _relative_path = os.path.join(
            cfg_train.experiment.parent, cfg_train.experiment.child, os.path.basename(cli_cfg.model_name).split("_")[-1]
        )
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)
        cfg.mesh_dump = os.path.join("exps", "meshes", _relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("video_dump", os.path.join("dumps", cli_cfg.model_name, "videos"))
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get("APP_ENABLED", None):
        assert cfg.image_input is not None, "image_input is required"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg


@REGISTRY_RUNNERS.register("infer.lrm")
class LRMInferrer(Inferrer):
    EXP_TYPE: str = "lrm"

    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()
        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )

        self.model = self._build_model(self.cfg).to(self.device)

    def _build_model(self, cfg):
        from kaolrm.models import model_dict

        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def _default_source_camera(self, dist_to_center: float = 2.0, device: torch.device = torch.device("cpu")):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, -dist_to_center],
                    [0, 1, 0, 0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(1, 1)

    def _default_render_cameras(
        self, n_views: int, render_size: int, synth_focal: float, device: torch.device = torch.device("cpu")
    ):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
        render_camera_intrinsics = create_intrinsics(
            f=synth_focal,
            c=0.5,
            device=device,
        )
        render_camera_intrinsics *= render_size

        gs_inputs = get_gs_inputs(render_camera_intrinsics, render_camera_extrinsics)
        pt3d_inputs = get_pt3d_inputs(render_camera_intrinsics, render_camera_extrinsics)
        return gs_inputs, pt3d_inputs

    def infer_planes(self, image: torch.Tensor, source_cam_dist: float):
        N = image.shape[0]
        source_camera = self._default_source_camera(dist_to_center=source_cam_dist, device=self.device)
        planes = self.model.forward_planes(image, source_camera)
        assert N == planes.shape[0]
        return planes

    def infer_results(
        self,
        planes: torch.Tensor,
        frame_size: int,
        render_size: int,
        render_views: int,
        num_sampling: int,
        synth_focal: float,
        skip_video: bool,
    ):

        decoded_params = self.model.flame_decoder(planes)
        fix_z = True if synth_focal > 1.0 else False
        vertices, lmks, vertices_render = self.model.flame2mesh(decoded_params, num_sampling=num_sampling, fix_z_trans=fix_z)

        N = planes.shape[0]  # always 1 in the current version
        gs_inputs, pt3d_inputs = self._default_render_cameras(
            n_views=int(render_views / frame_size), device=self.device, render_size=render_size, synth_focal=synth_focal
        )

        faces = self.model.flame_model.faces_tensor.repeat(N, 1, 1)

        if skip_video:
            return {
                "vertices": vertices,
                "faces": faces,
                "lmks": lmks,
                "params": decoded_params,
            }

        K, R, T = (
            pt3d_inputs["K"].unsqueeze(0).to(planes.device),
            pt3d_inputs["R"].unsqueeze(0).to(planes.device),
            pt3d_inputs["T"].unsqueeze(0).to(planes.device),
        )

        flame_normal = self.model.diffrenderer.render_normal(vertices, faces, K, R, T, render_size)["normal"]
        flame_phong = self.model.diffrenderer.render_phong(vertices, faces, K, R, T, render_size)["rgba"]

        flame_mask = self.model.diffrenderer.render_normal_wFM(vertices, faces, K, R, T, render_size)["facial_mask"]

        gaussians = self.model.gaussian_decoder(vertices_render, planes, scale_coords=synth_focal)
        gaussians = torch.cat((vertices_render, gaussians), dim=-1)

        if synth_focal > 1.0:
            # if monocular data (weak perspective), do backface culling
            gaussians = self.model.backface_culling(vertices_render, gaussians)

        world_view_mat = gs_inputs["world_view_matrix"].unsqueeze(0).to(planes.device)
        full_proj_mat = gs_inputs["full_proj_matrix"].unsqueeze(0).to(planes.device)
        optical_center = gs_inputs["optical_center"].unsqueeze(0).to(planes.device)
        fov_x, fov_y = gs_inputs["fov_x"].unsqueeze(0).to(planes.device), gs_inputs["fov_y"].unsqueeze(0).to(planes.device)
        bg_color = torch.ones([1, render_views, 3], device=planes.device)

        frames = self.model.synthesizer.render(
            gaussians,
            cam_view=world_view_mat,
            cam_view_proj=full_proj_mat,
            cam_pos=optical_center,
            bg_color=bg_color,
            fov_x=fov_x,
            fov_y=fov_y,
            render_size=render_size,
        )

        return {
            "frames": frames,
            "vertices": vertices,
            "faces": faces,
            "lmks": lmks,
            "params": decoded_params,
            "phong": flame_phong,
            "normal": flame_normal,
            "mask": flame_mask,
        }

    def infer_single(self, image_path: str, source_cam_dist: float, dump_video_path: str, dump_mesh_path: str):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_views = self.cfg.render_views
        render_fps = self.cfg.render_fps
        frame_size = self.cfg.frame_size
        source_cam_dist = self.cfg.source_cam_dist if source_cam_dist is None else source_cam_dist
        synth_focal = self.cfg.synth_focal
        num_sampling = self.cfg.num_sampling
        skip_video = self.cfg.skip_video

        # prepare image: [1, C_img, H_img, W_img], 0-1 scale
        image_raw = torch.from_numpy(np.array(Image.open(image_path))).to(self.device)
        image_raw = image_raw.permute(2, 0, 1).unsqueeze(0) / 255.0
        if image_raw.shape[1] == 4:  # RGBA
            image_raw = image_raw[:, :3, ...] * image_raw[:, 3:, ...] + (1 - image_raw[:, 3:, ...])
        image = torch.nn.functional.interpolate(image_raw, size=(source_size, source_size), mode="bicubic", align_corners=True)
        image = torch.clamp(image, 0, 1)

        with torch.no_grad():
            planes = self.infer_planes(image, source_cam_dist=source_cam_dist)

            results_dict = {
                "image": torch.nn.functional.interpolate(
                    image_raw, size=(render_size, render_size), mode="bicubic", align_corners=True
                )
            }
            results = self.infer_results(
                planes,
                frame_size=frame_size,
                render_size=render_size,
                render_views=render_views,
                synth_focal=synth_focal,
                num_sampling=num_sampling,
                skip_video=skip_video,
            )
            results_dict.update(results)

            self.dump_results(results_dict, dump_video_path, dump_mesh_path, render_fps, synth_focal, skip_video)

    def infer(self):
        image_paths = []
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths.append(self.cfg.image_input)
        else:
            omit_prefix = self.cfg.image_input
            for root, dirs, files in os.walk(self.cfg.image_input):
                for file in files:
                    if file.endswith(".png") and "depth" not in file and "normal" not in file:
                        image_paths.append(os.path.join(root, file))
            image_paths.sort()

        # alloc to each DDP worker
        image_paths = image_paths[self.accelerator.process_index :: self.accelerator.num_processes]

        for image_path in tqdm(image_paths, disable=not self.accelerator.is_local_main_process):
            # prepare dump paths
            image_name = os.path.basename(image_path)
            uid = image_name.split(".")[0]
            subdir_path = os.path.dirname(image_path).replace(omit_prefix, "")
            subdir_path = subdir_path[1:] if subdir_path.startswith("/") else subdir_path
            dump_video_path = os.path.join(
                self.cfg.video_dump,
                subdir_path,
                f"{uid}.gif",
            )
            dump_mesh_path = os.path.join(
                self.cfg.mesh_dump,
                subdir_path,
                f"{uid}.ply",
            )

            self.infer_single(
                image_path,
                source_cam_dist=None,
                dump_video_path=dump_video_path,
                dump_mesh_path=dump_mesh_path,
            )

    def dump_results(self, results, dump_video_path, dump_mesh_path, render_fps, synth_focal, skip_video=False):

        vertices, faces, decoded_params, lmks = results["vertices"], results["faces"], results["params"], results["lmks"]

        # to align with gt mesh scale
        mesh = trimesh.Trimesh(vertices=synth_focal / 0.75 * vertices[0].cpu().numpy(), faces=faces[0].cpu().numpy())
        lmks = synth_focal / 0.75 * lmks.cpu().numpy()[0]

        os.makedirs(os.path.dirname(dump_mesh_path), exist_ok=True)
        mesh.export(dump_mesh_path)
        shape_np = decoded_params["shape"].detach().cpu().numpy()[0]
        expression_np = decoded_params["expression"].detach().cpu().numpy()[0]
        pose_np = decoded_params["pose"].detach().cpu().numpy()[0]
        np.save(dump_mesh_path.replace(".ply", "_shape.npy"), shape_np)
        np.save(dump_mesh_path.replace(".ply", "_expression.npy"), expression_np)
        np.save(dump_mesh_path.replace(".ply", "_pose.npy"), pose_np)
        np.save(dump_mesh_path.replace(".ply", "_lmks.npy"), lmks)

        if skip_video:
            return

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        frames = results["frames"]
        images_to_video(
            images=frames["image"][0],
            output_path=dump_video_path,
            fps=render_fps,
            gradio_codec=self.cfg.app_enabled,
        )
        save_images(dump_video_path, results)


def save_images(save_path, results):
    input_image = results["image"][0].permute(1, 2, 0)
    normal = results["normal"][0, 0].permute(1, 2, 0)
    phong = results["phong"]
    rendering = results["frames"]["image"][0, 0].permute(1, 2, 0)
    _mask = results["mask"][0, 0]

    input_image = input_image.detach().cpu().numpy() * 255.0

    phong_image = phong[0, 0, :3].permute(1, 2, 0).detach().cpu().numpy() * 255.0
    alpha_image = phong[0, 0, -1].detach().cpu().numpy()

    flame_normal = (normal.detach().cpu().numpy() + 1.0) / 2.0 * 255.0
    flame_normal[alpha_image < 0.001, :] = 255

    rendering_image = rendering.detach().cpu().numpy() * 255

    combined = np.concatenate(
        [
            input_image.astype(np.uint8),
            rendering_image.astype(np.uint8),
            flame_normal.astype(np.uint8),
            phong_image.astype(np.uint8),
        ],
        axis=1,
    )
    Image.fromarray(combined).save(save_path.replace(".gif", "_vis.png"))
