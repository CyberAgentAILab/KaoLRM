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
import os

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from torchvision.utils import draw_keypoints, make_grid
from tqdm.auto import tqdm

from kaolrm.runners import REGISTRY_RUNNERS
from kaolrm.utils.profiler import DummyProfiler

from .base_trainer import Trainer

logger = get_logger(__name__)


@REGISTRY_RUNNERS.register("train.lrm")
class LRMTrainer(Trainer):
    def __init__(self):
        super().__init__()
        assert self.cfg.train.render_size == self.cfg.dataset.render_image_res

        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        self.train_loader, self.val_loader = self._build_dataloader(self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        (
            self.pixel_loss_fn,
            self.perceptual_loss_fn,
            self.dssim_loss_fn,
            self.lmk_loss_fn,
            self.shape_reg_loss_fn,
            self.expr_reg_loss_fn,
            self.mask_loss_fn,
            self.depth_loss_fn,
            self.normal_loss_fn,
        ) = self._build_loss_fn(self.cfg)

    def _build_model(self, cfg):
        assert cfg.experiment.type == "lrm", (
            f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        )
        from kaolrm.models import KaoLRM

        model = KaoLRM(**cfg.model)
        return model

    def _build_optimizer(self, model: nn.Module, cfg):
        """
        Build AdamW optimizer with separate weight-decay groups.

        Following the GPT-style convention (Andrej Karpathy / nanoGPT):
        - Weight decay is applied only to weight matrices (linear, embedding).
        - LayerNorm parameters (both weight and bias) and all bias vectors are
          exempt from weight decay, because regularizing these can destabilize
          training without meaningful regularization benefit.
        """
        decay_params, no_decay_params = [], []

        # LayerNorm weights + biases and all bias terms → no weight decay.
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, "bias") and module.bias is not None:
                no_decay_params.append(module.bias)

        # Everything else (weight matrices) → apply weight decay.
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in model.parameters() if id(p) not in _no_decay_ids]

        # Exclude frozen parameters from both groups.
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # monitor this to make sure we don't miss any parameters
        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        # Optimizer
        opt_groups = [
            {"params": decay_params, "weight_decay": cfg.train.optim.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=cfg.train.optim.lr,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )

        return optimizer

    def _build_scheduler(self, optimizer, cfg):
        local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
        total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
        logger.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
        logger.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
        if cfg.train.scheduler.type == "cosine":
            from kaolrm.utils.scheduler import CosineWarmupScheduler

            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
        return scheduler

    def _build_dataloader(self, cfg):
        # dataset class
        from torch.utils.data._utils.collate import default_collate

        from kaolrm.datasets import MixerDataset

        def _smart_none_collate_fn(batch):
            collated = {}
            keys = batch[0].keys()
            for key in keys:
                values = [item[key] for item in batch]
                if any(v is None for v in values):
                    collated[key] = None
                else:
                    collated[key] = default_collate(values)
            return collated

        # build dataset
        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res=cfg.dataset.render_image_res,
            source_image_res=cfg.dataset.source_image_res,
            normalize_camera=cfg.dataset.normalize_camera,
            normed_dist_to_center=cfg.dataset.normed_dist_to_center,
            synth_focal=cfg.dataset.synth_focal,
        )
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res=cfg.dataset.render_image_res,
            source_image_res=cfg.dataset.source_image_res,
            normalize_camera=cfg.dataset.normalize_camera,
            normed_dist_to_center=cfg.dataset.normed_dist_to_center,
            synth_focal=cfg.dataset.synth_focal,
        )

        # build data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.dataset.num_train_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=True,
            collate_fn=_smart_none_collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.dataset.num_val_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=False,
            collate_fn=_smart_none_collate_fn,
        )

        return train_loader, val_loader

    def _build_loss_fn(self, cfg):
        from kaolrm.losses import DSSIMLoss, GeometryLoss, L2RegLoss, LandmarkLoss, LPIPSLoss, PixelLoss

        pixel_loss_fn = PixelLoss()
        lmk_loss_fn = LandmarkLoss()

        with self.accelerator.main_process_first():
            perceptual_loss_fn = LPIPSLoss(device=self.device, prefech=True)

        dssim_loss_fn = DSSIMLoss()

        shape_reg_loss_fn = L2RegLoss()
        expr_reg_loss_fn = L2RegLoss()

        mask_loss_fn = nn.L1Loss()

        depth_loss_fn = GeometryLoss()
        normal_loss_fn = GeometryLoss()

        return (
            pixel_loss_fn,
            perceptual_loss_fn,
            dssim_loss_fn,
            lmk_loss_fn,
            shape_reg_loss_fn,
            expr_reg_loss_fn,
            mask_loss_fn,
            depth_loss_fn,
            normal_loss_fn,
        )

    def register_hooks(self):
        pass

    def forward_loss_local_step(self, data):

        source_camera = data["source_camera"]
        source_image = data["source_image"]
        render_image = data["render_image"]
        render_bg_colors = data["render_bg_colors"]
        world_view_matrix = data["world_view_matrix"]
        projection_matrix = data["projection_matrix"]
        optical_center = data["optical_center"]
        fov_x, fov_y = data["fov_x"], data["fov_y"]
        render_alpha = data["render_alpha"]
        render_projection = data["render_projection"]

        K, R, T = data["K"], data["R"], data["T"]

        render_size, focal, num_sampling = self.cfg.train.render_size, self.cfg.train.synth_focal, self.cfg.train.num_sampling

        # forward
        outputs = self.model(
            image=source_image,
            source_camera=source_camera,
            world_view_matrix=world_view_matrix,
            projection_matrix=projection_matrix,
            optical_center=optical_center,
            render_bg_colors=render_bg_colors,
            fov_x=fov_x,
            fov_y=fov_y,
            K=K,
            R=R,
            T=T,
            # inference-time hyper-params
            render_size=render_size,
            focal=focal,
            num_sampling=num_sampling,
        )

        render_facial_mask = data["mask"] if data["mask"] is not None else outputs["flame_mask"]

        # loss calculation
        loss = 0.0
        loss_pixel = None
        loss_perceptual = None
        loss_dssim = None
        loss_lmk = None
        loss_beta = None
        loss_psi = None
        loss_mask = None
        loss_depth = None
        loss_normal = None

        # warm-up epochs
        if self.current_epoch < 5:  # self.current_epoch < 3: # epoch=0,1,2
            pixel_weight = 0.0
            perceptual_weight = 0.0
            dssim_weight = 0.0
            lmk_weight = 1.0
            shape_reg_weight = 1e-02
            expr_reg_weight = 1e-02
            mask_weight = 0.1
            depth_weight = 0.0
            normal_weight = 0.0

        else:
            pixel_weight = self.cfg.train.loss.pixel_weight
            perceptual_weight = self.cfg.train.loss.perceptual_weight
            dssim_weight = self.cfg.train.loss.dssim_weight
            lmk_weight = self.cfg.train.loss.lmk_weight
            shape_reg_weight = self.cfg.train.loss.shape_reg_weight
            expr_reg_weight = self.cfg.train.loss.expr_reg_weight
            mask_weight = self.cfg.train.loss.mask_weight
            depth_weight = self.cfg.train.loss.depth_weight
            normal_weight = self.cfg.train.loss.normal_weight

        if pixel_weight > 0.0:
            loss_pixel = self.pixel_loss_fn(outputs["rgb"], render_image, render_facial_mask, 0.7)
            loss += loss_pixel * pixel_weight
        if perceptual_weight > 0.0:
            loss_perceptual = self.perceptual_loss_fn(outputs["rgb"], render_image)
            loss += loss_perceptual * perceptual_weight
        if dssim_weight > 0.0:
            loss_dssim = self.dssim_loss_fn(outputs["rgb"], render_image)
            loss += loss_dssim * dssim_weight
        if lmk_weight > 0.0:
            projected_lmks2d = _project_to_views(outputs["lmk3d"], render_projection)
            loss_lmk = self.lmk_loss_fn(projected_lmks2d, data["lmks2d"])
            loss += loss_lmk * lmk_weight
        if shape_reg_weight > 0.0:
            loss_beta = self.shape_reg_loss_fn(outputs["beta"])
            loss += loss_beta * shape_reg_weight
        if expr_reg_weight > 0.0:
            loss_psi = self.expr_reg_loss_fn(outputs["psi"])
            loss += loss_psi * expr_reg_weight
        if mask_weight > 0.0:
            loss_mask = self.mask_loss_fn(outputs["alpha"], render_alpha)
            loss += loss_mask * mask_weight
        if depth_weight > 0.0:
            loss_depth = self.depth_loss_fn(outputs["gs_depth"], outputs["flame_depth"], mask=render_facial_mask)
            loss += loss_depth * depth_weight
        if normal_weight > 0.0:
            loss_normal = self.normal_loss_fn(outputs["gs_normal"], outputs["flame_normal"], mask=render_facial_mask)
            loss += loss_normal * normal_weight

        return (
            outputs,
            loss,
            loss_pixel,
            loss_perceptual,
            loss_dssim,
            loss_lmk,
            loss_beta,
            loss_psi,
            loss_mask,
            loss_depth,
            loss_normal,
        )

    def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile):
        self.model.train()

        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        for data in loader:
            logger.debug(f"======== Starting global step {self.global_step} ========")
            with self.accelerator.accumulate(self.model):
                # forward to loss
                (
                    outs,
                    loss,
                    loss_pixel,
                    loss_perceptual,
                    loss_dssim,
                    loss_lmk,
                    loss_beta,
                    loss_psi,
                    loss_mask,
                    loss_depth,
                    loss_normal,
                ) = self.forward_loss_local_step(data)

                # backward
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # track local losses
                local_step_losses.append(
                    torch.stack(
                        [
                            _loss.detach() if _loss is not None else torch.tensor(float("nan"), device=self.device)
                            for _loss in [
                                loss,
                                loss_pixel,
                                loss_perceptual,
                                loss_dssim,
                                loss_lmk,
                                loss_beta,
                                loss_psi,
                                loss_mask,
                                loss_depth,
                                loss_normal,
                            ]
                        ]
                    )
                )

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                logger.debug("======== Scheduler step ========")
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
                (
                    loss,
                    loss_pixel,
                    loss_perceptual,
                    loss_dssim,
                    loss_lmk,
                    loss_beta,
                    loss_psi,
                    loss_mask,
                    loss_depth,
                    loss_normal,
                ) = global_step_loss.unbind()
                loss_kwargs = {
                    "loss": loss.item(),
                    "loss_lmk": loss_lmk.item(),
                    "loss_beta": loss_beta.item(),
                    "loss_psi": loss_psi.item(),
                    "loss_pixel": loss_pixel.item(),
                    "loss_perceptual": loss_perceptual.item(),
                    "loss_dssim": loss_dssim.item(),
                    "loss_mask": loss_mask.item(),
                    "loss_depth": loss_depth.item(),
                    "loss_normal": loss_normal.item(),
                }
                self.log_scalar_kwargs(step=self.global_step, split="train", **loss_kwargs)
                self.log_optimizer(step=self.global_step, attrs=["lr"], group_ids=[0, 1])
                local_step_losses = []
                global_step_losses.append(global_step_loss)

                # manage display
                pbar.update(1)
                description = {
                    **loss_kwargs,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                description = "[TRAIN STEP]" + ", ".join(
                    f"{k}={tqdm.format_num(v)}" for k, v in description.items() if not math.isnan(v)
                )

                # periodic actions
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.cfg.val.global_step_period == 0:
                    self.evaluate()
                    self.model.train()
                if self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:
                    self.log_xyz_monitor(
                        epoch=self.global_step,
                        split="train_vertices",
                        xyz=outs["vertices"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        projection=data["render_projection"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        canvas_image=data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )
                    self.log_xyz_monitor(
                        epoch=self.global_step,
                        split="train_lmks",
                        xyz=outs["lmk3d"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        projection=data["render_projection"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        canvas_image=data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        uv=data["lmks2d"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )
                    self.log_image_monitor(
                        step=self.global_step,
                        split="train_rgb",
                        renders=outs["rgb"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        gts=data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )
                    self.log_image_monitor(
                        step=self.global_step,
                        split="train_alpha",
                        renders=outs["alpha"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        gts=data["render_alpha"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )
                    self.log_image_monitor(
                        step=self.global_step,
                        split="train_normal",
                        renders=outs["gs_normal"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        gts=outs["flame_normal"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    self.log_image_monitor(
                        step=self.global_step,
                        split="train_depth",
                        renders=outs["gs_depth"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        gts=outs["flame_depth"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        normalize=True,
                    )

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        (
            epoch_loss,
            epoch_loss_pixel,
            epoch_loss_perceptual,
            epoch_loss_dssim,
            epoch_loss_lmk,
            epoch_loss_beta,
            epoch_loss_psi,
            epoch_loss_mask,
            epoch_loss_depth,
            epoch_loss_normal,
        ) = epoch_losses.unbind()
        epoch_loss_dict = {
            "loss": epoch_loss.item(),
            "loss_lmk": epoch_loss_lmk.item(),
            "loss_beta": epoch_loss_beta.item(),
            "loss_psi": epoch_loss_psi.item(),
            "loss_pixel": epoch_loss_pixel.item(),
            "loss_perceptual": epoch_loss_perceptual.item(),
            "loss_dssim": epoch_loss_dssim.item(),
            "loss_mask": epoch_loss_mask.item(),
            "loss_depth": epoch_loss_depth.item(),
            "loss_normal": epoch_loss_normal.item(),
        }
        self.log_scalar_kwargs(
            epoch=self.current_epoch,
            split="train",
            **epoch_loss_dict,
        )
        logger.info(
            f"[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: "
            + ", ".join(f"{k}={tqdm.format_num(v)}" for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )

    def train(self):
        if self.cfg.train.find_unused_parameters:
            print("===== FREEZING Image2Triplane =====")
            # DO FREEZING HERE
            for submodule in [self.model.module.encoder, self.model.module.upsampler]:
                for param in submodule.parameters():
                    param.requires_grad = False

        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        with tqdm(
            range(0, self.N_max_global_steps),
            initial=self.global_step,
            disable=(not self.accelerator.is_main_process),
        ) as pbar:
            profiler = (
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(
                        wait=10,
                        warmup=10,
                        active=100,
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        os.path.join(
                            self.cfg.logger.tracker_root,
                            self.cfg.experiment.parent,
                            self.cfg.experiment.child,
                        )
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                if self.cfg.logger.enable_profiler
                else DummyProfiler()
            )

            with profiler:
                self.optimizer.zero_grad()
                for _ in range(self.current_epoch, self.cfg.train.epochs):
                    loader = skipped_loader or self.train_loader
                    skipped_loader = None
                    self.train_epoch(pbar=pbar, loader=loader, profiler=profiler)
                    if self.accelerator.check_trigger():
                        break

            logger.info(f"======== Training finished at global step {self.global_step} ========")

            # final checkpoint and evaluation
            self.save_checkpoint()
            self.evaluate()

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self, epoch: int = None):
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
        running_losses = []
        sample_data, sample_outs = None, None

        for data in tqdm(self.val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):
            if len(running_losses) >= max_val_batches:
                logger.info(f"======== Early stop validation at {len(running_losses)} batches ========")
                break

            (
                outs,
                loss,
                loss_pixel,
                loss_perceptual,
                loss_dssim,
                loss_lmk,
                loss_beta,
                loss_psi,
                loss_mask,
                loss_depth,
                loss_normal,
            ) = self.forward_loss_local_step(data)
            sample_data, sample_outs = data, outs

            running_losses.append(
                torch.stack(
                    [
                        _loss if _loss is not None else torch.tensor(float("nan"), device=self.device)
                        for _loss in [
                            loss,
                            loss_pixel,
                            loss_perceptual,
                            loss_dssim,
                            loss_lmk,
                            loss_beta,
                            loss_psi,
                            loss_mask,
                            loss_depth,
                            loss_normal,
                        ]
                    ]
                )
            )

        total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()

        (
            total_loss,
            total_loss_pixel,
            total_loss_perceptual,
            total_loss_dssim,
            total_loss_lmk,
            total_loss_beta,
            total_loss_psi,
            total_loss_mask,
            total_loss_depth,
            total_loss_normal,
        ) = total_losses.unbind()
        total_loss_dict = {
            "loss": total_loss.item(),
            "loss_lmk": total_loss_lmk.item(),
            "loss_beta": total_loss_beta.item(),
            "loss_psi": total_loss_psi.item(),
            "loss_pixel": total_loss_pixel.item(),
            "loss_perceptual": total_loss_perceptual.item(),
            "loss_dssim": total_loss_dssim.item(),
            "loss_mask": total_loss_mask.item(),
            "loss_depth": total_loss_depth.item(),
            "loss_normal": total_loss_normal.item(),
        }

        if epoch is not None:
            self.log_scalar_kwargs(
                epoch=epoch,
                split="val",
                **total_loss_dict,
            )
            logger.info(
                f"[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: "
                + ", ".join(f"{k}={tqdm.format_num(v)}" for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self.log_image_monitor(
                epoch=epoch,
                split="val_rgb",
                renders=sample_outs["rgb"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )
            self.log_image_monitor(
                epoch=epoch,
                split="val_alpha",
                renders=sample_outs["alpha"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data["render_alpha"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )
            self.log_image_monitor(
                epoch=epoch,
                split="val_depth",
                renders=sample_outs["gs_depth"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_outs["flame_depth"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                normalize=True,
            )
            self.log_image_monitor(
                epoch=epoch,
                split="val_normal",
                renders=sample_outs["gs_normal"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_outs["flame_normal"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                normalize=True,
                value_range=(-1, 1),
            )
            self.log_xyz_monitor(
                epoch=epoch,
                split="val_vertices",
                xyz=sample_outs["vertices"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                projection=sample_data["render_projection"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                canvas_image=sample_data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )
            self.log_xyz_monitor(
                epoch=epoch,
                split="val_lmks",
                xyz=sample_outs["lmk3d"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                projection=sample_data["render_projection"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                canvas_image=sample_data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                uv=sample_data["lmks2d"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )

        else:
            self.log_scalar_kwargs(
                step=self.global_step,
                split="val",
                **total_loss_dict,
            )
            logger.info(
                f"[VAL STEP] {self.global_step}/{self.N_max_global_steps}: "
                + ", ".join(f"{k}={tqdm.format_num(v)}" for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self.log_image_monitor(
                step=self.global_step,
                split="val_rgb",
                renders=sample_outs["rgb"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )
            self.log_image_monitor(
                step=self.global_step,
                split="val_alpha",
                renders=sample_outs["alpha"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data["render_alpha"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )
            self.log_image_monitor(
                epoch=self.global_step,
                split="val_depth",
                renders=sample_outs["gs_depth"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_outs["flame_depth"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                normalize=True,
            )
            self.log_image_monitor(
                epoch=self.global_step,
                split="val_normal",
                renders=sample_outs["gs_normal"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_outs["flame_normal"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                normalize=True,
                value_range=(-1, 1),
            )
            self.log_xyz_monitor(
                epoch=self.global_step,
                split="val_lmks",
                xyz=sample_outs["lmk3d"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                projection=sample_data["render_projection"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                canvas_image=sample_data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                uv=sample_data["lmks2d"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )
            self.log_xyz_monitor(
                epoch=self.global_step,
                split="val_vertices",
                xyz=sample_outs["vertices"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                projection=sample_data["render_projection"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
                canvas_image=sample_data["render_image"][: self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )

    @Trainer.control("on_main_process")
    def log_image_monitor(
        self,
        epoch: int = None,
        step: int = None,
        split: str = None,
        renders: torch.Tensor = None,
        gts: torch.Tensor = None,
        normalize: bool = False,
        value_range: tuple = None,
    ):
        M = renders.shape[1]  # V: number of views per sample
        merged = torch.stack([renders, gts], dim=1)[0].view(-1, *renders.shape[2:])
        renders, gts = renders.view(-1, *renders.shape[2:]), gts.view(-1, *gts.shape[2:])
        renders, gts, merged = (
            make_grid(renders, nrow=M, normalize=normalize, value_range=value_range),
            make_grid(gts, nrow=M, normalize=normalize, value_range=value_range),
            make_grid(merged, nrow=M, normalize=normalize, value_range=value_range),
        )
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f"/{split}" if split else ""
        self.log_images(
            {
                f"Images_split{split}/rendered": renders.unsqueeze(0),
                f"Images_split{split}/gt": gts.unsqueeze(0),
                f"Images_merged{split}": merged.unsqueeze(0),
            },
            log_progress,
        )

    @Trainer.control("on_main_process")
    def log_xyz_monitor(
        self,
        epoch: int = None,
        step: int = None,
        split: str = None,
        xyz: torch.Tensor = None,
        projection: torch.Tensor = None,
        uv: torch.Tensor = None,
        canvas_image: torch.Tensor = None,
    ):
        canvas_image = (canvas_image * 255).byte()
        marked_image = _project_to_multi_view(xyz, projection, canvas_image)
        if uv is not None:
            marked_image = _draw_kpts_to_canvas(uv, marked_image, "cyan")
        M = canvas_image.shape[1]
        marked_image = marked_image.view(-1, *marked_image.shape[2:])
        grid_image = make_grid(marked_image, nrow=M)
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f"/{split}" if split else ""
        self.log_images(
            {
                f"Images_split{split}": grid_image.unsqueeze(0),
            },
            log_progress,
        )


# called by _project_to_multi_view
def _project_to_views(xyz, render_projection):
    """
    Batched IO, each batch V images, differentiable
    xyz: (B, 68 or 5023, 3)
    full_projection: (B, V, 3, 4)
    """
    B, N, _ = xyz.shape  # Get batch size
    V = render_projection.shape[1]  # Get number of views

    lmks_h = torch.cat([xyz, torch.ones(B, N, 1, device=xyz.device)], dim=-1)  # (B, N, 4)
    lmks_h = lmks_h.unsqueeze(1).expand(-1, V, -1, -1)  # (B, V, N, 4)
    lmks_image_h = torch.matmul(render_projection, lmks_h.permute(0, 1, 3, 2))  # (B, V, 3, N)
    lmks_image_h = lmks_image_h.permute(0, 1, 3, 2)  # (B, V, N, 3)
    lmks_image = lmks_image_h[:, :, :, :2] / lmks_image_h[:, :, :, 2:3]  # (B, V, N, 2)
    return lmks_image


def _project_to_multi_view(xyz, full_projection, canvas_image):
    """
    Batched IO, multiple views supported
    xyz: (B, N, 3)  # 3D landmarks
    full_projection: (B, V, 3, 4)  # Projection matrices for multiple views
    canvas_image: (B, V, 3, H, W)  # Input images for different views
    """
    B, V, _, _ = full_projection.shape  # Number of views

    lmks_image = _project_to_views(xyz, full_projection)
    marked_images = _draw_kpts_to_canvas(lmks_image, canvas_image)

    return marked_images


def _draw_kpts_to_canvas(kpts2d, canvas_image, colors="red"):
    B, V, _, H, W = canvas_image.shape

    # Convert to pixel coordinates
    lmks_image = kpts2d.round().long()

    # Clamp values to ensure they are within image bounds
    lmks_image[..., 0] = lmks_image[..., 0].clamp(0, W - 1)
    lmks_image[..., 1] = lmks_image[..., 1].clamp(0, H - 1)

    output_images = []

    # Iterate over batch and views
    for b in range(B):
        view_images = []
        for v in range(V):
            img_with_lmks = draw_keypoints(
                canvas_image[b, v].clone(),  # Image: (3, H, W)
                lmks_image[b, v].unsqueeze(0),  # Landmark locations (1, N, 2)
                colors=colors,
                radius=1,
            )
            view_images.append(img_with_lmks)
        output_images.append(torch.stack(view_images, dim=0))  # Stack per-view images

    marked_images = torch.stack(output_images, dim=0)  # Stack batch images back together
    return marked_images  # Output shape: (B, V, 3, H, W)
