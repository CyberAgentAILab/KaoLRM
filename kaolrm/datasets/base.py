# Copyright (c) 2023-2024, Zexin He
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


import json
from abc import ABC, abstractmethod

import numpy as np
import torch
from megfile import smart_open
from PIL import Image


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root_dir: str, meta_path: str):
        super().__init__()
        self.root_dir = root_dir
        self.uids = self._load_uids(meta_path)

    def __len__(self):
        return len(self.uids)

    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            print(f"[DEBUG-DATASET] Error when loading {self.uids[idx]}")
            raise e

    @staticmethod
    def _load_uids(meta_path: str):
        if meta_path is None:
            uids = []
        else:
            with open(meta_path) as f:
                uids = json.load(f)
        return uids

    @staticmethod
    def _load_rgba_image(file_path, bg_color: float = 1.0):
        """Load and blend RGBA image to RGB with certain background, 0-1 scaled"""
        rgba = np.array(Image.open(smart_open(file_path, "rb")))
        rgba = torch.from_numpy(rgba).float() / 255.0
        rgba = rgba.permute(2, 0, 1).unsqueeze(0)
        alpha = rgba[:, 3:4, :, :]
        rgb = rgba[:, :3, :, :] * rgba[:, 3:4, :, :] + bg_color * (1 - rgba[:, 3:, :, :])
        return rgb, alpha
