# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import minedojo
import numpy as np
import torch
import argparse

from .minigpt import MineDojoMiniGPT4


class MineDojoMiniGPT4Env:
    def __init__(self,
                 cmd_args: argparse.Namespace = None,
                 image_size: tuple[int, int] = (160, 256),
                 seed: int = 0,
                 biome: str = "plains",
                 device: torch.device = None,
                 task_id: str = "harvest",
                 target_name: str = "log",
                 target_quantity: int = 1,
                 save_rgb: bool = False,
                 max_steps: int = 3000,
                 **kwargs):

        # Essential arguments to create base env
        self.task_id = task_id
        self.target_name = target_name
        self.target_quantity = target_quantity
        self.image_size = image_size
        self.biome = biome
        self.seed = seed
        self.image_size = image_size
        self.kwargs = kwargs  # should contain: initial inventory, initial mobs

        self.device = device
        self.save_rgb = save_rgb

        self.max_step = max_steps

        self.__cur_step = 0
        self.__minigpt = MineDojoMiniGPT4(cmd_args)

        self.remake_env()


    def __del__(self):
        if hasattr(self, "base_env"):
            self.base_env.close()


    def remake_env(self):
        if hasattr(self, "base_env"):
            self.base_env.close()

        if self.target_name.endswith("_nearby"):
            self.target_item_name = self.target_name[:-7]
        else:
            self.target_item_name = self.target_name

        self.base_env = minedojo.make(
            task_id=self.task_id,
            image_size=self.image_size,
            target_names=self.target_item_name,
            target_quantities=self.target_quantity,
            world_seed=self.seed,
            seed=self.seed,
            specified_biome=self.biome,
            fast_reset=True,
            fast_reset_random_teleport_range_low=0,
            fast_reset_random_teleport_range_high=500,
            **self.kwargs)

        print("Environment remake: reset all the destroyed blocks!")


    @staticmethod
    def obs_rgb_transpose(obs: dict) -> np.ndarray:
        """
        MineDojo returns observation in (C, H, W) format.

        Most libraries (Pillow, imageio) expect (H, W, C) format.

        MiniGPT's Blip2EvalProcessor also expects (H, W, C) format as input,
        despite it internally using torchvision's ToTensor transform when using
        the eval vision processor. Torchvision's ToTensor converts (H, W, C) to 
        (C, H, W).
        """
        return np.transpose(obs["rgb"], [1, 2, 0]).astype(np.uint8)


        