# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import minedojo
import numpy as np
import torch
import argparse

from typing import Any, Optional

from .minigpt import MineDojoMiniGPT4


class MineDojoMiniGPT4Env:
    def __init__(self,
                 cmd_args: Optional[argparse.Namespace] = None,
                 image_size: tuple[int, int] = (160, 256),
                 seed: int = 0,
                 biome: str = "plains",
                 device: Optional[torch.device] = None,
                 task_id: str = "harvest",
                 target_name: str = "log",
                 target_quantity: int = 1,
                 save_rgb: bool = False,
                 max_steps: int = 3000,
                 render_mode: Optional[str] = "human",
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

        self.render_mode = render_mode
        self.save_rgb = save_rgb

        self.max_step = max_steps

        self.__cur_step = 0
        self.__minigpt = MineDojoMiniGPT4(cmd_args)

        self.__remake_env()



    def __del__(self):
        self.close()

    def __remake_env(self):
        self.close()

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


    def reset(self) -> tuple[dict, dict]:
    def close(self):
        if hasattr(self, "base_env"):
            self.base_env.close()


        self.__cur_step = 0

        self.base_env.reset(move_flag=True)  # reset after random teleport
        self.base_env.unwrapped.set_time(6000)
        self.base_env.unwrapped.set_weather("clear")

        # make agent fall onto the ground after teleport
        for _ in range(4):
            no_op_act = self.base_env.action_space.no_op()
            obs, _, _, info = self.base_env.step(no_op_act)

        rgb_image = self.obs_rgb_transpose(obs)
        self.__minigpt.upload_img(rgb_image)

        if self.save_rgb:
            self.rgb_list = [rgb_image]

        return obs, info


    def step(self, act: dict):
        obs, _, done, info = self.base_env.step(act)

        rgb_image = self.obs_rgb_transpose(obs)
        self.__minigpt.upload_img(rgb_image)

        # Reward established as proximity to goal completion, 0 - 100
        reward = self.__minigpt.current_reward(obs)
        assert reward >= 0 and reward <= 100

        self.__cur_step += 1

        done = done or \
            obs["life_stats"]["life"] == 0 or \
                self.__cur_step >= self.max_step or \
                    reward == 100

        if self.save_rgb:
            self.rgb_list.append(rgb_image)

        return obs, reward, done, info
    

    def render(self):
        self.base_env.render(self.render_mode)

    
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


        
