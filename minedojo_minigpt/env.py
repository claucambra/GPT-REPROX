# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import minedojo
import numpy as np
import torch
import argparse

from copy import deepcopy
from typing import Any, Optional

from gymnasium import Env, spaces
from gymnasium.envs.registration import EnvSpec

from .minigpt import MineDojoMiniGPT4, MIN_REWARD, MAX_REWARD, COMPLETION_REWARD
from .gym_compat import convert_minedojo_space

ACTION_USE_IDX = 5

class MineDojoMiniGPT4Env(Env):
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
                 img_only_obs: bool = False,
                 guard_actions: bool = True,
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

        self.max_steps = max_steps
        self.img_only_obs = img_only_obs
        self.guard_actions = guard_actions

        self.__cur_step = 0
        self.__first_reset = True
        self.__reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

        self.__task_string = task_id + " " + str(target_quantity) + " " + target_name
        self.__task_string = self.__task_string + "s" if target_quantity > 1 else self.__task_string
        self.__minigpt = MineDojoMiniGPT4(cmd_args)

        self.__remake_env()

        # Compliance with gymnasium.Env, conversions from MineDojo's gym.Env
        gym_obs_space = self.base_env.observation_space
        if img_only_obs:
            gymnasium_obs_space = convert_minedojo_space(self.base_env.observation_space["rgb"])
        else:
            gymnasium_obs_space = convert_minedojo_space(gym_obs_space)

        self.observation_space = gymnasium_obs_space
        self.action_space = convert_minedojo_space(self.base_env.action_space)
        self.reward_range = (MIN_REWARD, MAX_REWARD)
        self.np_random = self.base_env.unwrapped._rng
        self.spec = EnvSpec("minedojo-minigpt-v0",
                            entry_point="minedojo_minigpt.env:MineDojoMiniGPT4Env",
                            reward_threshold=MAX_REWARD,
                            nondeterministic=False,
                            max_episode_steps=max_steps,
                            order_enforce=True,
                            autoreset=False)


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
            **self.kwargs)

        print("Environment remake: reset all the destroyed blocks!")

    def __guarded_action(self, action: np.ndarray) -> np.ndarray:
        if not self.guard_actions:
            return action

        action_copy = deepcopy(action)
        obs = self.base_env.unwrapped._prev_obs
        equip_list = obs["equipment"]
        allow_use = True if len(equip_list) > 0 else False
        action_copy[ACTION_USE_IDX] = 1 if allow_use and action[ACTION_USE_IDX] == 1 else 0
        return action_copy

    # Duck typing methods for gymnasium.Env
    def close(self):
        if hasattr(self, "base_env"):
            self.base_env.close()

    def output_obs(self, obs: dict) -> dict:
        return obs["rgb"] if self.img_only_obs else obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[dict, dict]:
        if seed is not None:
            self.seed = seed
            self.__remake_env()

        self.__cur_step = 0

        if not self.__first_reset:
            for cmd in self.__reset_cmds:
                self.base_env.unwrapped.execute_cmd(cmd)
        self.__first_reset = False

        self.base_env.reset()  # reset after random teleport
        self.base_env.unwrapped.set_time(6000)
        self.base_env.unwrapped.set_weather("clear")

        # make agent fall onto the ground after teleport
        for _ in range(4):
            no_op_act = self.base_env.action_space.no_op()
            obs, _, _, info = self.base_env.step(no_op_act)

        rgb_image = self.obs_rgb_transpose(obs)
        self.__minigpt.upload_rgb_array(rgb_image)

        if self.save_rgb:
            self.rgb_list = [rgb_image]

        return self.output_obs(obs), info

    def step(self, act: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        act = self.__guarded_action(act)

        obs, _, done, info = self.base_env.step(act)
        rgb_image = self.obs_rgb_transpose(obs)

        self.__minigpt.upload_rgb_array(rgb_image)

        # Reward established as proximity to goal completion
        reward = self.__minigpt.current_reward(self.__task_string)
        completed = reward == COMPLETION_REWARD
        print(f"Received reward {reward} from MiniGPT.")
        assert (reward >= MIN_REWARD and reward <= MAX_REWARD) or completed, \
               f"Received unexpected reward: {reward}"
        reward -= 1 # Penalise for each step taken

        self.__cur_step += 1

        terminated = done or completed or obs["life_stats"]["life"] == 0
        truncated = self.__cur_step >= self.max_steps
        info["TimeLimit.truncated"] = truncated

        if self.save_rgb:
            self.rgb_list.append(rgb_image)

        return self.output_obs(obs), reward, terminated, truncated, info
    
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

