# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import os
import torch

from minedojo_minigpt.env import MineDojoMiniGPT4Env
from .task_config import setup_task_conf


def setup_minedojo_gpt_env(args: argparse.Namespace,
                           device: torch.device,
                           img_only: bool = False) -> MineDojoMiniGPT4Env:

    task_conf = setup_task_conf(args.task)

    if (args.max_episode_steps != task_conf["max_steps"]):
        task_conf["max_steps"] = args.max_episode_steps

    return MineDojoMiniGPT4Env(
        cmd_args=args,
        seed=args.seed,
        device=device,
        save_rgb=False,
        img_only_obs=img_only,
        **task_conf)
