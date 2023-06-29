# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import os
import torch

from minedojo_minigpt.env import MineDojoMiniGPT4Env
from task_config import setup_task_conf


def setup_minedojo_gpt_env(args: argparse.Namespace, device: torch.device) -> MineDojoMiniGPT4Env:
    save_dir = os.path.join(args.save_results_dir, args.task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    task_conf = setup_task_conf(args.task)

    return MineDojoMiniGPT4Env(
        cmd_args=args,
        seed=args.seed,
        device=device,
        save_rgb=False,
        **task_conf)
