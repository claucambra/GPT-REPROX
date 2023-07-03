# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import os

from stable_baselines3 import PPO

from minedojo_minigpt.setup import setup_minedojo_gpt_env
from minedojo_minigpt.env import MineDojoMiniGPT4Env

from utils.setup import set_device, set_seed, setup_save_dir


def run_minedojo_test(args: argparse.Namespace, env: MineDojoMiniGPT4Env):
    ppo_agent = PPO("MlpPolicy", 
                    env, 
                    device=env.device, 
                    seed=env.seed, 
                    verbose=True,
                    n_steps=env.max_steps,
                    tensorboard_log=os.path.join(args.save_results_dir,
                                                 args.task,
                                                 "tensorboard"))

    time_steps = env.max_steps * args.episode_count
    ppo_agent.learn(total_timesteps=time_steps, progress_bar=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to MiniGPT-4 configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the MiniGPT-4 model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the MiniGPT-4 used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--task", type=str, default="craft_stick")
    parser.add_argument("--episode-count", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--save-results-dir", type=str, default="results")
    args = parser.parse_args()

    device = set_device()
    set_seed(args.seed)
    setup_save_dir(args)

    env = setup_minedojo_gpt_env(args, device, img_only=True)
    run_minedojo_test(args, env)
