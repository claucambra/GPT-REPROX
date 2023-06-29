# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse

from stable_baselines3 import PPO

from .minedojo_minigpt.setup import setup_minedojo_gpt_env
from .minedojo_minigpt.env import MinecraftMiniGPT4Env

from .utils.setup import set_device, set_seed


def run_minedojo_test(args: argparse.Namespace, env: MinecraftMiniGPT4Env):
    ppo_agent = PPO("LnMlpPolicy", 
                    env, 
                    device=env.device, 
                    seed=env.seed, 
                    verbose=True)

    time_steps = env.max_steps * args.episode_count
    ppo_agent.learn(total_timesteps=time_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to MiniGPT-4 configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the MiniGPT-4 model.")
    parser.add_argument("--task", type=str, default="craft_stick")
    parser.add_argument("--episode-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    device = set_device()
    set_seed(args.seed)

    env = setup_minedojo_gpt_env(args, device)
    run_minedojo_test(args, env)
