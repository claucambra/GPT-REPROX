# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

from gym import spaces as old_spaces
from gymnasium import spaces


def convert_gym_space_multidiscrete(old_gym_multidescrete: old_spaces.MultiDiscrete) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(old_gym_multidescrete.nvec,
                                dtype=old_gym_multidescrete.dtype)


def convert_gym_space_box(old_gym_box: old_spaces.Box) -> spaces.Box:
    return spaces.Box(low=old_gym_box.low,
                        high=old_gym_box.high,
                        shape=old_gym_box.shape,
                        dtype=old_gym_box.dtype)


def convert_gym_space_dict(old_gym_dict: old_spaces.Dict) -> dict:
    normal_dict = spaces.Dict()
    for key, value in old_gym_dict.items():
        if isinstance(value, old_spaces.Dict):
            normal_dict[key] = convert_gym_space_dict(value)
        elif isinstance(value, old_spaces.Box):
            normal_dict[key] = convert_gym_space_box(value)
        elif isinstance(value, old_spaces.Text):
            val_length = value.shape[0]
            normal_dict[key] = spaces.Text(min_length=val_length, 
                                            max_length=val_length)
        elif isinstance(value, old_spaces.Discrete):
            normal_dict[key] = spaces.Discrete(value.n)
        elif isinstance(value, old_spaces.MultiDiscrete):
            normal_dict[key] = convert_gym_space_multidiscrete(value)
        else:
            normal_dict[key] = value

    return normal_dict
