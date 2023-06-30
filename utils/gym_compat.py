# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Callable

from gym import spaces as old_spaces
from gymnasium import spaces


def convert_gym_space_multidiscrete(old_gym_multidescrete: old_spaces.MultiDiscrete) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(old_gym_multidescrete.nvec,
                                dtype=old_gym_multidescrete.dtype)


def convert_gym_space_discrete(old_gym_discrete: old_spaces.Discrete) -> spaces.Discrete:
    return spaces.Discrete(old_gym_discrete.n)


def convert_gym_text(old_gym_text: old_spaces.Text) -> spaces.Text:
    return spaces.Text(max_length=old_gym_text.max_length,
                       min_length=old_gym_text.min_length,
                       charset=old_gym_text.charset)


def convert_gym_space_box(old_gym_box: old_spaces.Box) -> spaces.Box:
    return spaces.Box(low=old_gym_box.low,
                        high=old_gym_box.high,
                        shape=old_gym_box.shape,
                        dtype=old_gym_box.dtype)


def convert_gym_space_dict(old_gym_dict: old_spaces.Dict) -> dict:
    normal_dict = spaces.Dict()

    for key, value in old_gym_dict.items():
        # Will recursively call to convert sub-dicts
        normal_dict[key] = convert_gym_space(value)

    return normal_dict


def convert_gym_space_with_fallback_callback(old_gym_space: old_spaces.Space, 
                                             callback: Callable[[Any], spaces.Space]) -> spaces.Space:
    if isinstance(old_gym_space, old_spaces.Dict):
        return convert_gym_space_dict(old_gym_space)
    elif isinstance(old_gym_space, old_spaces.Box):
        return convert_gym_space_box(old_gym_space)
    elif isinstance(old_gym_space, old_spaces.Text):
        return convert_gym_text(old_gym_space)
    elif isinstance(old_gym_space, old_spaces.Discrete):
        return convert_gym_space_discrete(old_gym_space)
    elif isinstance(old_gym_space, old_spaces.MultiDiscrete):
        return convert_gym_space_multidiscrete(old_gym_space)
    else:  # Don't know, so invoke the callback
        return callback(old_gym_space)


def convert_gym_space(old_gym_space: old_spaces.Space) -> spaces.Space:
    def fallback_callback(space: Any) -> spaces.Space:
        print("Unknown space type: ", type(space))
        return space

    return convert_gym_space_with_fallback_callback(old_gym_space, fallback_callback)