# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import minedojo.sim.spaces as minedojo_spaces
from gymnasium import spaces

from utils.gym_compat import convert_gym_space_with_fallback_callback


# All of minedojo's spaces fit the old gym spaces, except for the Text space.
# They both fulfill similar purposes but have different interfaces, so we do
# the best we can here with a loose conversion to a gymnasium space.Text.
def convert_minedojo_space_text(old_minedojo_text: minedojo_spaces.Text) -> spaces.Text:
    text_length = old_minedojo_text.shape[0]
    return spaces.Text(max_length=text_length, min_length=text_length)


# Converts a minedojo space to a gymnasium space, accounting for the Text space.
def convert_minedojo_space(old_minedojo_space: minedojo_spaces.MineRLSpace) -> spaces.Space:
    return convert_gym_space_with_fallback_callback(old_minedojo_space, convert_minedojo_space_text)