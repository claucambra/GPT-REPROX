# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

from ..minigpt_utils import ConversationWithImages


class MineDojoMiniGPT4Conversation(ConversationWithImages):
    def __init__(self, 
                 system: str, 
                 roles: tuple = ("Human", "Assistant"), 
                 messages: list = [], 
                 offset: int = 2, 
                 sep: str = "###"):
        
        super.__init__(self, 
                       system=system, 
                       roles=roles, 
                       messages=messages, 
                       offset=offset, 
                       sep=sep)