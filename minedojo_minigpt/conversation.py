# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

from ..utils.conversation_images import ConversationWithImages


MINEDOJO_TASKPROMPT = "Give the following image: <Img>ImageContent</Img>. " \
                      "You will be provided with images of the video game Minecraft. " \
                      "These images will show the player attempting to complete a task. " \
                      "You will be told what task the player is attempting to complete. " \
                      "You will then be asked to provide an integer score ranging from 0-100 based on how close you think the player is to completing the task. " \
                      "Use the provided images to formulate the score. " \
                      "Please respond only with a numerical score."

CURRENT_STATE_PROMPT = "Generate a score from 0-100 based on how close you think the player is to completing the task. "


def build_current_task_prompt(task: str) -> str:
    return CURRENT_STATE_PROMPT + "The current task is: " + task + "."


class MineDojoMiniGPT4Conversation(ConversationWithImages):
    def __init__(self, 
                 system: str = MINEDOJO_TASKPROMPT, 
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