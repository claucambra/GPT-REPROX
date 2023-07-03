# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

from utils.conversation_images import ConversationWithImages


MINEDOJO_TASKPROMPT = """
You will be provided with images of the video game Minecraft.
These images are of unmodded Minecraft being played in survival mode from the player's perspective.
These images will show the player attempting to complete a task.
The images are in sequential order, with the last image provided being the latest.
Firstly, you must identify and remember the objects within the image.
You will be asked to provide an integer score ranging from -1 to 1 based on how close the player is to completing a task.
You will be told what task the player is trying to complete.
Use the provided images to formulate the score.

Your reply must follow these rules:

Your reply must be strictly a number between -1 and 1 inclusive.

If within the image the location, objects, and player actions are not at all related to the task, you must reply only with the number -1.
If the player is performing a different task, or you cannot see any relevant information in the image, you must also reply only with the number -1.
If within the image the player is making no clear progress towards completing the task, reply only with the number 0.
If within the image all of the location, objects and actions are related to this task, reply only with the number 1.

If within the image the task is shown to be completed, reply only with the number 10.

When assesing these scores, do so with reference to the rules and mechanics of traditional, unmodified, survival mode Minecraft.

You must exclusively provide a numerical response.
There must be no description, explanation, or reasoning included in your answer.
"""

CURRENT_STATE_PROMPT = "Please provide a score for how relevant the information provided is to the task of "


def build_current_task_prompt(task: str) -> str:
    return CURRENT_STATE_PROMPT + "The current task is: " + task + "."


class MineDojoMiniGPT4Conversation(ConversationWithImages):
    def __init__(self, 
                 system: str = MINEDOJO_TASKPROMPT,
                 roles: list[str] = ["Human", "Assistant"],
                 messages: list[list[str]] = [],
                 offset: int = 2,
                 img_limit: int = 3):
        
        super().__init__(system, roles, messages, offset)
        self.image_limit = img_limit
