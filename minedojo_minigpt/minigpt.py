# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import numpy as np

from PIL import Image

from .conversation import MineDojoMiniGPT4Conversation, build_current_task_prompt

from MiniGPT4.minigpt4.common.config import Config
from MiniGPT4.minigpt4.common.registry import registry
from MiniGPT4.minigpt4.conversation.conversation import Chat
from MiniGPT4.minigpt4.models.mini_gpt4 import MiniGPT4
from MiniGPT4.minigpt4.processors import Blip2ImageEvalProcessor

class MineDojoMiniGPT4:
    def __init__(self, args: argparse.Namespace):
        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_config.prompt_path = ""  # Unset

        model = MiniGPT4.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = Blip2ImageEvalProcessor.from_config(vis_processor_cfg)
        
        self.__gpt = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        self.__gpt_conversation = MineDojoMiniGPT4Conversation()

    # Unfortunately we can't always trust the model to return an integer-only answer.
    # If it returns a number-only answer, great -- just use that. If not, we need to
    # parse the text answer and extract the reward from it.
    def __parse_answer_for_reward(self, answer: str) -> int:
        try:
            return int(answer)
        except ValueError:
            # If MiniGPT returns a long-winded text answer, split it by its spaces and extract the
            # numbers from it. Then return the last number found.
            numbers = [int(i) for i in answer.split(" ") if i.isdigit()]

            if len(numbers) == 0:
                print("WARNING: Received no numbers in MiniGPT answer. "
                      "Full answer was: {}. Returning 0 as reward.".format(answer))
                return 0
            elif len(numbers) == 1:
                return numbers[0]
            else:
                print("WARNING: Received multiple numbers in MiniGPT answer. "
                      "Full answer was: {}. Returning last number as reward.".format(answer))
                return numbers[-1]

    def current_reward(self, task: str) -> int:
        prompt = build_current_task_prompt(task)

        # Ask is really just a convenience method to append the question/prompt to the conversation.
        # We don't really want/need every prompt to be contained in the history, just the images.
        # So we create a temporary copy of the conversation with the prompt applied to it.
        convo_copy = self.__gpt_conversation.copy()
        self.__gpt.ask(prompt, convo_copy)

        text_reply, _ = self.__gpt.answer(convo_copy, convo_copy.images)
        assert isinstance(text_reply, str)
        
        return self.__parse_answer_for_reward(text_reply)

    def upload_img(self, rgb_image: Image.Image):
        self.__gpt.upload_img(rgb_image, self.__gpt_conversation, self.__gpt_conversation.images)

    def upload_rgb_array(self, rgb_image_array: np.ndarray):
        # We want the vision processor to do its thing. For this, we need to pass the rgb_array as
        # a proper image, so that it is processed by the vision processor used by MiniGPT
        image = Image.fromarray(rgb_image_array)
        self.upload_img(image)
