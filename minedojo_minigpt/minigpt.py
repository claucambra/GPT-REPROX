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
            return int(float(answer)) # Prevent failures from floats like 1.0
        except ValueError:
            # If MiniGPT returns a long-winded text answer, split it by its spaces and extract the
            # numbers from it. Also work around float strings. Then return the last number found. 
            numbers = [int(float(i)) for i in answer.split() if i.isdigit() or i.replace('.', '').isdigit()]

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

        text_reply, _ = self.__gpt.answer(convo_copy, convo_copy.images, max_length=20000)
        assert isinstance(text_reply, str)
        
        return self.__parse_answer_for_reward(text_reply)

    # A slightly simplified and reworked version of upload_img from MiniGPT4.
    # It has modifications to work better with the ConversationWithImages class.
    def upload_img(self, rgb_image: Image.Image):
        image = self.__gpt.vis_processor(rgb_image).unsqueeze(0).to(self.__gpt.device)
        image_emb, _ = self.__gpt.model.encode_img(image)
        self.__gpt_conversation.add_image(image_emb)

    def upload_rgb_array(self, rgb_image_array: np.ndarray):
        # We want the vision processor to do its thing. For this, we need to pass the rgb_array as
        # a proper image, so that it is processed by the vision processor used by MiniGPT
        image = Image.fromarray(rgb_image_array)
        self.upload_img(image)
