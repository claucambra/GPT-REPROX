# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse

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


    def current_reward(self, task: str) -> int:
        prompt = build_current_task_prompt(task)
        self.__gpt.ask(prompt, self.__gpt_conversation)

        text_reply, _ = self.__gpt.answer(self.__gpt_conversation, self.__gpt_conversation.images)
        assert isinstance(text_reply, str)
        
        return int(text_reply)


    def upload_img(self, rgb_image):
        self.__gpt.upload_img(rgb_image, self.__gpt_conversation, self.__gpt_conversation.images)
