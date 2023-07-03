# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

from PIL.Image import Image
from MiniGPT4.minigpt4.conversation.conversation import Conversation

IMAGE_MARKER_STRING = "<Img><ImageHere></Img>"
REPLACEMENT_MARKER_STRING = "<TEMP_REPLACEMENT>"

# A simple wrapper class that also adds an images member, so that this does not
# need to be taken care of independently of the Conversation class.
class ConversationWithImages(Conversation):
    __images: list = []
    __image_limit: int = -1

    @property
    def images(self) -> list:
        return self.__images

    @property
    def image_limit(self) -> int:
        return self.__image_limit
    
    @image_limit.setter
    def image_limit(self, image_limit: int) -> None:
        self.__image_limit = image_limit
        self.__clean_up_convo_img_limits()

    def __clean_up_imgs(self) -> None:
        while len(self.__images) > self.image_limit and self.image_limit >= 0:
            self.__images.pop(0)

    def __clean_up_img_convo_markers(self) -> None:
        prompt = self.get_prompt()
        img_split_prompt = prompt.split(IMAGE_MARKER_STRING)

        num_img_markers = len(img_split_prompt) - 1
        limit_diff = num_img_markers - self.image_limit

        for [role, message] in self.messages:
            if limit_diff <= 0:
                break
            elif message == IMAGE_MARKER_STRING:
                self.messages.remove([role, message])
                limit_diff -= 1

    def __clean_up_convo_img_limits(self) -> None:
        if not self.over_img_limit():
            return
        
        self.__clean_up_imgs()
        self.__clean_up_img_convo_markers()

        new_split_prompt = self.get_prompt().split(IMAGE_MARKER_STRING)
        num_new_image_tags = len(new_split_prompt)-1
        num_new_images = len(self.__images)

        assert num_new_image_tags == num_new_images, \
               f"Should have matching image tags {num_new_image_tags} and images {num_new_images}"
        
    def over_img_limit(self):
        return self.image_limit > 0 and len(self.images) >= self.image_limit

    def add_image(self, image: Image) -> None:
        if self.image_limit == 0:
            return

        self.__images.append(image)
        self.append_message(self.roles[0], IMAGE_MARKER_STRING)
        self.__clean_up_convo_img_limits()

    def copy(self) -> "ConversationWithImages":
        convo = ConversationWithImages(system=self.system,
                                       roles=[x for x in self.roles],
                                       messages=[[x, y] for x, y in self.messages],
                                       offset=self.offset,
                                       sep_style=self.sep_style,
                                       sep=self.sep,
                                       sep2=self.sep2,
                                       conv_id=self.conv_id)
        convo.__images = [x for x in self.images]
        convo.__image_limit = self.image_limit
        return convo
