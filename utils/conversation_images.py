# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

from ..MiniGPT4.minigpt4.conversation import Conversation


# A simple wrapper class that also adds an images member, so that this does not
# need to be taken care of independently of the Conversation class.
class ConversationWithImages(Conversation):
    images: list = []