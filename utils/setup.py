# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
import random
import argparse
import os

def set_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print('Running on device: ', device)
    return device


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_save_dir(args: argparse.Namespace):
    save_dir = os.path.join(args.save_results_dir, args.task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
