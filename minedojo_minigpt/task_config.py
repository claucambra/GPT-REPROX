# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import yaml

DEFAULT_TASK_CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("task_config.yaml")

def config_for_task(task: str, yaml_file_path: pathlib.Path = DEFAULT_TASK_CONFIG_PATH):
    file = open(yaml_file_path, "r", encoding="utf-8")
    file_contents = file.read()
    file.close()

    data = yaml.load(file_contents, Loader=yaml.FullLoader)
    assert task in data, f"Task {task} not found in {yaml_file_path}"
    return data[task]