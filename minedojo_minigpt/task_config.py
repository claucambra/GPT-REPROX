# SPDX-FileCopyrightText: 2023 Claudio Cambra <developer@claudiocambra.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import yaml

from minedojo.sim import InventoryItem


DEFAULT_TASK_CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("task_config.yaml")

def config_for_task(task: str, yaml_file_path: pathlib.Path = DEFAULT_TASK_CONFIG_PATH):
    file = open(yaml_file_path, "r", encoding="utf-8")
    file_contents = file.read()
    file.close()

    data = yaml.load(file_contents, Loader=yaml.FullLoader)
    assert task in data, f"Task {task} not found in {yaml_file_path}"
    return data[task]


def create_initial_inventory(task_conf: dict) -> dict:
    init_items = {}

    if "initial_inventory" in task_conf:
        init_items = task_conf["initial_inventory"]
        init_inv = [InventoryItem(slot=i,
                                  name=k,
                                  variant=None,
                                  quantity=task_conf["initial_inventory"])[k]
                    for i, k in enumerate(list(task_conf["initial_inventory"].keys()))]

        task_conf["initial_inventory"] = init_inv

    return init_items


def setup_task_conf(task: str) -> dict:
    task_conf = config_for_task(task)
    create_initial_inventory(task_conf)
    return task_conf