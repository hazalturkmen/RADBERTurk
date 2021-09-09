import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)
    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    return config
