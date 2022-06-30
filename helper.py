import os
import math
from datetime import datetime

import torch

import yaml

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return '%s' % str('\n'.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def read_config(config_paths):
    final_config = {}
    for config_path in config_paths:
        with open(config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
            final_config = {**final_config, **config}
    
    return Struct(**final_config)
