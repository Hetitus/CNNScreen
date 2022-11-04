"""Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import argparse
from cmath import nan
from email import utils
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)