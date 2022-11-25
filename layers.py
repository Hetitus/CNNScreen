#!/bin/python3
# %%

import argparse
import os
import random
from random import shuffle
import sys
import datetime
import time
import math
import json
import copy
from pathlib import Path

import numpy as np
import utils

# AI suits
import tensorflow as tf


class Dense_L(object):
    def __init__(self,numNodes):
        self.numNodes_range = numNodes

    def get_layer(self):
        curr_nodes = np.random.randint(self.numNodes_range[0],self.numNodes_range[1])
        current_layer = tf.keras.layers.Dense(curr_nodes)
        return current_layer

class Act_L(object):
    def __init__(self,act_prob):
        self.act_prob_vec = act_prob
        self.poss_lay_class = ['relu','leaky_relu','prelu','elu','thresholded_relu','softmax']
        self.act_layer_dict = make_layer_dict()
    def get_layer(self):
        layer_name = random.choices(
            self.poss_lay_class, weights=self.act_prob_vec, k=1)

        curr_layer_o =  self.act_layer_dict[layer_name[0]]     
        curr_layer = curr_layer_o.get_layer()
        return curr_layer
        


class Conv2D_L(object):
    def __init__(self) -> None:
        pass
    def get_layer(self):
        current_layer = tf.keras.layers.Conv2D(4, 3,padding="same")
        return current_layer

class Pool_L():
    def __init__(self,pool_prob):
        self.pool_prob_vec = pool_prob
        self.poss_lay_class = ['MaxPool','AveragePool']
    
    def get_layer(self):
        layer_name = random.choices(
            self.poss_lay_class, weights=self.pool_prob_vec, k=1)
        if layer_name == 'MaxPool':
            current_layer = tf.keras.layers.MaxPooling2D(padding="same")
        else:
            current_layer = tf.keras.layers.AveragePooling2D(padding="same")
        return current_layer
        

class Batch_L():
    def __init__(self,norm_prob):
        self.norm_prob_vec = norm_prob
        self.poss_lay_class = ['Batch','Cross']
    
    def get_layer(self):
        layer_name = random.choices(
            self.poss_lay_class, weights=self.norm_prob_vec, k=1)
        if layer_name == 'Batch':
            current_layer = tf.keras.layers.BatchNormalization()
        else:
            current_layer = tf.keras.layers.UnitNormalization()
        return current_layer

class Norm_L():
    def __init__(self) -> None:
        pass

    def get_layer(self):
        current_layer = tf.keras.layers.Dropout(.2)
        return current_layer

class Flatten_L():
    def __init__(self) -> None:
        pass

    def get_layer(self):
        current_layer = tf.keras.layers.Flatten()
        return current_layer


# #activation layer

class ReLu_L():
    def __init__(self) -> None:
        pass

    def get_layer(self):
        current_layer = tf.keras.layers.ReLU()
        return current_layer


class LeakyReLu_L():
    def __init__(self) -> None:
        pass

    def get_layer(self):
        current_layer = tf.keras.layers.LeakyReLU()
        return current_layer

class PReLU_L():
    def __init__(self) -> None:
        pass

    def get_layer(self):
        current_layer = tf.keras.layers.PReLU()
        return current_layer

class ELU_L():
    def __init__(self) -> None:
        pass

    def get_layer(self):
        current_layer = tf.keras.layers.ELU()
        return current_layer

class ThresholdedReLU_L():
    def __init__(self) -> None:
        pass

    def get_layer(self):
        current_layer = tf.keras.layers.ThresholdedReLU()
        return current_layer

class Softmax_L():
    def __init__(self) -> None:
        pass

    def get_layer(self):
        current_layer = tf.keras.layers.Softmax()
        return current_layer


def make_layer_dict():

    act_layer_dict = {
        "relu": ReLu_L(),
        "leaky_relu": LeakyReLu_L(),
        "prelu": PReLU_L(),
        "elu": ELU_L(),
        "thresholded_relu": ThresholdedReLU_L(),
        "softmax": Softmax_L()
    }

    return act_layer_dict