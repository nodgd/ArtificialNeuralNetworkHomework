from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return np.sum((input - target) * (input - target)) / (2 * input.size)

    def backward(self, input, target):
        return (input - target) / input.size
