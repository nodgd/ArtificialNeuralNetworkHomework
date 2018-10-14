from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return np.sum((input - target) * (input - target)) / (2 * input.size)

    def backward(self, input, target):
        return (input - target) / input.size


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        input = input - np.max(input, axis=1, keepdims=True)
        self._exp_input = np.exp(input)
        self._sum_exp_input = np.sum(self._exp_input, axis=1)
        self._sum_target = np.sum(target, axis=1)
        loss = (np.sum(self._sum_target * np.log(self._sum_exp_input)) - np.sum(target * input)) / input.size
        return loss

    def backward(self, input, target):
        grad_input = ((self._sum_target / self._sum_exp_input).reshape(-1, 1) * self._exp_input - target) / input.size
        return grad_input
