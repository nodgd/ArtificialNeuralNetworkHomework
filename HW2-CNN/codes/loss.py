from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Using your codes in Homework 1'''
        pass

    def backward(self, input, target):
        '''Using your codes in Homework 1'''
        pass


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        pass

    def backward(self, input, target):
        '''Your codes here'''
        pass
