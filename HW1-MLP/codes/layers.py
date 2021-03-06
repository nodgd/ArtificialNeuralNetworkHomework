import numpy as np
from threading import Thread

THREAD_COUNT = 8

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        self.input = input
        return np.maximum(input, 0)

    def backward(self, grad_output):
        grad_output[self.input < 0] = 0
        return grad_output

class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        self.output = 1.0 / (1.0 + np.exp(-input))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        self.input = input
        output = np.dot(input, self.W) + self.b
        return output

    def backward(self, grad_output):
        self.grad_b = np.sum(grad_output, axis=0)
        self.grad_W = np.dot(self.input.transpose(), grad_output)
        return np.dot(grad_output, self.W.transpose())

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
