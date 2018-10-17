from loss import *
import numpy as np
from load_data import load_mnist_4d
from solve_net import *
from layers import *
from network import Network
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
import sys

def testSoftmax():
    a = np.random.randn(3, 4)
    b = a + np.random.randn(3, 4) * 0.01
    print('a = ', a)
    print('b = ', b)
    Loss = SoftmaxCrossEntropyLoss('s')
    c = Loss.forward(a, b)
    print('c = ', c)
    d = Loss.backward(a, b)
    print('d = ', d)

def testNetwork():
    model = Network()
    model.add(Conv2D('conv1', 1, 4, 3, 1, 0.01))
    model.add(Relu('relu1'))
    model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 14 x 14 x 4
    model.add(Conv2D('conv2', 4, 4, 3, 1, 0.01))
    model.add(Relu('relu2'))
    model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 7 x 7 x 4
    model.add(Reshape('flatten', (-1, 196)))
    model.add(Linear('fc3', 196, 10, 0.1))

    loss = SoftmaxCrossEntropyLoss(name='loss')
    
    train_data, test_data, train_label, test_label = load_mnist_4d('data')
    
    config = {
        'learning_rate': 0.1,
        'weight_decay': 0.0,
        'momentum': 0.0,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 5,
        'test_epoch': 5
    }
    loss_list = []
    acc_list = []
    for input, label in data_iterator(train_data, train_label, 3):
        target = onehot_encoding(label, 10)

        # forward net
        print('Network.input = ', input)
        output = model.forward(input)
        print('Network.output = ', output)
        # calculate loss
        loss_value = loss.forward(output, target)
        print('loss = ', loss_value)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

        sys.exit(0)
    

if __name__ == '__main__':
    testNetwork()

