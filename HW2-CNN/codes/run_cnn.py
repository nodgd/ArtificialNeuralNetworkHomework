from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d

train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture

model = Network()                                       # input.shape = [n, 28, 28, 1]
model.add(Conv2D('conv1', 1, 4, 3, 1, 0.01))            # output.shape = [n, 28, 28, 4]
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))                     # output.shape = [n, 14, 14, 4]
model.add(Conv2D('conv2', 4, 4, 3, 1, 0.01))            # output.shape = [n, 14, 14, 4]
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))                     # output.shape = [n, 7, 7, 4]
model.add(Reshape('flatten', (-1, 196)))                # output.shape = [n, 196]
model.add(Linear('fc3', 196, 10, 0.01))

loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.1,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 5,
    'test_epoch': 5
}

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])
