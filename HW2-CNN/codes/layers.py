import numpy as np

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

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
        self._input = input
        return np.maximum(input, 0)

    def backward(self, grad_output):
        grad_output[self._input < 0] = 0
        return grad_output


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        self._output = 1.0 / (1.0 + np.exp(-input))
        return self._output

    def backward(self, grad_output):
        return grad_output * self._output * (1.0 - self._output)


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
        self._input = input
        output = np.dot(input, self.W) + self.b
        return output

    def backward(self, grad_output):
        self.grad_b = np.sum(grad_output, axis=0)
        self.grad_W = np.dot(self._input.transpose(), grad_output)
        return np.dot(grad_output, self.W.transpose())

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b


class Reshape(Layer):
    def __init__(self, name, new_shape):
        super(Reshape, self).__init__(name)
        self.new_shape = new_shape

    def forward(self, input):
        self._input_shape = input.shape
        return input.reshape(self.new_shape)

    def backward(self, grad_output):
        return grad_output.reshape(self._input_shape)


class Conv2D(Layer):
    def __init__(self, name, in_channel, out_channel, kernel_size, pad, init_std):
        super(Conv2D, self).__init__(name, trainable=True)
        self.ks = kernel_size
        self.pad = pad
        self.ci = in_channel
        self.co = out_channel
        self.w = np.random.randn(kernel_size, kernel_size, in_channel, out_channel) * init_std
        self.b = np.zeros(out_channel)

        self.diff_w = np.zeros(self.w.shape)
        self.diff_b = np.zeros(out_channel)

    def forward(self, input):
        self.n, self.hi, self.wi, ci = input.shape
        assert(ci == self.ci)
        self._input_with_pad = np.zeros([self.n, self.hi + self.pad * 2, self.wi + self.pad * 2, self.ci])
        self._input_with_pad[ : , self.pad : self.pad + self.hi, self.pad : self.pad + self.wi, : ] = input
        self.ho = self.hi + self.pad * 2 - self.ks + 1
        self.wo = self.wi + self.pad * 2 - self.ks + 1
        output = np.zeros([self.n, self.ho, self.wo, self.co])
        output += self.b.reshape([1, 1, 1, self.co])
        for kh in range(self.ks):
            for kw in range(self.ks):
                output += np.dot(self._input_with_pad[ : , kh : kh + self.ho, kw : kw + self.wo, : ].reshape(-1, self.ci), self.w[kh, kw, : , : ].reshape(self.ci, self.co)).reshape(output.shape)
        return output

    def backward(self, grad_output):
        self.grad_b = np.sum(grad_output, axis=(0, 1, 2))
        self.grad_w = np.zeros([self.ks, self.ks, self.ci, self.co])
        for kh in range(self.ks):
            for kw in range(self.ks):
                self.grad_w[kh, kw, : , : ] = np.dot(self._input_with_pad[ : , kh : kh + self.ho, kw : kw + self.wo, : ].reshape(-1, self.ci).transpose(), grad_output.reshape(-1, self.co))
        grad_input_with_pad = np.zeros(self._input_with_pad.shape)
        for kh in range(self.ks):
            for kw in range(self.ks):
                grad_input_with_pad[ : , kh : kh + self.ho, kw : kw + self.wo, : ] += np.dot(grad_output.reshape(-1, self.co), self.w[kh, kw, : , : ].reshape(self.ci, self.co).transpose()).reshape(self.n, self.ho, self.wo, self.ci)
        grad_input = grad_input_with_pad[ : , self.pad : self.pad + self.hi, self.pad : self.pad + self.wi, : ]
        return grad_input

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_w = mm * self.diff_w + (self.grad_w + wd * self.w)
        self.w = self.w - lr * self.diff_w

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b


class AvgPool2D(Layer):
    def __init__(self, name, kernel_size, pad):
        super(AvgPool2D, self).__init__(name)
        self.ks = kernel_size
        self.pad = pad

    def forward(self, input):
        self.n, self.hi, self.wi, self.c = input.shape
        self._input_with_pad = np.zeros([self.n, self.hi + self.pad * 2, self.wi + self.pad * 2, self.c])
        self._input_with_pad[ : , self.pad : self.pad + self.hi, self.pad : self.pad + self.wi, : ] = input
        self.ho = (self.hi + self.pad * 2) // self.ks
        self.wo = (self.wi + self.pad * 2) // self.ks
        output = np.zeros([self.n, self.ho, self.wo, self.c])
        for kh in range(self.ks):
            for kw in range(self.ks):
                output += self._input_with_pad[ : , kh : kh + (self.ho - 1) * self.ks + 1 : self.ks, kw : kw + (self.wo - 1) * self.ks + 1 : self.ks, : ]
        output /= self.ks * self.ks
        return output

    def backward(self, grad_output):
        grad_input_with_pad = np.zeros(self._input_with_pad.shape)
        temp = grad_output * (self.ks * self.ks)
        for kh in range(self.ks):
            for kw in range(self.ks):
                grad_input_with_pad[ : , kh : kh + (self.ho - 1) * self.ks + 1 : self.ks, kw : kw + (self.wo - 1) * self.ks + 1 : self.ks, : ] = temp
        grad_input = grad_input_with_pad[ : , self.pad : self.pad + self.hi, self.pad : self.pad + self.wi, : ]
        return grad_input
