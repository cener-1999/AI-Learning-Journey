import numpy as np
from deep_learning_from_scratch.common.functions import cross_entropy_error, softmax
from deep_learning_from_scratch.common.grad import numerical_gradient

class Affine:
    def __init__(self, shape: tuple, weight_init_std):
        self.W = weight_init_std * np.random.rand(shape[0], shape[1])
        self.b = weight_init_std * np.random.rand(shape[1])
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        out = np.dot(x, self.W) + self.b
        self.x = x
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Relu:

    def __init__(self):
        self.mask = None

    def forward(self, x:np.ndarray):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(x, t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化权重, 权重使用符合高斯分布的随机数进行初始化，偏置使用0初始化
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param weight_init_std:
        """
        affine1 = Affine(shape=(input_size, hidden_size), weight_init_std=weight_init_std)
        affine2 = Affine(shape=(hidden_size, output_size), weight_init_std=weight_init_std)
        relu = Relu()
        softmax_loss = SoftmaxWithLoss()
        self.layers = [affine1, relu, affine2, softmax_loss]

    def predict(self, x):
        for layer in self.layers[:-1]: # no softmax when predict
            x = layer.forward(x)
        return x

    def accuracy(self, x, t):
        y = self.predict(x)
        z = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1) if t.ndim != 1 else t
        accuracy = np.sum(z==t) / float(t.shape[0])
        return accuracy

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.layers[-1].forward(y, t)
        return loss

    def grad_backprop(self, x, t):
        self.loss(x, t) # need predict first
        dout = 1
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
        grad = [(layer.dw, layer.db) for layer in self.layers if isinstance(layer, Affine)]
        return grad

    def grad_numerical(self, x, t):
        """
        :param x: 输入数据
        :param t: 监督数据
        :return:
        """
        W1, b1 = self.layers[0].W, self.layers[0].b
        W2, b2 = self.layers[2].W, self.layers[2].b

        def f(W):
            return self.loss(x, t)

        grads_W1 = numerical_gradient(f, W1)
        grads_b1 = numerical_gradient(f, b1)
        grads_W2 = numerical_gradient(f, W2)
        grads_b2 = numerical_gradient(f, b2)
        grads = [(grads_W1, grads_b1), (grads_W2, grads_b2)]
        return grads