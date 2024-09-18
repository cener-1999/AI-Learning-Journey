import numpy as np
from deep_learning_from_scratch.common.functions import softmax, cross_entropy_error, identity_function, sigmoid
from deep_learning_from_scratch.common.grad import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化权重, 权重使用符合高斯分布的随机数进行初始化，偏置使用0初始化
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param weight_init_std:
        """
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """
        :param x: 输入数据
        :return:
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = softmax(a2)
        y = identity_function(z2)
        return y

    def loss(self, x, t):
        """
        :param x: 输入数据
        :param t: 监督数据
        :return:
        """
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

    def grad(self, x, t):
        """
        :param x: 输入数据
        :param t: 监督数据
        :return:
        """

        grads = dict()
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        def f(W):
            return self.loss(x, t)

        grads['W1'] = numerical_gradient(f, W1)
        grads['b1'] = numerical_gradient(f, b1)
        grads['W2'] = numerical_gradient(f, W2)
        grads['b2'] = numerical_gradient(f, b2)
        return grads

    def accuracy(self, x, t):
        """
        :param x: 测试数据集
        :param t: 测试数据集的真实标签
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    x = np.random.rand(100, 784)
    net.predict(x)