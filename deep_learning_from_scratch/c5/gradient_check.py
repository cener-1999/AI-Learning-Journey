import numpy as np
from deep_learning_from_scratch.dataset.mnist import load_mnist
from deep_learning_from_scratch.c5.network import TwoLayerNet

(x_train, t_train), (x_text, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch, t_batch = x_train[:3], t_train[:3]

grad_num = net.grad_numerical(x_batch, t_batch)
grad_backprop = net.grad_numerical(x_batch, t_batch)

for i in range(len(grad_num)):
    w_diff = np.average(np.abs(grad_num[i][0] - grad_backprop[i][0]))
    b_diff = np.average(np.abs(grad_num[i][1] - grad_backprop[i][1]))
    print(f'W{i} Diff: {w_diff}\n'
          f'b{i} Diff: {b_diff}')

