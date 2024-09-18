import numpy as np
from deep_learning_from_scratch.dataset.mnist import load_mnist
from deep_learning_from_scratch.c4.two_layer_net import TwoLayerNet

"""
stochastic gradient descent
step1: mini-batch
step2: calculate grad
step3: update W
step4: repeat 1-2-3
"""

(x_train, t_train), (x_text, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 超参数
learn_rate = 0.1
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)


net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iter_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(net.loss(x_batch, t_batch))
    print(f'round {i}: loss: {loss}')

    grad = net.grad(x_batch, t_batch)

    # update params
    for key in net.params.keys():
        net.params[key] -= learn_rate * grad[key]

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, x_text)
        test_acc = net.accuracy(x_text, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'train acc : {test_acc} | test_acc : {test_acc}')


with open('mnist_net_res.txt', 'w+') as f:
    f.writelines(f"{item}\n" for item in train_loss_list)
    f.write('------------------------------\n')
    f.writelines(f"{item}\n" for item in train_acc_list)
    f.write('------------------------------\n')
    f.writelines(f"{item}\n" for item in test_acc_list)
    f.write('------------------------------\n')
