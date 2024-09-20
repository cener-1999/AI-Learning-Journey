![image-20240920023748670](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409200237697.png)

# 误差反向传播法

*这章主要看代码实现部分*

通过数值微分计算权重参数的梯度费时，误差反向传导法更快；



## 计算图

*计算图的思想可以用来构建 **Layer**的概念*

- 共享步骤结果
- 只关注每个Layer的构建
  - 中间的计算结果保存下来
- 总结每种运算的特点和结果
- 对Affile Layer，Active Fuction Layer, Loss Layer 等的构建可以复用，并且搭建神经网络时结构变动更加清晰



**局部计算**

计算图的特征是可以通过传递“局部计算”

- 局部是指与自己相关的某个小范围；
- 局部计算是指无论全局发生了什么，都能只根据自己相关的信息输出接下来的结果；
- 无论全局有多么复杂，各个步骤要做的就是对象结点的局部计算；



## Layer



### **Affine Layer**

考虑一个简单的仿射变换（Affine Transformation）：

$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$

我们通常需要计算以下三部分的导数：
- 对输入 $\mathbf{x}$ 的导数（这将继续传递到前一层）。
- 对权重矩阵$\mathbf{W}$的导数（用于更新权重）。
- 对偏置向量 $\mathbf{b}$的导数（用于更新偏置）。



**对$\mathbf{W}$求导：**

损失函数 \(L\) 对权重 $\mathbf{W}$的梯度是：

$\displaystyle \frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{z}} \cdot \mathbf{x}^T$

其中：
- $\frac{\partial L}{\partial \mathbf{z}}$是从后续层反向传播回来的梯度。
- $\mathbf{x}^T$是输入向量的转置。

这个表达式可以理解为<u>计算输入向量 $\mathbf{x}$对每个输出 $\mathbf{z}$ 的影响，并结合反向传播的梯度来更新权重矩阵</u>。



**对$ \mathbf{x}$ 求导：**

损失函数 $L$ 对输入 $\mathbf{x}$的梯度是：

$\displaystyle \frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^T \cdot \frac{\partial L}{\partial \mathbf{z}}$



这是因为 $\mathbf{x}$ 通过权重矩阵 $\mathbf{W}$ 被线性变换到输出 $\mathbf{z}$，因此反向传播时梯度通过 $\mathbf{W}$ 反向传递。



**对 $\mathbf{b}$ 求导：**

损失函数 $L$ 对偏置 $\mathbf{b}$的梯度是：

$ \displaystyle \frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$

因为偏置 $\mathbf{b}$ 只是直接加到 $\mathbf{z}$ 上，因此其梯度就是 $\mathbf{z}$ 的梯度。



```python
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
```



### Active Function Layer

在神经网络中，激活函数（如 ReLU、Sigmoid、Tanh）通常逐元素应用，因此激活函数的导数也逐元素求解。



#### ReLU

对于 ReLU 函数 $ f(x) = \max(0, x) $，导数是：

$\displaystyle f'(x) = 
\begin{cases} 
1 & \text{if } x > 0 \\ 
0 & \text{if } x \leq 0 
\end{cases}$

<u>这个导数会逐元素应用在反向传播的梯度上。</u>

```python
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
```





#### Sigmoid

对于 Sigmoid 函数 $\displaystyle \sigma(x) = \frac{1}{1 + e^{-x}} $，导数是：

$\displaystyle \sigma'(x) = \sigma(x)(1 - \sigma(x)) $

反向传播时，这个导数也会逐元素乘以上一层的梯度



```python
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
```





### SoftmaxWithLoss Layer
Softmax层包含交叉熵误差，也称为Sortmax-with-Loss层

![img](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409200300434.png)

![img](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409200300484.png)

Softmax层的反向传播得到了$(y_1 − t_1,  y_2 −t_2, y_3 − t_3)$这样“漂亮”的结果

- （*y*1*, y*2*, y*3）是Softmax层的输出
- （*t*1*, t*2*, t*3）是监督数据
- 所以$(y_1 − t_1,  y_2 −t_2, y_3 − t_3)$是Softmax层的输出和监督标签的差分。

**神经网络的反向传播会把这个差分表示的误差传递给前面的层，这是神经网络学习中的重要性质**



神经网络中进行的处理有推理（inference）和学习两个阶段。**神经网络的推理通常不使用Softmax层。** 神经网络中未被正规 被称为“得分”。也就是说，当神经网络的推理只需要给出一个答案 的情况下，因为此时只对得分最大值感兴趣，所以不需要Softmax层。 不过，**神经网络的学习阶段则需要Softmax层**。

```python
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
```



## 组装层构建神经网络

通过将神经网络的组成元素以层的方式实现，可以轻松地构建神经网络。这个用层进行模块化的实现具有很大优点。因为想另外构建一个神经网络（比如5层、10层、20层……的大的神经网络）时，只需像组装乐高积木那样添加必要的层就可以了。之后，通过各个层内部实现的正向传播和反向传播，就可以正确计算进行识别处理或学习所需的梯度



```python
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
```





**梯度确认(gradient check)：**误差反向传播法的实现很复杂，容易出错。比较数值微分的结果和误差反向传播法的结果，以确认误差反向传播法的实现是否正确。

```python
net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch, t_batch = x_train[:3], t_train[:3]

grad_num = net.grad_numerical(x_batch, t_batch)
grad_backprop = net.grad_numerical(x_batch, t_batch)

for i in range(len(grad_num)):
    w_diff = np.average(np.abs(grad_num[i][0] - grad_backprop[i][0]))
    b_diff = np.average(np.abs(grad_num[i][1] - grad_backprop[i][1]))
    print(f'W{i} Diff: {w_diff}\n'
          f'b{i} Diff: {b_diff}')
```



学习：

```python
learn_rate = 0.1
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100

train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)


net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iter_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]

    loss = net.loss(x_batch, t_batch)

    grads = net.grad_backprop(x_batch, t_batch)

    affine_layers = [layer for layer in net.layers if isinstance(layer, Affine)]
    for layer, gard in zip(affine_layers, grads):
        layer.W -= learn_rate * gard[0]
        layer.b -= learn_rate * gard[1]

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_text, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'epoch {int(i // iter_per_epoch)}: train acc: {round(train_acc, 4)} | test_acc  {round(test_acc, 4)}')
```



## 小结

本章介绍：

- 将计算过程可视化的**计算图**，并使用计算图，介绍了神经网络中的误差反向传播法，并以层为单位实现了神经网络中的处理。
- 层：ReLU层、Softmax-with-Loss层、Affine层、Softmax层等，这些层中实现了`forward`和`backward`方法，通过**将数据正向和反向地传播，可 以高效地计算权重参数的梯度。**
- 通过使用层进行**模块化**，神经网络中可以自由地组装层，轻松构建出自己喜欢的网络。 



本章所学的内容:

- 通过使用计算图，可以直观地把握计算过程。
- 计算图的节点是由局部计算构成的。局部计算构成全局计算。
- 计算图的正向传播进行一般的计算。通过计算图的反向传播，可以计算各个节点的导数。
- 通过将神经网络的组成元素实现为层，可以高效地计算梯度（反向传播法)。 
- 通过比较数值微分和误差反向传播法的结果，可以确认误差反向传 播法的实现是否正确(梯度确认)。







