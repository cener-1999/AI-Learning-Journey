## 神经网络

衔接刚才的感知机，感知机的缺点是参数（权重，偏差）都要需要人工设置。而<u>神经网络的一个重要性质是它可以自动地从数据中学习到和尚的权重参数。</u>



## 从感知机到神经网络



### **神经网络的结构** 

输入层 —— 隐藏层 —— 输出层

![img](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409091951422.png)



### 激活函数闪亮登场✨

简化感知机的数学式，用一个函数来表示分情况的动作，引入新函数:

$y = \left\{ 
  \begin{array}{ll}
  0 & (b + w_1x_1 + w_2x_2) \leq 0 \\
  1 & (b + w_1x_1 + w_2x_2)   > 0
  \end{array}
\right.$    -->   $y = h(b + w_1x_1 + w_2x_2)$



**激活函数(activation function)**： 会将输入信号和总和转换为输出信号



## 激活函数



### sigmoid函数与阶跃函数 

- **sigmoid函数**

  - $h(x) = \dfrac {1} {1 + e^{-x}}$

  - ```python
    def sigmoid_function(x: np.ndarray):
        y = 1/ (np.exp(-x) + 1)
        return y
    ```

    

  - <img src="https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409092015180.png" alt="image-20240909201532114" style="zoom:33%;" />

- **阶跃函数**

  - $
    h(x) = \left\{ 
    \begin{array}{ll}
    0 & x \leq 0 \\
    1 & x > 0
    \end{array}
    \right.$  

  - ```python
    def step_function(x: np.ndarray):
        y = x > 0
        return y.astype(np.int64)
    ```

  - <img src="https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409092017985.png" alt="image-20240909201719930" style="zoom:33%;" />



**sigmoid函数和阶跃函数的比较**

**不同点**

| -      | sigmoid函数 | 阶跃函数            |
| ------ | ----------- | ------------------- |
| 平滑性 | 平滑        | 以0为界，急剧性变化 |
| 返回值 | 0~1的实数   | 只能0,1             |



**相同点**

- 形状相似：输入小时，输出接近0，输入变大，输入向1靠近
- 输出都在0~1之间
- 都是非线性函数



### 神经网络的激活函数必须使用非线性函数

**非线性函数**

线性函数：输出是输入的常数倍  (c是常数)



**神经网络的激活函数必须使用非线性函数**

- 使用线性函数，加深神经网络的层数就没有意义了；

线性函数的问题在于，不管如何加深层数，总是存在与之等效的“无 隐藏层的神经网络”。

> e..g.
> **激活函数**: 线性函数$h(x)=cx$
> **对应3层神经网络**: $y(x)=h(h(h(c)))$
>
> 运算会进行 $y(x)=c×c×c×x$的乘法运算 == $y(x)=c^3x$（即没有隐藏层的神经网络)
>
> 如本例所示, 使用线性函数时，无法发挥多层网络带来的优势。因此，为了发挥叠加层所 带来的优势，激活函数必须使用非线性函数。



### ReLU函数

最近主要使用ReLU函数: <u>输入大于0输出本身，输入小于0输出0</u>

$h(x) = \left\{ 
\begin{array}{11} 
x & x>0 \\
0 & x\leq0\end{array} \right.$



```python
def relu_function(x: np.ndarray):
    return np.maximum(x, 0)
```



<img src="https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409092027456.png" alt="image-20240909202701400" style="zoom:33%;" />

### 多维数组的运算

<img src="https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409092030396.png" alt="image-20240909203025330" style="zoom: 20%;" />



- 神经网络的运算可以作为矩阵运算打包进行
- `X = np.dot(A, B)`





## 输出层的设计



### 恒等函数和 softmax函数

根据情况改变输出层的激活函数，神经网络可以用在分类问题和回归问题上

一般来说：

- 回归问题  ->  **恒等函数** （回归问题是指预测连续数值）
- 分类问题  -> **softmax函数**



**恒等函数**

![img](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409092037310.png)

```python
def identity_function(a: np.ndarray):
    return a
```



**sorfmax函数**

![img](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409092037126.png)



$ y = \dfrac{a_k^e}{{\sum{}_{i=1}^{n} a_i^e}}$

~分子是输入信号$a_k$的指数函数，分母是所有输入信号的指数函数的和~



- softmax函数的输出是0.0到1.0之间的实数
- <u>softmax函数的输出值的总和是1，正因为有了这个性质，我们才可以把softmax函数的输出解释为“概率”</u>。
- 使用了softmax函数，各个元素之间的大小关系也不会改变。因为指数函数（y = exp(x)）是单调递增函数。



```python
def softmax_function(x: np.array):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
```



### 输出层的神经元数量

输出层的神经元数量需要根据问题来决定

e.g. 对于分类问题，要分为几类，输出层就设定为几类



## 神经网络的前向推理 forward propagation



当训练过程已经全部结束，使用学习到的参数，实现神经网络的“推理处理”。这个过程也称为神经网络的**前向传播**。



机器学习一样，使用神经网络解决问题时，也需要:

1. 首先使用训练数据（学习数据）进行权重参数的学习
2. 进行推理时，使用刚才学习到的参数，对输入数据进行分类



**预处理：**对神经网络的输入数据进行某种既定的转换；

**正规化：**把数据限定到某个范内的处理；

- 预处理在深度学习中非常实用，可以提高识别性能和学习效率；
- 很多预处理都会考虑到数据的整体分布，比如
  - 利用数据整体的均值或标准差，移动数据，使数据整体以 0为中心分布
  - 正规化，把数据的延展控制在一定范围内
  - 将数据整体的分布形状均匀化的方法，即数据白化（whitening）等



### 批处理

**使用批处理，可以实现高速且高效的运算**

批处理一次性计算大型数组要比分开逐步计算 各个小型数组速度更快

![img](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409092048441.png)



为什么批处理可以缩短处理时间？

> - 大多数处理数值计算的库都进行了能够高效处理大型数组运算的最优化
> - 在神经网络的运算中，当数据传送成为瓶颈时，批处理可以减轻数据总线的负荷（严格地讲，相对于数据读入，可以将更多的时间用在计算上



## 本章小结

- 神经网络中的激活函数使用平滑变化的`sigmoid`函数或`ReLU`函数。 
- 通过巧妙地使用NumPy多维数组，可以高效地实现神经网络。
-  机器学习的问题大体上可以分为回归问题和分类问题。 
- 关于输出层的激活函数，回归问题中一般用恒等函数，分类问题中 一般用softmax函数。 
- 分类问题中，输出层的神经元的数量设置为要分类的类别数。 
- 输入数据的集合称为批。通过以批为单位进行推理处理，能够实现 高速的运算。