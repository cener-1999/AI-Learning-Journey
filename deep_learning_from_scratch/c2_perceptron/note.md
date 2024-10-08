# 第二章 感知机

感知机是神经网络（深度学习）的起源的算法



## 感知机是什么

感知机

- 接受多个输入信号，输入一个信号
- 这里的信号类似“流”
- 感知机的信号只有1/0两种取值

    ![](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409050004520.png)

$x_1$,$x_2$是输入信号，$y$是输入信号，$w_1$,$w_2$是权重, $o$是“神经元（结点）”

- 输入信号被送往神经元时，会被分别乘以固定的权重$(w_1x_1, w_2x_2)$
- **神经元被激活**：神经元会计算传送过来的信号的总和，只有当这个总和**超过**了某个界限值时，才会输出1
- 这个值被称为阈值，用$θ$表示



![](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409050004680.png)

- 权重：控制信号流动难度，感知机的权重越大，通过的信号就越大



## 简单的逻辑电路

与或非

与非门 not And (只有输入为11时才取0)



### 用感知机来表示逻辑电路

感知机可以表示逻辑电路，并且与门，与非门，或门的感知机构造是一样的，它们只有参数的值不同（权重与阈值）

构造相同的感知机，只需要通过适当地调整参数的值，就可以改变其逻辑功能。



**与**

$（w_1,w_2, \theta）=(0.5,0.5,0.7)$

$（w_1,w_2, \theta）=(1, 1, 1)$



与非门

$(w_1,w_2, \theta）= (-0.5,-0.5,-0.7)$



或门 

$(w_1,w_2, \theta）= (1, 1, 0.5)$



### 导入权重和偏置

修改实现形式 $\theta$ 换成$b$

![img](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409050004551.png)

**b**：偏置



### 为什么简单的感知机无法实现 或与门(XOR)？

![image-20240905012625401](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409050126508.png)



## 从与非门到计算机

计算机与感知机一样，有输入输出，会按照某个既定的规则进行计算；

多层感知机能够进行复杂的表示，感知机通过叠加层能够进行非线性的表示，理论上还可以表示计算机的处理

理论上而言，两层感知机就可以构建计算机，因为激活函数使用了非线性的sigmoid函数的感知机，可以表示任何函数



## 本章小结

![img](https://mypicsformarkdown.oss-cn-shanghai.aliyuncs.com/imgs/202409050130291.png)