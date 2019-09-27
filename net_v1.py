import numpy as np


# sigmoid激活函数
def sigmoid(net):
    y = 1/(1+np.exp(-net))
    return y

# tanh激活函数
def abtanh(net,a=1.716,b=2/3):
    y = a*((1-np.exp(-b*net))/(1+np.exp(-b*net)))
    return y

# 均方误差的定义
def loss_mse(y_true: np.ndarray, y_pred: np.ndarray):
    return ((y_true-y_pred)**2).mean()


class Neuron:
    # 神经元初始化 权重 偏置
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias

    # 前向传播
    def feedforward(self,inputs):
        net = np.dot(inputs,self.weights)+self.bias  #inputs和weight一个行向量一个列向量
        outputs = sigmoid(net)
        return outputs

class Layer:
    # 初始化神经网络层 初始化参数：输入节点数，输出节点数，权重矩阵，偏置向量
    def __init__(self,input_size,output_size,weights,bias):
        self.input_size = input_size
        self.output_size = output_size
        self.ns = ([])
        for i in range(output_size):
            self.ns = np.hstack((self.ns,Neuron(weights[i],bias[i])))  # 初始化该层神经节点
    # 该层前向传播 得到总输出向量
    def feedforward(self,inputs):
        outputs = ([])
        for i in range(self.output_size):
            outputs = np.hstack((outputs, self.ns[i].feedforward(inputs)))
        return outputs


if __name__ == '__main__':
    weights1 = np.array([[0.,1.],[0.,1.]])
    bias1 = np.array([0.,0.])
    weights2 = np.array([[0.,1.]])
    bias2 = np.array([0.])
    layer1 = Layer(2,2,weights1,bias1)
    layer2 = Layer(2,1,weights2,bias2)
    h=layer1.feedforward(np.array([2,3]))
    y = layer2.feedforward(h)
    print(type(y))

