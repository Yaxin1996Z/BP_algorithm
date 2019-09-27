import numpy as np

# sigmoid激活函数
def sigmoid(net):
    y = 1/(1+np.exp(-net))
    return y

# sigmoid函数的导数
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

# tanh激活函数
def abtanh(net,a=1.716,b=2/3):
    y = a*((1-np.exp(-b*net))/(1+np.exp(-b*net)))
    return y

# 均方误差
def loss_mse(y_true: np.ndarray, y_pred: np.ndarray):
    return 0.5 *((y_true-y_pred)**2).sum()


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
    # 1.初始化网络，权重、偏置、输入、实际输出
    weights1 = np.array([[0.1, 0.8], [0.4, 0.6]])
    weights2 = np.array([[0.3, 0.9]])
    bias1 = np.array([0., 0.])
    bias2 = np.array([0.])
    inputs = np.array([0.35, 0.9])
    y_true = np.array([0.5])
    d, n_H, c = 2, 2, 1
    learning_rate = 1

    # 定义隐藏层和输出层，前向传播计算出预测输出
    layer1 = Layer(d,n_H,weights1,bias1)
    layer2 = Layer(n_H,c,weights2,bias2)
    h=layer1.feedforward(inputs)
    y_pred = layer2.feedforward(h)
    # 初始损失
    loss = loss_mse(y_true, y_pred)

    # 训练100epoch
    for epoch in range(100):
        net_h, h, net_y, y = np.array([]), np.array([]), np.array([]), np.array([])
        net_h = np.dot(weights1, inputs) + bias1
        h = sigmoid(net_h)
        net_y = np.dot(weights2, h) + bias2
        y_pred = sigmoid(net_y)

        # 更新输出层权重
        for k in range(c):
            delta_k = (y_pred - y_true) * deriv_sigmoid(net_y)
            for j in range(n_H):
                # weights2[k][j]-=learning_rate*(y_pred[k]-y_true[k])*deriv_sigmoid(net_y[k])*h[j]
                weights2[k][j] -= learning_rate * (delta_k[k]) * h[j]
        # 更新隐层权重
        for j in range(n_H):
            delta_j = np.dot(delta_k, weights2[:, j]) * deriv_sigmoid(net_h[j])
            #print(delta_j)
            for i in range(d):
                weights1[j][i] -= learning_rate * delta_j * inputs[i]

        h = layer1.feedforward(inputs)
        y_pred = layer2.feedforward(h)
        loss = loss_mse(y_true, y_pred)
        print("y_pred:%f, loss:%f" % (y_pred,loss))