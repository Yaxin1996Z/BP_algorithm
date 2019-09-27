# 解决了MNIST数据集的问题
# mini-batch训练
#

import numpy as np
from load_mnist import load_train_images,load_train_labels
import os
import pickle

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

class BPnn():
    # 初始化网络，隐层和输出层权重、偏置
    def __init__(self, d, n_H, c):
        self.weights1 = np.random.normal(0, 1, (n_H, d))
        self.bias1 = np.zeros((n_H))
        self.weights2 = np.random.normal(0, 1, (c, n_H))
        self.bias2 = np.zeros((c))
        # 定义隐藏层和输出层，前向传播计算出预测输出
        self.layer1 = Layer(d, n_H, self.weights1, self.bias1)
        self.layer2 = Layer(n_H, c, self.weights2, self.bias2)

    # 检测网络效果
    def test(self, testin):
        for sample in testin:
            inputs = np.array(sample)
            h = self.layer1.feedforward(inputs)
            y_pred = self.layer2.feedforward(h)
            print("ouput : ", y_pred)

    # # 计算损失
    # def calLoss(self, samples, samples_result):
    #     loss = 0.0
    #     for sample in samples:
    #         inputs = np.array(sample[0])
    #         y_true = np.array(sample[1])
    #         h = self.layer1.feedforward(inputs)
    #         y_pred = self.layer2.feedforward(h)
    #         loss += loss_mse(y_true, y_pred)
    #     return loss

    # 训练网络
    def train(self, samples, samples_result, steps=100, learning_rate=0.1):
        dataset_size = np.shape(samples)[0]
        for step in range(steps):
            start = (step * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            mini_samples = samples[start:end]
            mini_samples_result = samples_result[start:end]
            for t in range(batch_size):
                inputs = mini_samples[t]
                y_true = mini_samples_result[t]
                net_h, h, net_y, y = np.array([]), np.array([]), np.array([]), np.array([])
                net_h = np.dot(self.weights1, inputs) + self.bias1
                h = sigmoid(net_h)
                net_y = np.dot(self.weights2, h) + self.bias2
                y_pred = sigmoid(net_y)

                # 更新输出层权重
                for k in range(c):
                    delta_k = (y_pred - y_true) * deriv_sigmoid(net_y)
                    for j in range(n_H):
                        # weights2[k][j]-=learning_rate*(y_pred[k]-y_true[k])*deriv_sigmoid(net_y[k])*h[j]
                        self.weights2[k][j] -= learning_rate * (delta_k[k]) * h[j]
                # 更新隐层权重
                for j in range(n_H):
                    delta_j = np.dot(delta_k, self.weights2[:, j]) * deriv_sigmoid(net_h[j])
                    # print(delta_j)
                    for i in range(d):
                        self.weights1[j][i] -= learning_rate * delta_j * inputs[i]
            if step % display_dteps == 0: print("steps:%d, loss:%f" % (step, loss_mse(y_true, y_pred)))


def f(y):
    act = np.zeros((10))
    act[int(y[0])] = 1
    return act


# 定义训练数据的batch大小
batch_size = 1
# 设定学习率
learning_rate = 0.5
# 设定训练轮数
steps = 1000
dataset_size = 4
display_dteps = 10
# 输入层中间层输出层
d, n_H, c = 4, 10, 1
# d, n_H, c = 28*28, 15, 10

if __name__ == '__main__':
    # 训练样本 简单训练
    # samples = [[-2, -1], [25, 6], [17, 4], [-15, -6]]
    # samples_result = np.array([[1],[0],[0],[1]])
    samples = [[5.1,3.5,1.4,0.2],
        [4.9,3.0,1.4,0.2],
        [4.7,3.2,1.3,0.2],
        [4.6,3.1,1.5,0.2],
        [5.0,3.6,1.4,0.2],
        [5.4,3.9,1.7,0.4],
        [4.6,3.4,1.4,0.3],
        [5.0,3.4,1.5,0.2],
        [4.4,2.9,1.4,0.2],
        [4.9,3.1,1.5,0.1],
        [5.4,3.7,1.5,0.2],
        [4.8,3.4,1.6,0.2],
        [4.8,3.0,1.4,0.1],
        [4.3,3.0,1.1,0.1],
        [7.0,3.2,4.7,1.4],
        [6.4,3.2,4.5,1.5],
        [6.9,3.1,4.9,1.5],
        [5.5,2.3,4.0,1.3],
        [6.5,2.8,4.6,1.5],
        [5.7,2.8,4.5,1.3],
        [6.3,3.3,4.7,1.6],
        [4.9,2.4,3.3,1.0]]
    samples_result = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1]])

    # 加载MNIST数据集
    # if os.path.exists(os.path.abspath("mnist.pkl")):
    #     pkl_file = open('mnist.pkl','rb')
    #     samples = pickle.load(pkl_file)
    #     samples_result = pickle.load(pkl_file)
    #     pkl_file.close()
    # else:
    #     samples = load_train_images()
    #     samples.resize((dataset_size, 28*28))
    #     labels = load_train_labels()
    #     samples_result = np.apply_along_axis(f, 1, labels[:, np.newaxis])
    #     # 将解析出来的数据集存储
    #     pickle_output = open('mnist.pkl','wb')
    #     pickle.dump(samples,pickle_output)
    #     pickle.dump(samples_result, pickle_output)
    #     pickle_output.close()


    bp = BPnn(d, n_H, c)
    bp.train(samples, samples_result, steps, learning_rate)
    # 测试样本
    # testsamples = [[-7, -3], [20, 2]]
    testsamples = [[4.7,3.2,1.3,0.2],
    [4.9, 2.4, 3.3, 1.0]]
    bp.test(testsamples)  # 输出应该是1,0