# Layer类里自动更新权重 但是不能定义隐层层数

import numpy as np
from load_mnist import load_train_images,load_train_labels
import os
import pickle

# sigmoid激活函数
def sigmoid(net):
    y = 1/(1+np.exp(-net))
    return y
# 均方误差
def loss_mse(y_true: np.ndarray, y_pred: np.ndarray):
    return 0.5 *((y_true-y_pred)**2).sum()
# sigmoid函数的导数
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)
class Layer:
    # 初始化神经网络层 初始化参数：输入节点数，输出节点数，权重矩阵，偏置向量
    def __init__(self,input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.normal(0, 1, (output_size,input_size))
        self.bias = np.zeros((output_size))
    # 该层前向传播 得到总输出向量
    def feedforward(self,inputs):
        net = np.dot(self.weights,inputs) + self.bias  # inputs和weight一个行向量一个列向量
        outputs = sigmoid(net)
        return outputs

    # input假设有d个单元 用i计数 该层输出是n个 用j计数 该层后面一层是c个
    # 要更新的权矩阵为n*h  后面层w_next的权矩阵为c*n  后面层的delt有c个 每一行有c个
    def update_weights_return_delt(self,inputs,delts,w_next, nets):
        num_in = np.size(inputs[0])
        num_out = np.size(nets[0])
        batch_size = np.size(nets[:,0])
        delt_pre = np.dot(delts,w_next)*deriv_sigmoid(nets)
        for j in range(num_out):
            for i in range(num_in):
                for k in range(batch_size):
                    self.weights[j][i] -= learning_rate*delt_pre[k][j]*inputs[k][i]
        return delt_pre

class BPnn():
    # 初始化网络，隐层和输出层权重、偏置
    def __init__(self, d, n_H, c):
        # 定义隐藏层和输出层，前向传播计算出预测输出
        self.d = d
        self.n_H = n_H
        self.c = c
        self.layer1 = Layer(d, n_H)
        self.layer2 = Layer(n_H, c)

    # 检测网络效果
    def test(self, testin):
        for sample in testin:
            inputs = np.array(sample)
            h = self.layer1.feedforward(inputs)
            y_pred = self.layer2.feedforward(h)
            print("ouput : ", y_pred)

    # 训练网络
    def train(self, samples, samples_result, steps=100, learning_rate=0.1):
        dataset_size = np.shape(samples)[0]
        for step in range(steps):
            start = (step * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            mini_samples = samples[start:end]  # bs*d
            mini_samples_result = samples_result[start:end]   #bs*c
            net_h, h, net_y, y_pred = np.zeros((batch_size,self.n_H)), np.zeros((batch_size,self.n_H)), np.zeros((batch_size,self.c)), np.zeros((batch_size,self.c))
            delta_last = np.zeros((batch_size,self.c))
            for t in range(batch_size):
                inputs = mini_samples[t]
                y_true = mini_samples_result[t]
                net_h[t] = np.dot(self.layer1.weights, inputs) + self.layer1.bias
                h[t] = sigmoid(net_h[t])
                net_y[t] = np.dot(self.layer2.weights, h[t]) + self.layer2.bias
                y_pred[t] = sigmoid(net_y[t])


            delta_last = (y_pred - mini_samples_result) * deriv_sigmoid(net_y)
            for j in range(c):
                for i in range(n_H):
                    for k in range(batch_size):
                        # weights2[k][j]-=learning_rate*(y_pred[k]-y_true[k])*deriv_sigmoid(net_y[k])*h[j]
                        self.layer2.weights[j][i] -= learning_rate * (delta_last[k][j]) * h[k][i]
                # 更新输出层权重

            delt_2 = self.layer1.update_weights_return_delt(mini_samples,delta_last,self.layer2.weights,net_h )
                # 更新隐层权重
            # y_pred=
            if step % display_dteps == 0: print("steps:%d, loss:%f" % (step, loss_mse(mini_samples_result, y_pred)))


batch_size = 16
# 设定学习率
learning_rate = 0.01
# 设定训练轮数
steps = 10000
dataset_size = 60000
display_dteps = 50

if __name__ == '__main__':

    # samples = [[-2, -1], [25, 6], [17, -4], [-15, 4]]
    # samples_result = np.array([[0,0],[1,1],[1,0],[0,1]])
    # d, n_H, c = 2, 3, 2

    # samples = [[-2, -1], [25, 6], [17, 4], [-15, -6]]
    # samples_result = np.array([[1], [0], [0], [1]])
    # d, n_H, c = 2, 3, 1

    # 加载MNIST数据集
    if os.path.exists(os.path.abspath("mnist.pkl")):
        pkl_file = open('mnist.pkl','rb')
        samples = pickle.load(pkl_file)
        samples_result = pickle.load(pkl_file)
        pkl_file.close()
    else:
        samples = load_train_images()
        samples.resize((dataset_size, 28*28))
        labels = load_train_labels()
        samples_result = np.apply_along_axis(f, 1, labels[:, np.newaxis])
        # 将解析出来的数据集存储
        pickle_output = open('mnist.pkl','wb')
        pickle.dump(samples,pickle_output)
        pickle.dump(samples_result, pickle_output)
        pickle_output.close()
    d, n_H, c = 28*28, 15, 10
    bp = BPnn(d, n_H, c)
    bp.train(samples, samples_result, steps, learning_rate)

    # 测试样本
    # testsamples = [[-7, -3], [20, 2]]
    # bp.test(testsamples)  # 输出应该是1,0