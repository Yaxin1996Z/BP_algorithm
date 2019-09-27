# Layer类里自动更新权重
# 能定义隐层层数和每个隐层的单元个数

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
    def __init__(self,input_size=0,output_size=0):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.normal(0, 1, (output_size,input_size))
        self.bias = np.zeros((output_size))
    def initWeights(self,input_size,output_size):
        self.weights = np.random.normal(0, 1, (output_size, input_size))
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
    def __init__(self, d, num_hidden, hidden_cells, c):
        # 定义隐藏层和输出层，前向传播计算出预测输出
        self.d = d
        self.num_hidden = num_hidden
        self.hidden_cells = hidden_cells
        self.c = c
        self.layer_hidden = [Layer()for i in range(num_hidden)]
        for i in range(self.num_hidden):
            if i==0:
                self.layer_hidden[i].initWeights(d,self.hidden_cells[i])
            else:
                self.layer_hidden[i].initWeights(self.hidden_cells[i-1],self.hidden_cells[i])
        self.layer_out = Layer(hidden_cells[-1], c)

    # 检测网络效果
    def test(self, testin):
        inputs = np.array(testin)
        for i in range(self.num_hidden):
            h = self.layer_hidden[i].feedforward(inputs)
            inputs = h
        y_pred = self.layer_out.feedforward(h)
        predicted = np.argmax(y_pred)
        print("ouput : ", predicted)

    # 训练网络
    def train(self, samples, samples_result, steps=100, learning_rate=0.1):
        dataset_size = np.shape(samples)[0]
        for step in range(steps):
            start = (step * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            mini_samples = samples[start:end]  # bs*d
            mini_samples_result = samples_result[start:end]   #bs*c
            net_h, h = [np.array([])for i in range(self.num_hidden)], [np.array([])for i in range(self.num_hidden)],
            for i in range(self.num_hidden):
                net_h[i]=np.zeros((batch_size,hidden_cells[i]))
                h[i]=np.zeros((batch_size,hidden_cells[i]))
            net_y, y_pred = np.zeros((batch_size,self.c)), np.zeros((batch_size,self.c))
            delta_last = np.zeros((batch_size,self.c))
            for t in range(batch_size):
                inputs = mini_samples[t]
                y_true = mini_samples_result[t]
                for k in range(self.num_hidden):
                    if k==0:
                        net_h[k][t] = np.dot(self.layer_hidden[k].weights, inputs) + self.layer_hidden[k].bias
                    else:
                        net_h[k][t] = np.dot(self.layer_hidden[k].weights, h[k-1][t]) + self.layer_hidden[k].bias
                    h[k][t] = sigmoid(net_h[k][t])
                net_y[t] = np.dot(self.layer_out.weights, h[k][t]) + self.layer_out.bias
                y_pred[t] = sigmoid(net_y[t])


            delta_last = (y_pred - mini_samples_result) * deriv_sigmoid(net_y)
            for j in range(c):
                for i in range(self.hidden_cells[-1]):
                    for k in range(batch_size):
                        # weights2[k][j]-=learning_rate*(y_pred[k]-y_true[k])*deriv_sigmoid(net_y[k])*h[j]
                        self.layer_out.weights[j][i] -= learning_rate * (delta_last[k][j]) * h[-1][k][i]
                # 更新输出层权重
            delta = delta_last
            if self.num_hidden==1:
                delta = self.layer_hidden[0].update_weights_return_delt(mini_samples, delta, self.layer_out.weights, net_h[0])
            else:
                pre_inputs = h[-2]

                for i in range(self.num_hidden):
                    if self.num_hidden-i==1:
                        delta = self.layer_hidden[0].update_weights_return_delt(mini_samples, delta, self.layer_hidden[1].weights,
                                                                       net_h[0])
                    elif i==0:
                        delta = self.layer_hidden[-1].update_weights_return_delt(pre_inputs, delta,
                                                                                self.layer_out.weights,
                                                                                net_h[-1])
                    else:
                        p = self.num_hidden-i-1
                        delta= self.layer_hidden[p].update_weights_return_delt(pre_inputs,delta,self.layer_hidden[p+1].weights,net_h[p] )
                        pre_inputs=h[p-1]
                # 更新隐层权重
            # y_pred=
            if step % display_dteps == 0: print("steps:%d, loss:%f" % (step, loss_mse(mini_samples_result, y_pred)))

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self))

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            net = pickle.loads(f.read())
        return net

batch_size = 16
# 设定学习率
learning_rate = 0.01
# 设定训练轮数
steps = 1000
dataset_size = 60000
display_dteps = 50

if __name__ == '__main__':

    # samples = [[-2, -1], [25, 6], [17, -4], [-15, 4]]
    # samples_result = np.array([[0,0],[1,1],[1,0],[0,1]])
    d,c = 28*28, 10
    num_hidden = 2
    hidden_cells = [15,20]

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

    # bp = BPnn(d, num_hidden, hidden_cells, c)
    if os.path.exists(os.path.abspath("net.pkl")):
        bp = BPnn.load("net.pkl")
    else:
        bp = BPnn(d, num_hidden, hidden_cells, c)
    bp.train(samples, samples_result, steps, learning_rate)
    bp.save('net.pkl')

    # testsamples = samples[3]
    # bp.test(testsamples)  # 输出应该是1,0
    # 测试样本
    # testsamples = [[-7, -3], [20, 2]]
    # bp.test(testsamples)  # 输出应该是1,0