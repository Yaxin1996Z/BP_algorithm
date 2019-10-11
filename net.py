import numpy as np
import pickle

batch_size = 16
# 设定学习率
learning_rate = 0.001
# 设定训练轮数
steps = 1000
display_dteps = 50

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
    def __init__(self,n_in,n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.weights = np.random.normal(0, 0.1, (n_out,n_in))
        self.bias = np.ones((n_out))

    # 该层前向传播 得到总输出向量
    def feedforward(self,inputs):
        net = np.dot(self.weights,inputs) + self.bias  # inputs和weight一个行向量一个列向量
        outputs = sigmoid(net)
        return outputs

    # input假设有d个单元 用i计数 该层输出是n个 用j计数 该层后面一层是c个
    # 要更新的权矩阵为n*h  后面层w_next的权矩阵为c*n  后面层的delt有c个 每一行有c个
    def update_weights_return_delt(self, inputs, delts, w_next, nets):
        delt_pre = np.dot(delts,w_next)*deriv_sigmoid(nets)
        for j in range(self.n_out):
            for k in range(batch_size):
                for i in range(self.n_in):
                    self.weights[j][i] -= learning_rate*delt_pre[k][j]*inputs[k][i]
                self.bias[j] -= learning_rate*delt_pre[k][j]*self.bias[j]
        return delt_pre


class BPnn:
    # 初始化网络，隐层和输出层权重、偏置
    def __init__(self):
        self.layers = []

    # 添加一层，可选激活函数
    def addLayer(self,n_in,n_out):
        self.layers.append(Layer(n_in,n_out))

    # 检测网络效果
    def forward(self, batch_in):
        inputs = np.array(batch_in)
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        y_pre = inputs
        return y_pre

    # 训练网络
    def train(self, samples, samples_result, dataset_size):
        for step in range(steps):
            start = (step * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            mini_samples = samples[start:end]  # bs*d
            mini_samples_result = samples_result[start:end]   #bs*c
            net_h, h = [np.array([])for i in range(self.num_hidden)], [np.array([])for i in range(self.num_hidden)],
            for i in range(self.num_hidden):
                net_h[i]=np.zeros((batch_size,self.hidden_cells[i]))
                h[i]=np.zeros((batch_size,self.hidden_cells[i]))
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
            for j in range(self.c):
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