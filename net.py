import numpy as np
import pickle

# 定义了sigmoid、relu、和tanh三种激活函数的类

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    a = np.exp(x)
    b = np.exp(-x)
    return (a-b)/(a+b)

class Sigmoid:
    def __init__(self):
        self.last_cal = None
    # sigmoid激活函数
    def cal(self, x):
        self.last_cal = sigmoid(x)
        return self.last_cal
    # sigmoid函数的导数
    def deriv(self, x=None):
        if x:self.last_cal = sigmoid(x)
        return self.last_cal*(1-self.last_cal)

class Relu:
    def __init__(self):
        self.last_cal = None
    def cal(self, x):
        return np.maximum(x,0.0)
    def deriv(self, x=None):
        return 1*np.logical_and(x,1)

class Tanh:
    def __init__(self):
        self.last_cal = None
    def cal(self, x):
        self.last_cal = tanh(x)
        return self.last_cal
    def deriv(self, x=None):
        if x:self.last_cal = tanh(x)
        return 1-self.last_cal**2

# 定义softmax的计算方法

def softmax(x):
    z = x-np.max(x,axis=1,keepdims=True)
    z_exp = np.exp(z)
    res = z_exp/np.sum(z_exp,axis=1,keepdims=True)
    return res

class Softmax:
    def __init__(self):
        pass
    def cal(self,x):
        self.last_cal=softmax(x)
        return self.last_cal
    def deriv(self,x=None):
        if x:self.last_cal=self.cal(x)
        return self.last_cal*(1-self.last_cal)

# 定义均方误差（和交叉熵）误差函数
class MSE:
    def __init__(self):
        pass
    @staticmethod
    def cal(y_true, y_pred):
        return np.average(0.5 *((y_true-y_pred)**2).sum(axis=1))
    @staticmethod
    def backward(y_true, y_pred):
        return y_true-y_pred
class cross_entropy:
    def __init__(self):
        pass
    def cal(y_true, y_pred):
        pass

# 定义梯度更新方法(优化器类)，子类分别为不同的优化器
# SGD等

class Optimizer:
    def __init__(self,lr=0.001, decay=1, lr_min=0.00001, lr_max=1):
        self.lr = lr
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.iter = 0
    def update(self):
        self.iter += 1
        self.lr *= self.decay**(self.iter/1000)
        np.clip(self.lr, self.lr_min, self.lr_max)
class SGD(Optimizer):
    def __init__(self):
        Optimizer.__init__(self)
    def update(self):
        pass


class Layer:
    # 初始化神经网络层 初始化参数：输入节点数，输出节点数，权重矩阵，偏置向量
    def __init__(self, n_in, n_out, isLastLayer = False):
        self.n_in = n_in
        self.n_out = n_out
        self.weights = np.random.uniform(-0.001, 0.001, (n_in, n_out))
        self.bias = np.random.uniform(-0.001,0.001,n_out)
        # self.bias = np.ones((n_out))

        self.pre_in, self.net, self.out = None, None, None
        self.delt = None, None
        self.isLastLayer = isLastLayer

    # 该层前向传播 得到总输出向量
    def feedforward(self, inputs):
        net = np.dot(inputs, self.weights) + self.bias  # inputs和weight一个行向量一个列向量
        outputs = sigmoid(net)
        return net, outputs

    # 该层反向更新 last_delt为后一层的delt假设p个元素，w为后面一层的
    def backward(self,lr, last_delt=None, last_w=None):
        self.delt = last_delt
        if self.isLastLayer:
            self.delt = last_delt * deriv_sigmoid(self.net)
        else:
            self.delt = np.dot(last_delt,last_w.T)*deriv_sigmoid(self.net)
        self.weights -= lr*((self.pre_in[:,:,np.newaxis]*self.delt[:,np.newaxis]).sum(axis=0))
        self.bias -= lr*(self.delt.sum(axis=0))

class BPnn:
    # 初始化网络，隐层和输出层权重、偏置
    def __init__(self):
        self.layers = []

    # 添加一层
    def addLayer(self, layer):
        self.layers.append(layer)

    # 前向传播
    def forward(self, batch_in):
        inputs = batch_in
        for layer in self.layers:
            _, inputs = layer.feedforward(inputs)
        y_pre = inputs
        return y_pre

    # 测试样例
    def testdemo(self, batch_in):
        inputs = batch_in
        for layer in self.layers:
            _, inputs = layer.feedforward(inputs)
        y_pre = inputs
        print(np.argmax(y_pre))

    def cal_accuracy(self, X_data, Y_data):
        y_pred = self.forward(X_data)
        count = 0
        total = len(Y_data)
        for i in range(total):
            if np.argmax(y_pred[i]) == np.argmax(Y_data[i]):
                count += 1
        return count / total
        # 训练网络
    def train(self, X_train, Y_train, X_test, Y_test, lr=0.01, batch_size=16, epochs=100):
        dataset_size = len(X_train)
        print("init train loss:%f" % (loss_mse(Y_train, self.forward(X_train))))
        for epo in range(epochs):

            for t in range(0,dataset_size,batch_size):
                x_batch = X_train[t:t+batch_size]
                y_batch = Y_train[t:t+batch_size]  # bs*c

                bat_in =x_batch
                # 前向传播算出每层净输出和输出
                for layer in self.layers:
                    layer.pre_in = bat_in
                    layer.net, layer.out = layer.feedforward(layer.pre_in)
                    bat_in = layer.out

               #  反向传播更新权重和偏置
                last_delt = self.layers[-1].out-y_batch
                last_w = None

                for layer in self.layers[::-1]:
                    if layer.isLastLayer:
                        layer.backward(lr,last_delt)
                    else:
                        layer.backward(lr,last_delt,last_w)
                    last_w = layer.weights
                    last_delt = layer.delt
                # print("iter train loss:%f" % ( loss_mse(y_batch, self.forward(x_batch))))
            # 学习率衰减
            lr *= 0.99**epo
            if epo%1==0 :
                # train_loss = loss_mse(Y_train, self.forward(X_train))
                # test_loss = loss_mse(Y_test, self.forward(X_test))
                train_accuracy = self.cal_accuracy(X_train,Y_train)
                test_accuracy = self.cal_accuracy(X_test, Y_test)
                # print("epoch:%d, train loss:%f, test loss:%f, train accuracy:%f, test accuracy:%f" % (epo,train_loss,test_loss,train_accuracy,test_accuracy))
                print("epoch:%d,train accuracy:%f, test accuracy:%f" % (
                epo,train_accuracy, test_accuracy))

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self))
            print("trained network saved")

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            net = pickle.loads(f.read())
        return net