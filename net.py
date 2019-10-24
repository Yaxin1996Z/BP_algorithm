# coding utf8
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
        self.last_cal=np.maximum(x,0.0)
        return self.last_cal
    def deriv(self, x=None):
        if x:self.last_cal=self.cal(x)
        return 1.*np.logical_and(self.last_cal,1)

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
        return y_pred-y_true
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
        # 学习率衰减
        self.iter += 1
        # self.lr *= self.decay**(self.iter/1000)
        # np.clip(self.lr, self.lr_min, self.lr_max)
class SGD(Optimizer):
    def __init__(self):
        Optimizer.__init__(self)
    def update(self, w, b, g_w, g_b):
        w -= self.lr*g_w
        b -= self.lr*g_b
        super(SGD, self).update()
        return w, b

def draw_result(iter, train_loss, test_accuracy, title):
    plt.plot(iter, train_loss, '-', label = 'train_loss')
    # plt.plot(iter, test_loss, '-.', label = 'test_loss')
    # plt.plot(iter, train_accuracy, '-', label = 'train_accuracy')
    plt.plot(iter, test_accuracy, '-.', label = 'test_accuracy')
    plt.xlabel('n epochs')
    plt.legend(loc = 'upper right')
    plt.title(title)

    plt.show()

class Layer:
    # 初始化神经网络层 初始化参数：输入节点数，输出节点数，权重矩阵，偏置向量
    def __init__(self, n_in, n_out, activation = Sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.ac_fn = activation()
        self.weights = np.random.uniform(-0.0001, 0.0001, (n_in, n_out))
        self.bias = np.random.uniform(-0.0001,0.0001,n_out)
        # self.bias = np.ones((n_out))
        self.inputs = None
        self.g_w,self.g_b = None, None

    # 该层前向传播 得到总输出向量
    def feedforward(self, inputs):
        self.inputs = inputs
        net = np.dot(inputs, self.weights) + self.bias  # inputs和weight一个行向量一个列向量
        outputs = self.ac_fn.cal(net)
        return outputs

    # 该层反向更新 last_delt为后一层的delt假设p个元素，w为后面一层的
    def backward(self, pre_grad=None):
        delte = pre_grad * self.ac_fn.deriv()
        self.g_w = np.dot(self.inputs.T, delte)
        self.g_b = delte.sum(axis=0)
        next_grad = np.dot(delte, self.weights.T)
        return next_grad

class BPnn:
    # 初始化网络，隐层和输出层权重、偏置
    def __init__(self, cost=MSE, optimizer=SGD):
        self.layers = []
        self.cost = cost()
        self.optimizer = optimizer()
    # 添加一层
    def addLayer(self, layer):
        self.layers.append(layer)

    # 前向传播
    def forward(self, batch_in):
        inputs = batch_in
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        y_pre = inputs
        return y_pre

    # 测试样例
    def testdemo(self, batch_in):
        inputs = batch_in
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        y_pre = inputs
        print(np.argmax(y_pre))

    def cal_accuracy(self, X_data, Y_data):
        Y_pred = self.forward(X_data)
        rets = [(np.argmax(y_), np.argmax(y_true))for y_,y_true in zip(Y_pred,Y_data)]
        count = sum(t==y for t,y in rets)
        return  count/len(X_data)
        # 训练网络
    def train(self, X_train, Y_train, X_test, Y_test, learning_rate, batch_size=16, epochs=100):
        self.optimizer.lr = learning_rate
        dataset_size = len(X_train)
        print("init train loss:%f" % (self.cost.cal(Y_train, self.forward(X_train))))
        train_loss, test_accuracy = [], []
        for epo in range(epochs):

            for t in range(0,dataset_size,batch_size):
                x_batch = X_train[t:t+batch_size]
                y_batch = Y_train[t:t+batch_size]  # bs*c
                # y_pred = x_batch
                # for layer in self.layers:
                #     y_pred = layer.feedforward(y_pred)

                pre_grad = self.cost.backward(y_batch,self.forward(x_batch))
                for layer in self.layers[::-1]:
                    pre_grad = layer.backward(pre_grad)

               #  反向传播更新权重和偏置
                for layer in self.layers:
                    layer.weights,layer.bias = self.optimizer.update(layer.weights, layer.bias, layer.g_w, layer.g_b)

            # 学习率衰减
            if epo%1==0 :
                loss = self.cost.cal(Y_train, self.forward(X_train))
                acc = self.cal_accuracy(X_test,Y_test)
                train_loss.append(loss)
                test_accuracy.append(acc)
                print("epoch:%d, train loss:%f,test accuracy:%f" % (epo,loss, acc))
                # print("epoch:%d,train accuracy:%f, test accuracy:%f" % (epo, acc_train, acc_test))
        draw_result(range(epochs),train_loss,test_accuracy,'training curve')

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self))
            print("trained network saved")

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            net = pickle.loads(f.read())
        return net