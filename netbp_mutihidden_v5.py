# Layer类里自动更新权重
# 能定义隐层层数和每个隐层的单元个数
import numpy as np
from load_mnist import load_train_images,load_train_labels
from sklearn.model_selection import train_test_split
import os
import pickle
from net import BPnn,Layer
import matplotlib.pyplot as plt

def f(y):
    act = np.zeros((10))
    act[int(y[0])] = 1
    return act

batch_size = 100
# 设定学习率
learning_rate = 0.0001
# 设定训练轮数
epochs = 50
dataset_size=60000

def draw_result(iter, train_loss, test_loss, train_accuracy, test_accuracy, title):
    plt.plot(iter, train_loss, '-', label = 'train_loss')
    plt.plot(iter, test_loss, '-.', label = 'test_loss')
    plt.plot(iter, train_accuracy, '-', label = 'train_accuracy')
    plt.plot(iter, test_accuracy, '-.', label = 'test_accuracy')
    plt.xlabel('n iteration')
    plt.legend(loc = 'upper right')
    plt.title(title)

    plt.show()

if __name__ == '__main__':

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

    X_train, X_test, Y_train, Y_test = train_test_split(samples, samples_result, test_size=0.3, random_state=0)

    saveNetPath = "./net_trained/net_layer500_bs100_lr0.0001_initw0.001.pkl"
    if os.path.exists(os.path.abspath(saveNetPath)):
        bp = BPnn.load(saveNetPath)
        print("continue train")
    else:
        bp = BPnn()
        bp.addLayer(Layer(28*28,500))
        bp.addLayer(Layer(500,10,True))
    bp.train(X_train,Y_train,X_test,Y_test,learning_rate,batch_size,epochs)
    bp.save(saveNetPath)
