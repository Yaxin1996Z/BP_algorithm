# Layer类里自动更新权重
# 能定义隐层层数和每个隐层的单元个数

import numpy as np
from load_mnist import load_train_images,load_train_labels
import os
import pickle
from net import BPnn,Layer


def f(y):
    act = np.zeros((10))
    act[int(y[0])] = 1
    return act


if __name__ == '__main__':

    # samples = [[-2, -1], [25, 6], [17, -4], [-15, 4]]
    # samples_result = np.array([[0,0],[1,1],[1,0],[0,1]])
    d, c = 28 * 28, 10
    dataset_size = 60000
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
        bp = BPnn()
        bp.addLayer(28*28,30)
        bp.addLayer(30,10)
    bp.train(samples, samples_result)
    bp.save('./net_trained/net.pkl')

    # testsamples = samples[3]
    # bp.test(testsamples)  # 输出应该是1,0
    # 测试样本
    # testsamples = [[-7, -3], [20, 2]]
    # bp.test(testsamples)  # 输出应该是1,0