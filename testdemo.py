import os
import matplotlib.pyplot as plt
import numpy as np
from net import BPnn,Layer
from sklearn.model_selection import train_test_split
import pickle
# 均方误差
def loss_mse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.average(0.5 *((y_true-y_pred)**2).sum(axis=1))

if __name__ == '__main__':

    bp = BPnn.load("./net_trained/net_layer2020_bs16_lr0.01_initw0.05.pkl")

    pkl_file = open('mnist.pkl', 'rb')
    samples = pickle.load(pkl_file)
    samples_result = pickle.load(pkl_file)
    pkl_file.close()

    X_train, X_test, Y_train, Y_test = train_test_split(samples, samples_result, test_size=0.3, random_state=0)
    train_loss = loss_mse(Y_train, bp.forward(X_train))
    test_loss = loss_mse(Y_test, bp.forward(X_test))
    train_accuracy = bp.cal_accuracy(X_train, Y_train)
    test_accuracy = bp.cal_accuracy(X_test, Y_test)
    print("train loss:%f, test loss:%f, train accuracy:%f, test accuracy:%f" % (train_loss, test_loss, train_accuracy, test_accuracy))
    # for i in range(10):
    #     plt.imshow(samples[i].reshape([28,28]), cmap='gray')
    #     plt.show()
    #     # 测试样本 [ 5.  0.  4.  1.  9.  2.  1.  3.  1.  4.] 前10张
    #     testsamples = samples[i]
    #     bp.testdemo(testsamples)  # 输出应该是1,0

