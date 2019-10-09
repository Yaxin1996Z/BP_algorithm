import os
import matplotlib.pyplot as plt
# import numpy as np
from net import BPnn,Layer
import pickle

if __name__ == '__main__':
    if os.path.exists(os.path.abspath("net.pkl")):
        bp = BPnn.load("net.pkl")

    pkl_file = open('mnist.pkl', 'rb')
    samples = pickle.load(pkl_file)
    samples_result = pickle.load(pkl_file)
    pkl_file.close()
    for i in range(10):
        plt.imshow(samples[i].reshape([28,28]), cmap='gray')
        plt.show()
        # 测试样本 [ 5.  0.  4.  1.  9.  2.  1.  3.  1.  4.] 前10张
        testsamples = samples[i]
        bp.test(testsamples)  # 输出应该是1,0

