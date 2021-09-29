import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans

colour = ["red", "blue", "green", "yellow", "purple", "orange", "white"]

if __name__ == '__main__':
    data_num = eval(input('請輸入資料量：'))
    train = np.random.uniform(1, 100, (2, data_num))
    print(train)
    for i in range(data_num):
        plt.scatter(train[0][i], train[1][i])
    plt.show()
    cluster_num = eval(input('請輸入群數：'))

    center, u, u0, d, jm, p, fpc = cmeans(train, m=2, c=cluster_num, error=0.005, maxiter=1000)

    for i in u:
        label = np.argmax(u, axis=0)  # 0:縱向，1:橫向
    print(label)
    for i, j in zip(range(data_num), label):

        plt.scatter(train[0][i], train[1][i], c=colour[j])

    for i in range(cluster_num):
        plt.scatter(center[i][0], center[i][1], c='black')
    plt.show()