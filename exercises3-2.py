import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans
from sklearn import datasets

colour = ["red", "blue", "green", "yellow", "purple", "orange", "white"]

class Iris():

    def __init__(self):
        self.generate()

    def generate(self):
        iris = datasets.load_iris()
        self.data = np.array(iris.data)
        self.y = iris.target
        print(self.data)
        print(self.y)

    def select(self, x, y):
        # l = len(self.data)
        # temp = np.array(self.data[:, x])
        # temp2 = np.array(self.data[:, y])
        # self.data = np.append(temp, temp2)
        #
        # self.data = self.data.reshape(2, -1)
        self.data = iris.data[:, [x, y]]
        self.data = self.data.T
        print(self.data)


def lostmetrix(ydata, label):
    metrix = np.array([])
    listv = np.array([0, 0, 0])
    for i, j in enumerate(label):
        if i != 0:
            if ydata[i] != ydata[i - 1]:
                metrix = np.append(metrix, listv)
                listv = np.array([0, 0, 0])
            else:
                listv[j] += 1
    metrix = np.append(metrix, listv)
    metrix = metrix.reshape(3, 3)
    return metrix


if __name__ == '__main__':
    iris = Iris()
    data_num = len(iris.y)

    x, y = map(int, input('選擇參考: ').split())
    iris.select(x, y)

    for i in range(data_num):
        plt.scatter(iris.data[0][i], iris.data[1][i])
    plt.show()

    cluster_num = 3
    center, u, u0, d, jm, p, fpc = cmeans(iris.data, m=2, c=cluster_num, error=0.005, maxiter=1000)

    for i in u:
        label = np.argmax(u, axis=0)  # 0:縱向，1:橫向
    print(label)

    for i, j in enumerate(label):
        plt.scatter(iris.data[0][i], iris.data[1][i], c=colour[j])

    metrix = lostmetrix(iris.y, label)
    correct = 0
    for i in range(3):
        correct += max(metrix[i])
    print(correct/len(iris.y))

    for i in range(cluster_num):
        plt.scatter(center[i][0], center[i][1], c='black')
    plt.show()

