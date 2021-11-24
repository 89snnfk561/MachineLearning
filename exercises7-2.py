# sigmoid activation function for logic operation

import numpy as np
import random
import matplotlib.pyplot as plt

# 產生學習資料
x = np.array([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 0.]])
x[:, 2] = 1.
d = np.array([0., 1., 1., 0.]).T

# 初始weights
wh = np.array([[1., 1., 1.], [1., 1., 1.]])     # hidden layer
wo = np.array([1., 2., 1.])                     # output layer

# 鍵盤輸入學習迴圈數及數度
# loops = eval(input('學習迴圈數:'))
# rate = eval(input('學習速度:'))

# 學習
ip = np.array([0., 0., 1.])
y = []
mse = []
loops = 2000
for loop in range(loops):   #epochs
    sum = 0
    for i in range(len(x)):
        for j in range(2):      #weight
            neth = np.dot(x[i], wh[j].T)
            ip[j] = 1 / (1 + np.exp(-1 * neth))
        #            print('neth=', neth, ip[j])
        neto = np.dot(ip, wo.T)
        #        print('neto',neto)
        y_pred = 1 / (1 + np.exp(-1 * neto))
        e = d[i] - y_pred
        sum += e ** 2
        print('dy=', d[i], 'y_pred=', y_pred, 'e=', e)
        #        print(e*o*(1-o)*ip)
        wo += 2 * e * y_pred * (1 - y_pred) * ip
        #        print('wo=',wo,'\n')

        for j in range(2):
            wh[j] += 2 * ip[j] * (1 - ip[j]) * x[i] * e * y_pred * (1 - y_pred) * wo[j]
    #        print(ip*(1-ip)*x[i]*e*o*(1-o)*wo)

    mse.append(np.sqrt(sum))

# 繪圖
plt.plot(mse)
plt.show()