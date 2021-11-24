import numpy as np
import neurolab as nl
import random
import matplotlib.pyplot as plt

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_trues = np.array([[0], [1], [1], [0]])

net = nl.net.newff([[0, 1], [0, 1]], [2, 1])
error = net.train(inputs, y_trues, epochs=2000, show=100, goal=0.0001)

y_preds = np.array([])
for x in inputs:
    y_pred = net.sim([[x[0], x[1]]])
    print(y_pred)
    y_preds = np.append(y_preds, y_pred)


plt.title("lost")
plt.plot(error)
plt.grid()
plt.show()