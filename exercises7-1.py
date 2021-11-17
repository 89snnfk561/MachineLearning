import numpy as np
import matplotlib

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

def mse_loss(y_trues, y_preds):
    return ((y_trues - y_preds) ** 2).mean()


class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class NeuronNetwork:

    def __init__(self):
        weights = np.array([0.5, 0.5])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforword(self, inputs):
        out_h1 = sigmoid(self.h1.feedforward(inputs))
        out_h2 = sigmoid(self.h2.feedforward(inputs))
        out_o1 = sigmoid(self.o1.feedforward(np.array(out_h1, out_h2)))
        return out_o1

    def train(self, data, y_trues):
        learn_rate = 0.001
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                # --- Do feedforward ---
                sum_h1 = self.h1.feedforward(x)
                out_h1 = sigmoid(sum_h1)
                sum_h2 = self.h2.feedforward(x)
                out_h2 = sigmoid(sum_h2)

                sum_o1 = self.h1.feedforward(x)
                out_o1 = sigmoid(sum_o1)
                y_pred = out_o1

                # --- Calculate partial derivatives ---
                # --- Naming d_L_d_w1 represent "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_o1w1 = out_h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_o1w2 = out_h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_o1b1 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.o1.weights[0] * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.o1.weights[1] * deriv_sigmoid(sum_o1)


