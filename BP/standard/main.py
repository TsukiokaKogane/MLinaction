from sklearn import datasets
import numpy as np
from random import SystemRandom


def sigmod(x):
    return 1.0 / (np.exp(-x) + 1.0)


def squared_error(x, y):
    assert len(x) == len(y)
    n = len(x)
    err = 0.0
    for i in range(n):
        err += (x[i] - y[i]) ** 2
    return 0.5 * err


class Net(object):
    def __init__(self, d, q, ll, eta):
        rand = SystemRandom()
        self.d = d
        self.q = q
        self.ll = ll
        self.eta = eta

        self.v = [[[rand()] for i in range(q)] for j in range(d)]
        self.gamma = [[rand()] * q]
        self.w = [[[rand()] for i in range(ll)] for j in range(q)]
        self.theta = [[rand()] * ll]

    def forward(self, x, y):
        assert isinstance(x, list) == True
        assert len(x) == self.d
        alpha = [[0.0] for i in range(self.q)]
        alpha = np.array(alpha)
        for i in range(self.d):
            for h in range(self.q):
                alpha[h] += self.v[i][h] * x[i]
        b = [[0.0] for i in range(self.q)]
        b = np.array(b)
        for h in range(self.q):
            b[h] += sigmod(alpha[h] - self.gamma[h])

        beta = [[0.0] for i in range(self.ll)]
        for j in range(self.ll):
            for h in range(self.q):
                beta[j] += self.w[h][j] * b[h]

        y_ = [[0.0] for i in range(self.ll)]
        y_ = np.array(y_)
        for j in range(self.ll):
            y_ = sigmod(beta[j] - self.theta[j])

        g = [[0.0] for i in range(self.ll)]
        for j in range(self.ll):
            g[j] = y_[j] * (y_[j] - y[j]) * (1.0 - y_[j])

        e = [[0.0] for i in range(self.q)]
        for h in range(self.q):
            for j in range(self.ll):
                e[h] += g[j] * self.w[h][j] * b[h] * (1.0 - b[h])

        return y_, g, e, b

    def backward(self, g, e, b, x):
        for h in range(self.q):
            for j in range(self.ll):
                self.w[h][j] = self.w[h][j] - self.eta * g[j] * b[h]

        for j in range(self.ll):
            self.theta[j] = self.theta[j] + self.eta * g[j]

        for i in range(self.d):
            for h in range(self.q):
                self.v[i][h] = self.v[i][h] - self.eta * e[h] * x[i]

        for h in range(self.q):
            self.gamma[h] = self.gamma[h] + self.eta * e[h]



if __name__ == '__main__':
    iris = datasets.load_iris()
    irisFeatures = iris["data"]
    irisFeaturesName = iris["feature_names"]
    irisLabels = iris["target"]

    print('Iris feature name:', irisFeaturesName)
    print('Iris data size :', irisFeatures.shape)
    print('Iris label size :', irisLabels.shape)

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for i in range(3):
        for j in range(40):
            train_data.append(irisFeatures[i * 50 + j])
            train_label.append(irisLabels[i * 50 + j])
        for j in range(10):
            test_data.append(irisFeatures[i * 50 + 40 + j])
            test_label.append(irisLabels[i * 50 + 40 + j])

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

