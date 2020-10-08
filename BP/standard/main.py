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

        self.v = [[rand.random() for i in range(q)] for j in range(d)]
        self.gamma = [rand.random() for i in range(q)]
        self.w = [[rand.random() for i in range(ll)] for j in range(q)]
        self.theta = [rand.random() for i in range(ll)]


    def forward(self, x, y):
        # assert isinstance(x, list) == True
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

        beta = np.array([0.0 for i in range(self.ll)])

        for j in range(self.ll):
            for h in range(self.q):
                beta[j] += self.w[h][j] * b[h]

        y_ = np.array([0.0 for i in range(self.ll)])

        for j in range(self.ll):
            y_[j] = sigmod(beta[j] - self.theta[j])

        g = np.array([0.0 for i in range(self.ll)])
        for j in range(self.ll):
            g[j] = y_[j] * (y_[j] - y[j]) * (1.0 - y_[j])

        e = np.array([[0.0] for i in range(self.q)])
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

    def slow(self):
        self.eta = self.eta * 0.9


if __name__ == '__main__':
    iris = datasets.load_iris()
    irisFeatures = iris["data"]
    irisFeaturesName = iris["feature_names"]
    irisLabels = iris["target"]

    print('Iris feature name:', irisFeaturesName)
    print('Iris data size :', irisFeatures.shape)
    print('Iris label size :', irisLabels.shape)

    # normalization
    for i in range(irisFeatures.shape[1]):
        minn = np.min(irisFeatures[:, i])
        irisFeatures[:, i] -= minn
        maxx = np.max(irisFeatures[:, i])
        irisFeatures[:, i] /= (maxx - minn)
    #
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for i in range(3):
        for j in range(40):
            train_data.append(irisFeatures[i * 50 + j])
            new_label = np.array([0] * 3)
            new_label[irisLabels[i * 50 + j]] = 1
            train_label.append(new_label)
        for j in range(10):
            test_data.append(irisFeatures[i * 50 + 40 + j])

            new_label = np.array([0] * 3)
            new_label[irisLabels[i * 50 + 40 + j]] = 1
            test_label.append(new_label)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    # print(test_data)
    iteration = 1000
    cnt = 0
    train_n = len(train_data)
    test_n = len(test_data)
    net = Net(d=4, q=8, ll=3, eta=0.05)
    while cnt < iteration:
        err = 0.0
        for i in range(train_n):
            y_, g, e, b = net.forward(train_data[i, :], train_label[i, :])
            net.backward(g, e, b, train_data[i])
            err += squared_error(train_label[i], y_)
        if cnt % 100 == 0:
            print("iteration= " + str(cnt) + " err=" + str(err))
            # net.slow()
        cnt += 1

    correct = 0
    for i in range(test_n):
        y_, _, _, _ = net.forward(test_data[i, :], test_label[i, :])
        # print(y_)
        max_index = 0
        for j in range(1, 3):
            if y_[j] > y_[max_index]:
                max_index = j
        if test_label[i][max_index] == 1:
            correct += 1
    print("accuracy=" + str(correct/test_n))

