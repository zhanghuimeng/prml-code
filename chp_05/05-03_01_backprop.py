# 5.3.2 Example of Back Propagation

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def xavier(m, h):
    return (np.random.rand(m, h) * 2 - 1) * math.sqrt(6 / (m + h))

class Model(object):
    def __init__(self, n_feature, n_hidden, n_output):
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.weight = []
        # Initialization: xavier
        self.weight.append(xavier(n_feature, n_hidden))
        self.weight.append(xavier(n_hidden, n_output))
    
    def forward(self, x):
        assert x.shape[1] == self.n_feature
        a = np.dot(x, self.weight[0])
        z = np.tanh(a)
        return np.dot(z, self.weight[1])
    
    def backprop(self, x, t, lr):
        # x.shape = n * D
        # a.shape = z.shape = n * M
        # w1.shape = D * K
        # w2.shape = K * M
        n = x.shape[0]
        assert x.shape[1] == self.n_feature
        a = np.dot(x, self.weight[0])
        assert(a.shape == (n, self.n_hidden))
        z = np.tanh(a)
        y = np.dot(z, self.weight[1])

        # delta1.shape = n * K
        # delta2.shape = n * M
        delta = []
        delta.append(np.zeros((n, self.n_hidden), dtype=np.float))
        delta.append(y - t)
        for i in range(n):
            for j in range(self.n_hidden):
                for k in range(self.n_output):
                    delta[0][i][j] += self.weight[1][j][k] * delta[1][i][k]
                delta[0][i][j] *= (1 - z[i][j]**2)
        
        # partial1.shape = n * D * K
        # partial2.shape = n * K * M
        partial = []
        partial.append(np.zeros((n, self.n_feature, self.n_hidden), dtype=np.float))
        partial.append(np.zeros((n, self.n_hidden, self.n_output), dtype=np.float))
        for i in range(n):
            for j in range(self.n_feature):
                for k in range(self.n_hidden):
                    partial[0][i][j][k] += delta[0][i][k] * x[i][j]
            for j in range(self.n_hidden):
                for k in range(self.n_output):
                    partial[1][i][j][k] += delta[1][i][k] * z[i][j]
        
        partial[0] = partial[0].mean(axis=0)
        assert(partial[0].shape == (self.n_feature, self.n_hidden))
        partial[1] = partial[1].mean(axis=0)

        self.weight[0] -= partial[0] * lr
        self.weight[1] -= partial[1] * lr

model = Model(1, 10, 1)
x = np.linspace(-3, 3, 100).reshape(100, 1)
y = np.sin(x)

for i in range(1000):
    model.backprop(x, y, 0.2)
    prediction = model.forward(x)
    print("step=%d, loss=%f" % (i, mean_squared_error(prediction, y)))

prediction = model.forward(x)
plt.plot(x, prediction, color="red", linewidth=1)
plt.scatter(x, y, s=1, color="blue")

plt.show()
