# 7.1.1 Using SMO algorithm to solve a soft margin SVM

import random
import numpy as np
import matplotlib.pyplot as plt

# k(x1, x2) kernel function
# (here just dot product)
def k(x1, x2):
    return x1[0] * x2[0] + x1[1] * x2[1]

# f(x) = w^T*x + b expressed in SVM terms
def f(x1, x, y, a, b):
    N = len(x1)
    ans = sum([a[i] * y[i] * k(x1, x[i]) for i in range(N)])
    return ans + b

# Simplifed SMO algorithm
# See http://cs229.stanford.edu/materials/smo.pdf
def SMO(C, tol, max_passes, x, y):
    N = len(x1)
    a = np.zeros(N, dtype=np.float)
    b = 0
    passes = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(N):
            Ei = f(x[i], x, y, a, b) - y[i]
            if (y[i] * Ei < -tol and a[i] < C) or (y[i] * Ei > tol and a[i] > 0):
                # select j != i randomly
                while True:
                    j = random.randint(0, N - 1)
                    if (j != i):
                        break
                Ej = f(x[j], x, y, a, b) - y[j]
                ai_old, aj_old = a[i], a[j]
                # Compute L <= aj <= H
                if y[i] != y[j]:
                    L = max(0, a[j] - a[i])
                    H = min(C, C + a[j] - a[i])
                else:
                    L = max(0, a[i] + a[j] - C)
                    H = min(C, a[i] + a[j])
                if L == H:
                    continue
                # Compute eta
                eta = 2 * k(x[i], x[j]) - k(x[i], x[i]) - k(x[j], x[j])
                if eta >= 0:
                    continue
                # Compute and clip a[j]
                a[j] = a[j] - y[j] * (Ei - Ej) / eta
                if a[j] > H:
                    a[j] = H
                elif a[j] < L:
                    a[j] = L
                if abs(a[j] - aj_old) < 1e-5:
                    continue
                # Compute a[i]
                a[i] = a[i] + y[i] * y[j] * (aj_old - a[j])
                # Compute b
                b1 = b - Ei - y[i] * (a[i] - ai_old) * k(x[i], x[i]) - y[j] * (a[j] - aj_old) * k(x[i], x[j])
                b2 = b - Ej - y[i] * (a[i] - ai_old) * k(x[i], x[j]) - y[j] * (a[j] - aj_old) * k(x[j], x[j])
                if 0 < a[i] < C:
                    b = b1
                elif 0 < a[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    
    return a, b

# Make noisy data
# Hyperplane: y = 1 - x
x1 = np.linspace(0, 1, 100)
x2 = np.random.uniform(0, 1, 100)
y = np.sign(x1 + x2 - 1)
noise = np.random.uniform(-0.2, 0.2, 100)
x1 += noise
x = np.column_stack((x1, x2))

# SMO algorithm
a, b = SMO(0.08, 1e-5, 2000, x, y)
print(a, b)

type1_x1 = []
type1_x2 = []
type2_x1 = []
type2_x2 = []
for i in range(len(x1)):
    if y[i] == 1:
        type1_x1.append(x1[i])
        type1_x2.append(x2[i])
    else:
        type2_x1.append(x1[i])
        type2_x2.append(x2[i])

plt.scatter(type1_x1, type1_x2, s=1, color="red")
plt.scatter(type2_x1, type2_x2, s=1, color="blue")

# Draw the classifer
N = len(a)
w = sum([a[i] * y[i] * x[i] for i in range(N)])
a0 = - w[0] / w[1]
b0 = - b / w[1]
y0 = a0 * x + b0
plt.plot(x, y0, color="green")

plt.show()
