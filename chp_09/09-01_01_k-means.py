# 9.1 K-means

import math
import numpy as np
import matplotlib.pyplot as plt

def classify(x, y, r):
    x_class = [[] for _ in range(K)]
    y_class = [[] for _ in range(K)]
    n = len(x)
    for i in range(n):
        x_class[r[i]].append(x[i])
        y_class[r[i]].append(y[i])
    return x_class, y_class

K = 2
x = []
y = []
with open("datasets/oldfaithful/oldfaithful.txt", "r") as f:
    for line in f:
        p = [float(a) for a in line.split()]
        x.append(p[0])
        y.append(p[1])

n = len(x)
x = np.array(x)
y = np.array(y)

# Standardize
x_mean = np.mean(x)
y_mean = np.mean(y)
x_range = np.max(x) - np.min(x)
y_range = np.max(y) - np.min(y)
x = (x - x_mean) / x_range
y = (y - y_mean) / y_range

# Start Center
x_center = [0.25, -0.25]
y_center = [-0.25, 0.25]
r = [0] * n
J_list = []
step = 1

while True:
    J = 0
    r_new = []
    for i in range(n):
        dist = []
        for j in range(K):
            dist.append((x[i] - x_center[j])**2 + (y[i] - y_center[j])**2)
        dist = np.array(dist)
        J += np.min(dist)
        r_new.append(np.argmin(dist))
    if r == r_new:
        break
    r = r_new
    # Calculate the new average centers
    x_class, y_class = classify(x, y, r)
    for i in range(K):
        x_center[i] = np.mean(np.array(x_class[i]))
        y_center[i] = np.mean(np.array(y_class[i]))
    # Show
    plt.subplot(2, 3, step)
    plt.axis("square")
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.scatter(x_class[0], y_class[0], color="red")
    plt.scatter(x_class[1], y_class[1], color="blue")
    plt.scatter(x_center[0], y_center[0], color="darkred", marker="x")
    plt.scatter(x_center[1], y_center[1], color="darkblue", marker="x")

    J_list.append(J)
    step += 1

plt.subplot(2, 3, 5)
plt.plot(J_list)
plt.show()
