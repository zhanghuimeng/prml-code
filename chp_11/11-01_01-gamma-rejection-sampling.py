# 11.1.1 Using Cauchy distribution and rejection sampling
# to sample from Gamma Distribution

import numpy as np
import scipy.stats as stats
from scipy.special import gamma as gamma_function
from matplotlib import pyplot as plt

def Gamma(x, a, b):
    return b**a * x**(a-1) * np.exp(-b*x) /  gamma_function(a)

def Cauchy(x, x0, g):
    return g**2 / (np.pi * (g**2 + (x - x0)**2))

a = 3
b = 2

# Draw PDF of Gamma and Cauchy distribution
x = np.linspace(0, 5, 100)
# yg = stats.gamma.pdf(x, a=a, scale=1/b)
yg = Gamma(x, a, b)

x0 = (a - 1) / b
g = np.sqrt(2*a - 1)
k = Gamma(x0, a, b) / Cauchy(x0, x0, g)
yc = Cauchy(x, x0, g)
yck = k * Cauchy(x, x0, g)

plt.plot(x, yg, "g-", label="Gamma Distribution")
plt.plot(x, yc, "r--", label="Cauchy Distribution")
plt.plot(x, yck, "r-", label = "k * Cauchy Distribution")

# Sample from Cauchy distribution
sample_cauchy = stats.cauchy.rvs(loc=x0, scale=g, size=5000)
bins = np.linspace(0, 5, 15)
plt.hist(sample_cauchy, bins=bins, density=True, color="red", alpha=0.5)

# Rejection Sample from Cauchy distribution
sample_gamma = []
for sample in sample_cauchy:
    maximal = k * Cauchy(sample, x0, g)
    u0 = np.random.uniform(low=0, high=maximal)
    if u0 < Gamma(sample, a, b):
        sample_gamma.append(sample)
plt.hist(sample_gamma, bins=bins, density=True, color="green", alpha=0.5)

plt.legend()
plt.show()
