import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

### read datas
p = open("p.txt", "r")
t = open("t.txt", "r")
p = np.loadtxt('p.txt')
t = np.loadtxt('t.txt')
###

N = 100.
x = np.array([np.array([i, 1]) for i in t]).T

theta = np.array([10., 8.])
alpha = 0.001

for i in range(1, 1000):
	theta = theta + alpha / N * np.dot(x, p - np.dot(x.T, theta))
	print(theta)