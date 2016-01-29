import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

### read datas
p = open("p.txt", "r")
t = open("t.txt", "r")
p = np.loadtxt('p.txt')
t = np.loadtxt('t.txt')
###

def f(x):
    return 2*x + 3

theta = np.array([2, 3])
N = 100.
x = np.array([np.array([i, 1]) for i in t]).T
print(x)
print(np.shape(x))

jl2 = np.dot((p - np.dot(x.T,theta)).T, (p - np.dot(x.T, theta))) * (1./100)
print(jl2)

### Display datas
plt.plot(t, p, 'r.')
plt.plot(range(5, 16), [f(x) for x in range(5, 16)], "g")
plt.xlabel('time (sec)')
plt.ylabel('position (m)')
#test
#plt.show()

