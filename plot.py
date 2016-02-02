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

jlabs = np.sum(np.absolute(p - theta.T.dot(x))) / N
jl2 = np.dot((p - np.dot(x.T,theta)).T, (p - np.dot(x.T, theta))) * (1./N)
jl1 = sqrt(jl2 * N) / N
jlinf = np.max(np.absolute(p - theta.T.dot(x)))

print "TP1"
print "jlabs =",jlabs
print "jl1 =",jl1
print "jl2 =",jl2
print "jlinfini =",jlinf

moindre_carre = np.linalg.inv(x.dot(x.T)).dot(x.dot(p))

print "Moindre Carres"
jlabs_mc = np.sum(np.absolute(p - moindre_carre.T.dot(x))) / N
jl2_mc = np.dot((p - np.dot(x.T,moindre_carre)).T, (p - np.dot(x.T, moindre_carre))) * (1./N)
jl1_mc = sqrt(jl2_mc * N) / N
jlinf_mc = np.max(np.absolute(p - moindre_carre.T.dot(x)))

print "jlabs =",jlabs_mc
print "jl1 =",jl1_mc
print "jl2 =",jl2_mc
print "jlinfini =",jlinf_mc

### Display datas
plt.plot(t, p, 'r.')
plt.plot(range(5, 16), [f(x) for x in range(5, 16)], "g")
plt.xlabel('time (sec)')
plt.ylabel('position (m)')
plt.show()