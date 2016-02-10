import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

##### READ ######

p = open("p.txt", "r")
t = open("t.txt", "r")
p = np.loadtxt('p.txt')
t = np.loadtxt('t.txt')

#################


### VARIABLES ###

def f(x):
    return 2*x + 3

theta = np.array([2, 3])
N = 100.
x = np.array([np.array([i, 1]) for i in t]).T

alpha = 0.001
MAX_IT = 10

#################

def jlabs() :
	return np.sum(np.absolute(p - theta.T.dot(x))) / N

def jl2() :
	return np.dot((p - np.dot(x.T,theta)).T, (p - np.dot(x.T, theta))) * (1./N)

def jl1() :
	return sqrt(jl2() * N) / N

def jlinf() :
	return np.max(np.absolute(p - theta.T.dot(x)))

def moindre_carre() :
   return np.linalg.inv(x.dot(x.T)).dot(x.dot(p))

def jlabs_mc() :
	return np.sum(np.absolute(p - moindre_carre().T.dot(x))) / N

def jl2_mc() :
	return np.dot((p - np.dot(x.T,moindre_carre())).T, (p - np.dot(x.T, moindre_carre()))) * (1./N)

def jl1_mc() :
	return sqrt(jl2_mc() * N) / N

def jlinf_mc() :
	return np.max(np.absolute(p - moindre_carre().T.dot(x)))

def descenteGradiant() :
	theta = np.array([10., 8.])
	tab = [theta]

	for i in range(1, 1000):
		theta = theta + alpha / N * np.dot(x, p - np.dot(x.T, theta))
		print(theta)
		tab.append(theta)
	return tab

#### Prints #####

print "TP1", theta
print "jlabs =", jlabs()
print "jl1 =", jl1()
print "jl2 =", jl2()
print "jlinfini =", jlinf()

print "Moindre Carres", moindre_carre()
print "jlabs =", jlabs_mc()
print "jl1 =", jl1_mc()
print "jl2 =", jl2_mc()
print "jlinfini =", jlinf_mc()

#################


#### Graphs #####

#plt.plot(t, p, 'ro')
#plt.plot(range(5, 16), [f(x) for x in range(5, 16)], "g", label="2x+3")
#plt.plot(range(5,16), [x*moindre_carre[0]+moindre_carre[1] for x in range(5,16)], "b", label="moindre carre")
DG = descenteGradiant() 
plt.plot(range(0, MAX_IT), [DG[i][0] for i in range(0, MAX_IT)], "r")
plt.plot(range(0, MAX_IT), [DG[i][1] for i in range(0, MAX_IT)], "g")
#plt.plot(t, np.dot(x.T, theta), "g")
#plt.axis([0, 10, -5, 5])
plt.legend()
plt.show()

#################