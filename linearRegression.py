#!/usr/bin/env python
import subprocess, os
from numpy import loadtxt, zeros, ones, arange
from pylab import plot, show, xlabel, ylabel

def costfunction(X, y, theta):
    #Number of training samples
    m = y.size
    predictions = X.dot(theta)
    sqErrors = (predictions - y)
    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):

    m = y.size
    J_history = zeros(shape=(num_iters, 1))
 
    for i in range(num_iters):
 
        predictions = X.dot(theta)
 
        theta_size = theta.size
 
        for it in range(theta_size):
 
            temp = X[:, it]
            temp.shape = (m, 1)
 
            errors_x1 = (predictions - y) * temp
 
            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()
 
        J_history[i, 0] = costfunction(X, y, theta)
 
    return theta, J_history


def loadTrainingData(degree, filepath):
    trainingdata = loadtxt(filepath, delimiter=',')
    
    X = trainingdata[:, :degree]
    y = trainingdata[:, degree]

    return X, y

X, y = loadTrainingData(6, os.path.join(os.getcwd(), "training.txt"))


m = y.size
y.shape = (m, 1)

it = ones(shape=(m, 7))
it[:, 1:7] = X

iterations = 100
alpha = .01

theta = zeros(shape=(7,1))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print "Using coefficients: ", theta[1:]

c = theta[1:]
c = [a for b in c for a in b]
pacmanloc = os.path.join(os.getcwd(), "multiagent/pacman.py")
os.chdir(os.path.join(os.getcwd(), "multiagent"))
betterArg = "evalFn=better,a={},b={},c={},d={},e={},f={}".format(c[0], c[1], c[2], c[3], c[4], c[5])
command = ["python", "pacman.py", "-l", "smallClassic", "-p", "ExpectimaxAgent", "-a", betterArg, "-n", "10", "-q"]
print ' '.join(command)
output = subprocess.check_output(command)
print output

plot(arange(iterations), J_history)
xlabel('iterations')
ylabel('cost')
show()
