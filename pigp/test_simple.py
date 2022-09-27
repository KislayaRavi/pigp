""" Simple Tests:
Performs regression tests on a simple systems such as 

    1. D_x u = 2\pi cos(2\pi x)
    2. D_{xx} u = - 2\pi^2 sin(2\pi x)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pigp import *

def target_function(X):
    return np.sin(2 * np.pi * X)

def first_derivative(X):
    return 2. * np.pi * np.cos(2 * np.pi * X)

def second_derivative(X):
    return -(2*np.pi)**2 * np.sin(2*np.pi*X)

def validate_first_derivative():
    lower, upper = [0], [1]
    X = tf.convert_to_tensor(np.linspace(lower, upper, 4))
    Y = tf.convert_to_tensor(target_function(X))  

    # First Derivative PIGP
    FD = D(1, X, Y, lower, upper, first_derivative, num_latent=4,num_samples=100)
    FD.train(30)
    FD.plot_1d()    
    plt.savefig("figures/1st_derivative.png")

def validate_second_derivative():
    lower , upper = [0],[1]
    X = tf.convert_to_tensor(np.linspace(lower,upper,4))
    Y = tf.convert_to_tensor(target_function(X))
    
    # Second Derivative PIGP
    SD = D2(1,X,Y,lower,upper,second_derivative,num_latent=4,num_samples=100)
    SD.train(30)
    SD.plot_1d()
    plt.savefig("figures/2nd_derivative.png")


if __name__ == '__main__':
    validate_first_derivative()
    plt.clf()
    validate_second_derivative()