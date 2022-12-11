import h5py
from pigp import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def forcing_function(x):
    -tf.math.sin(4*x) * tf.math.cos(6*x)

filename = "data/1d_poisson.hdf5"
file = h5py.File(filename,"r")
x = file["x"][:]
y = file["u"][:]

# Munge to tensor formats
Xdata = tf.convert_to_tensor(x.reshape(-1,1))
Ydata = tf.convert_to_tensor(y.reshape(-1,1))
lower = x.min()
upper = x.max()
num_iterations = 100

