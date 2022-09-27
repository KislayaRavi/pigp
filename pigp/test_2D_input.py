""" One dimensional Diffusion Equation
\D_{t} - D_{xx} = -sin(x)

x \in [0,2\pi]
t \in [0.0,4.0]

bcs = [u(0, x) ~ sin(x),
            u(t, 0) ~ 0,
            u(t, 2π) ~ 0]
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pigp import *

# Import the data from the 1D diffusion equation
file = h5py.File("data/1d_diffusion","r")
xdata = file["x"][:]
tdata = file["t"][:]
u = file["u"][:]
size = u.shape


"""
    Assume that GP does not explictly depend on x.
    That is dependence is on t alone.
    For testing fit!
"""
def validation_domain_agnostic():
    # Select a single location on the domain and perform this experiment.
    xidx = 6 # Middle of the 1D domain.

    # Input data to a PIGP
    T = tdata
    U = u[xidx,:]  

    # Derivative
    def rhs(X):
        return -np.sin(X)

    tmin,tmax = T.min(),T.max()
    xmin,xmax = xdata.min(),xdata.max()

    T_init = tf.convert_to_tensor(T[0:-1:2])
    U_init = tf.convert_to_tensor(U[0:-1:2])

    print(T_init,U_init)

    DFGP = Diffusion1D(1,T_init,[xdata[xidx]],U_init,(xmin,xmax),(tmin,tmax),rhs,num_latent=4,num_samples=100)
    DFGP.train(30)
    DFGP.plot_1d()
    plt.savefig(f"figures/1D_Diffusion@index_{xidx}.png")

if __name__ == '__main__':
    validation_domain_agnostic()