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
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import h5py
from pigp import *

# Import the data from the 1D diffusion equation solver.
file = h5py.File("data/1d_diffusion","r")
xdata = file["x"][:]
tdata = file["t"][:]
u = file["u"][:]
size = u.shape

def rhs(X):
    return -np.sin(8*X) -np.cos(6*X)

def validation_domain_aware(): 
    x = xdata[0:127:10]
    t = tdata[0:26:5]
    I = merge(x,t)
    O_temp = u[0:127:10,0:26:5]
    O = O_temp.flatten().reshape(-1,1)

    num_samples = 30
    num_latents = 15

    X = I
    Y = tf.convert_to_tensor(O)
    df = Diffusion1D(2,X,Y,rhs,num_latent=num_latents,num_samples=num_samples)

    
    df.train(int(1e2))


    surfX,surfY = np.meshgrid(t,x)
    Z = O_temp
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    Z1 = df.pigp.predict_f(I)[0].numpy().reshape(13,6)
    Error = Z1 - Z
    ax.plot_surface(surfX, surfY, Error,cmap='viridis', edgecolor='none')
    ax.set_title("Error in data points with PIGP")
    plt.savefig("figures/Error_2.png")



    # Validation with plots here.
    for i in range(len(t)):
        fig,axs = plt.subplots(1)
        axs.plot(x,O_temp[:,i],"g+",label=f"{t[i]}")

        # samples = df.input_samples
        # ypred = df.pigp.predict_f(samples)[0].numpy().reshape(num_samples,num_samples)
        # xsamples = df.xsamples
        # tsamples = df.tsamples
        # print(ypred.shape)
        
        samples = I
        ypred = df.pigp.predict_f(samples)[0].numpy().reshape(13,6)
        xsamples = x
        tsamples = t
        
        axs.plot(xsamples,ypred[:,i],label=f"{tsamples[i]}")
        axs.set_ylim([-1,1])
        axs.set_title("Data vs Prediction")
        axs.legend()
        plt.savefig(f"figures/heat_pigp/{i}.png")
        plt.clf()


if __name__ == '__main__':
    validation_domain_aware()