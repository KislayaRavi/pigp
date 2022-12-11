# %% 
import h5py
from pigp import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %% 
def forcing_function(x):
    return -tf.math.sin(4*x) * tf.math.cos(6*x)

filename = "../data/1d_poisson.hdf5"
file = h5py.File(filename,"r")
x = file["x"][::100]
y = file["u"][::100]

# %% 
# Munge to tensor formats
Xdata = x.reshape(-1,1)
Ydata = y.reshape(-1,1)
num_iters = 100
nlatent = int(2**(np.floor(np.log2(Xdata.shape[0]))+2))
nsampling = 128

# %%
# Compute loss
pgp = D2((Xdata,Ydata),Sobol,nlatent,nsampling,forcing_function)
l = pgp.loss()

# %%
# Check if gradient is right
with tf.GradientTape(persistent=True) as tape:
    tape.watch(pgp.latent_grid["y"])
    l = pgp.loss()
grad_loss = tape.gradient(l,pgp.latent_grid["y"])

# %%
pgp.train(num_iters,10)

# %% 
# Make predictions
xtest = np.linspace(0,12,100).reshape(-1,1)
ytest,_ = pgp.pigp.predict_f(xtest)

# %% 
plt.plot(Xdata,Ydata,"*r")
latent_points = pgp.latent_grid["x"].numpy()
plt.plot(latent_points,np.zeros(latent_points.shape),"ob")
plt.plot(xtest,ytest,"*k")
plt.legend(["Data","Latent Points","PIGP"])
plt.show()
# %%
