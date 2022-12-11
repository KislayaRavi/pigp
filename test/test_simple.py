# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pigp import *

# Turn off warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# %%
def target_function(X):
    return np.sin(2 * np.pi * X)

def first_derivative(X):
    return 2. * np.pi * tf.math.cos(2 * np.pi * X)

# %%
lower, upper = [0], [1]
X = np.linspace(lower, upper, 10)
Y = target_function(X)  
num_iters = 100

# %%
# Compute loss
pgp = D((X,Y),Sobol,8,16,first_derivative)
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
xtest = np.linspace(lower[0],upper[0],100).reshape(-1,1)
ytest,_ = pgp.pigp.predict_f(xtest)

# %% 
plt.plot(X,Y,"*r")
latent_points = pgp.latent_grid["x"].numpy()
plt.plot(latent_points,np.zeros(latent_points.shape),"ob")
plt.plot(xtest,ytest,"*k")
plt.legend(["Data","Latent Points","PIGP"])
plt.show()