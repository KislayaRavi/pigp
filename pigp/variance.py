# %% 
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# %% 
# Util functions 
def zero(A):
    s = A.shape
    return np.zeros(s)

def tensorize(data):
    return (tf.convert_to_tensor(data[0]),tf.convert_to_tensor(data[1]))

def initialize_gp(data):
    x,y = tensorize(data)
    kernel = gpf.kernels.SquaredExponential(variance=1, lengthscales=1)
    gp = gpf.models.GPR(data=(x,y), kernel=kernel)
    return gp

def initialize_gaussian(mean,correlation):
    dist =  tfp.distributions.MultivariateNormalFullCovariance(loc=mean,)
    return dist

# %% 
# Define evidence lower bound
def ELBO(residual,nsamples,prior,approx,gp_z,f):
    # loss = 0.0
    # for samples in range(nsamples):
        # u_s = approx.sample()
        # t1 = approx.log_prob(u_s)
        # t3 = prior.log_prob(u_s)
        # t2 = residual(gp_z,f)
        # loss += -t2+t3-t1
    # loss /= nsamples
    loss = -residual(gp_z,f)
    return loss 

# %% 
# Data
npoints = 100
xinput = np.linspace(0,2*np.pi,npoints)
yinput = np.sin(2*xinput)
plt.plot(xinput,yinput)
plt.title("Input data")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("variance_outputs/input_data.png")

# %% 
# Define grids - use linspace grids for now.
nlatent_points = 20
nsamping_points = 100
xmin,xmax = xinput.min(),xinput.max()
z = np.linspace(xmin,xmax,nlatent_points).reshape(-1,1)
x = np.linspace(xmin,xmax,nsamping_points).reshape(-1,1)

# %% 
# Governing equation
f = lambda x : 2*tf.math.cos(2*x) # With gradient

# Define a loss function that is propotional to the likelihood
def residual(gp_z,f):
    xtensor = tf.convert_to_tensor(x)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xtensor)
        mean,_ = gp_z.predict_f(xtensor)
    grad = tape.gradient(mean,xtensor)
    return tf.norm(grad - f(xtensor))

# %% 
# Define dummy data for latent gp
y_z = zero(z)

# Define latent GP
gp_z = initialize_gp(data=(z,y_z))
Kzz = gp_z.kernel(z,z)
plt.imshow(Kzz)
# %% 

# Intial parameters for multivariate distributions
mu = tf.Variable(np.zeros(len(z)).reshape(-1,1))
Ktemp = np.random.rand(np.shape(z)[0],np.shape(z)[0])
Kguess = tf.Variable(np.ones(len(z)))

# %% 
# Define prior and approximation gaussians
prior = tfp.distributions.MultivariateNormalFullCovariance(loc=np.zeros(len(z)),covariance_matrix=Kzz)
approx_gauss = tfp.distributions.MultivariateNormalDiag(loc=mu,scale_diag=Kguess)
nsamples = 100 # For computing the expectation

# %% 
# Define TF parsable loss function
def loss():
    gp_z = initialize_gp(data=(z,mu))

    # Update the length scale parameter
    gp_z.likelihood.variance.assign(1e-5)
    gpf.utilities.set_trainable(gp_z.likelihood.variance,False)
    opt = gpf.optimizers.Scipy()
    _  = opt.minimize(loss,gp_z.trainable_variables,options=dict(maxiter=1)) 
    
    return -ELBO(residual,nsamples,prior,approx_gauss,gp_z,f)

# %% 
# Optimizer
nepochs = 10
optimizer = tf.optimizers.Adam()
for epoch in range(nepochs):
    print(f"Epoch {epoch}\t:")
    optimizer.minimize(loss,var_list=[mu])  
    print(f"{loss()}")

# %% Test - LIkelihood only
gp_z = initialize_gp(data=(z,mu))
pred_likelihood,_ = gp_z.predict_f(x.reshape(-1,1))
plt.plot(x,pred_likelihood)
plt.plot(xinput,yinput,"ok")
