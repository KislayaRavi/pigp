from abc import abstractmethod
from mimetypes import init
from re import I
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt

from .utils import *


class PIGP():

    #----------------------------------------------------------------#
    #------------------------Contructor------------------------------#
    #----------------------------------------------------------------#

    def __init__(self, data, sampler, nlatent, nsampling,f):
        # Assignments 
        self.X, self.Y = data
        self.nlatent_points, self.nsampling_points = nlatent,nsampling
        self.sampler = sampler
        self.lbounds,self.ubounds = find_bounds(self.X)
        self.f = f

        # Actions
        self.latent_grid, self.sampling_grid = self.create_grids()
        self.latent_gp, self.pigp = self.create_gps()
        

    #----------------------------------------------------------------#
    #------------------------Internal Functions----------------------#
    #----------------------------------------------------------------#
    def create_grid(self,lbounds,ubounds,n):
        # Create grid
        grid_x = sampler(lbounds,ubounds,n)
        grid_y = np.zeros(grid_x.shape)        
        grid = Dict()
        grid["x"] = latent_x
        grid["y"] = latent_y
        return grid

    def latent_gp(self):
        pass

    def pigp(self):
        pass

    def create_grids(self):
        latent_grid = self.create_grid(self.lbounds,self.ubounds,self.nlatent_points)
        sampling_grid = self.create_grid(self.lbounds,self.ubounds,self.nsampling_points)
        return (latent_grid,sampling_grid)

    def create_gps(self):
        pass

    def local_train(self):
        pass 

    #----------------------------------------------------------------#
    #------------------------Interfaces------------------------------#
    #----------------------------------------------------------------#

    @abstractmethod
    def loss():
        pass 

    def train():
        pass


class PIGP():

    def __init__(self, dim, init_data_X, init_data_Y, lower_bound, upper_bound, function_rhs, num_latent=25, num_samples=500,*args):
        self.dim = dim
        self.kr_temp = gpflow.kernels.SquaredExponential(variance=1, lengthscales=1)
        self.kr = gpflow.kernels.SquaredExponential(variance=1, lengthscales=1)
        self.gp_temp = gpflow.models.GPR(data=(init_data_X, init_data_Y), kernel=self.kr_temp)
        self.ARD(self.gp_temp)
        self.latent_X = tf.convert_to_tensor(np.linspace(lower_bound, upper_bound, num_latent)) # TODO: have a look at this all over again`
        # self.latent_X = tf.convert_to_tensor(np.random.uniform(lower_bound, upper_bound, (num_latent, dim)))
        self.samples_X = tf.convert_to_tensor(np.linspace(lower_bound, upper_bound, num_samples))
        mean, _ = self.gp_temp.predict_f(self.latent_X)
        self.latent_Y = tf.Variable(mean.numpy())
        self.function_rhs = function_rhs
        self.rhs_sample = tf.convert_to_tensor(function_rhs(self.samples_X))
        self.create_pigp(self.latent_Y)
        # self.ARD(self.pigp)  # Something fishy happens when performing ARD, check it later

    def update_pigp(self, latent_Y):
        # temp_X, temp_Y = tf.concat([self.latent_X], 0), tf.concat([latent_Y], 0)
        # self.pigp.data = (temp_X, temp_Y)
        latent_x = self.pigp.data[0]
        self.pigp.data = (latent_x, latent_Y)
    
    def create_pigp(self, latent_Y):
        # temp_X, temp_Y = tf.concat([bc_x, self.latent_X], 0), tf.concat([bc_y, latent_Y], 0)
        self.pigp = gpflow.models.GPR(data=(self.latent_X, latent_Y), kernel=self.kr)

    def pigp_hyperpaprameter_optimize(self):
        self.pigp.likelihood.variance.assign(1e-5)
        gpflow.utilities.set_trainable(self.pigp.likelihood.variance, False)
        # gpflow.utilities.set_trainable(self.pigp.data[1], True)
        opt = gpflow.optimizers.Scipy()
        # print(model.training_variables)
        opt_logs = opt.minimize(self.loss, 
                                self.pigp.trainable_variables, 
                                options=dict(maxiter=1))

    def ARD(self, model):
        model.likelihood.variance.assign(1e-5)
        gpflow.utilities.set_trainable(model.likelihood.variance, False)
        opt = gpflow.optimizers.Scipy()
        # print(model.training_variables)
        opt_logs = opt.minimize(model.training_loss, 
                                model.trainable_variables, 
                                options=dict(maxiter=100))
    
    @abstractmethod
    def loss(self):
        pass

    def train(self, num_epochs, frequency=50, train_hyperparameter=False):
        optimiser = tf.keras.optimizers.Adam()
        tf.print(f"Initial loss {self.loss()}")
        for epoch_id in range(1, num_epochs+1):
            optimiser.minimize(self.loss, self.latent_Y)
            self.pigp_hyperpaprameter_optimize()
            if epoch_id % 10 == 0:
                tf.print(f"Epoch {epoch_id}: Residual (train) {self.loss()}") 
            
    
    def plot_1d(self):
        plt.plot(self.samples_X, self.pigp.predict_f(self.samples_X)[0].numpy(), 'ko', label='PIGP')
        plt.plot(self.latent_X.numpy(), self.latent_Y.numpy(), 'r+',label='Latent points')
        plt.legend()

    def plot_initial_estimate(self):
        plt.plot(self.gp_temp.data[0].numpy(),self.gp_temp.data[1].numpy(),"ro",label="Data")
        plt.plot(self.samples_X,self.gp_temp.predict_f(self.samples_X)[0].numpy(),'k-',label="Intial GP")
        plt.plot(self.latent_X.numpy(),self.gp_temp.predict_f(self.latent_X)[0].numpy(),"g+",label="Latent GP")
        plt.legend()

class D(PIGP):

    def __init__(self, dim, init_data_X, init_data_Y, lower_bound, upper_bound, function_rhs, num_latent=20, num_samples=100, *args):
        super().__init__(dim, init_data_X, init_data_Y, lower_bound, upper_bound, function_rhs, num_latent, num_samples, *args)

    def loss(self):
        self.update_pigp(self.latent_Y)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.samples_X)
            mean, _ = self.pigp.predict_f(self.samples_X)

        grad_mean = tape.gradient(mean, self.samples_X)
        loss_term = tf.math.reduce_mean(tf.square(self.rhs_sample - grad_mean))
        return loss_term

class D2(PIGP):
    def __init__(self,dims,init_data_X,init_data_Y,lower_bound,upper_bound,function_rhs,num_latent=20,num_samples=500,*args):
        super().__init__(dims,init_data_X,init_data_Y,lower_bound,upper_bound,function_rhs,num_latent,num_samples,*args)
    
    ## TODO : Test the second derivative.
    def loss(self):
        self.update_pigp(self.latent_Y)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.samples_X)
            mean,_ = self.pigp.predict_f(self.samples_X)
            gradient = tape.gradient(mean,self.samples_X)
        laplacian = tape.gradient(gradient,self.samples_X)
        loss_term = tf.math.reduce_mean(tf.square(self.rhs_sample - laplacian))
        return loss_term

## Combination and product .... 
## Throw this away and implment this here.
def merge(x,t):
    I = np.zeros((len(x)*len(t),2))
    k = 0
    for i,elemx in enumerate(x):
        for j,elemt in enumerate(t):
            I[k,0] = elemx
            I[k,1] = elemt
            k += 1 
    return tf.convert_to_tensor(I)

def unfold(func,samples):
    n = len(samples)
    Z = np.zeros((n*n,))
    for i in range(n):
        Z[n*i:n*(i+1)] = func(samples)
    return Z

class Diffusion1D(PIGP): 
    def __init__(self,dims,xinit,yinit,rhs,num_latent=25,num_samples=50):
        self.dims = dims
        self.num_latent = num_latent
        self.num_samples = num_samples

        # Original grid
        self.input = xinit
        self.output = yinit
        self.xmin, self.xmax = xinit.numpy()[:,0].min(),xinit.numpy()[:,0].max()
        self.tmin, self.tmax = xinit.numpy()[:,1].min(),xinit.numpy()[:,1].max()
        
        # Latent grid
        self.xlatent = np.linspace(self.xmin,self.xmax,num_latent)
        self.tlatent = np.linspace(self.tmin,self.tmax,num_latent)
        self.input_latent = merge(self.xlatent,self.tlatent)
    
        # Sample grid
        self.xsamples = np.linspace(self.xmin,self.xmax,num_samples)
        self.tsamples = np.linspace(self.tmin,self.tmax,num_samples)
        self.input_samples = merge(self.xsamples,self.tsamples)
 
        # Kernels
        self.initial_kernel = gpflow.kernels.SquaredExponential(variance=1, lengthscales=1)
        self.pigp_kernel = gpflow.kernels.SquaredExponential(variance=1, lengthscales=1)

        # RHS
        self.function_rhs = rhs
        self.f_rhs_samples = unfold(rhs,self.xsamples)

        # GPs
        self.initial_gp = gpflow.models.GPR(data=(self.input, self.output), kernel=self.initial_kernel)

        self.ARD(self.initial_gp)
        self.output_latent = tf.Variable(self.initial_gp.predict_f(self.input_latent)[0].numpy())
        
        self.pigp = gpflow.models.GPR(data=(self.input_latent, self.output_latent), kernel=self.pigp_kernel)


    def train(self,nepochs):
        tf.print(f"Initial loss : {self.loss()}")
        opt = tf.keras.optimizers.Adam()
        for epoch in range(nepochs):
            opt.minimize(self.loss,self.output_latent)
            self.pigp_hyperpaprameter_optimize()
            if epoch % 10 == 0:
                tf.print(f"Epoch {epoch}: Residual (train) {self.loss()}") 


    def loss(self):
        self.update_pigp(self.output_latent)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.input_samples) # Change variable name 
            mean,_ = self.pigp.predict_f(self.input_samples)
            gradient = tape.gradient(mean,self.input_samples)

        # Derivatives of interest
        laplacian = tape.gradient(gradient, self.input_samples)[:,0]        
        time_deriv = gradient[:,1]

        # # Update loss
        loss_term = tf.math.reduce_mean(tf.square(self.f_rhs_samples + laplacian - time_deriv))
        return loss_term
