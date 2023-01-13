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
    #------------------------Constructor------------------------------#
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
        self.initial_gp, self.pigp = self.create_gps()
        
    #----------------------------------------------------------------#
    #------------------------Internal Functions----------------------#
    #----------------------------------------------------------------#
    def create_grids(self):
        ##########################################################################################################
        # TODO: Add function to ensure that there are boundary nodes
        # latent_grid = self.create_grid(self.lbounds,self.ubounds,self.nlatent_points)
        grid_x = tf.convert_to_tensor(np.atleast_2d(np.linspace(self.lbounds,self.ubounds,self.nlatent_points)))
        grid_y = np.zeros(grid_x.shape)
        latent_grid = dict()
        latent_grid["x"] = tf.convert_to_tensor(grid_x)
        latent_grid["y"] = tf.convert_to_tensor(grid_y)
        ############################################################################################################
        sampling_grid = self.create_grid(self.lbounds,self.ubounds,self.nsampling_points)
        return (latent_grid,sampling_grid)
    
    def create_grid(self,lbounds,ubounds,n):
        # Create grid
        sampler = self.sampler(lbounds,ubounds,n)
        grid_x = sampler.sample()
        grid_y = np.zeros(grid_x.shape)        
        grid = dict()
        grid["x"] = tf.convert_to_tensor(grid_x)
        grid["y"] = tf.convert_to_tensor(grid_y)
        return grid

    def create_gps(self):
        igp = self.initial_gp()
        pigp = self.pigp(igp)
        return igp,pigp
        
    def initial_gp(self):
        kernel = gpflow.kernels.SquaredExponential(variance=1, lengthscales=1)
        X = tf.convert_to_tensor(self.X)
        Y = tf.convert_to_tensor(self.Y)
        igp = gpflow.models.GPR(data=(X,Y),kernel=kernel)
        self.latent_train(igp)
        return igp

    def pigp(self,igp):
        X = self.latent_grid["x"]
        mean,_ = igp.predict_f(X)
        self.latent_grid["y"] = mean
        kernel = gpflow.kernels.SquaredExponential(variance=1, lengthscales=1)
        pigp = gpflow.models.GPR(data=(X,mean),kernel=kernel)
        return pigp

    def latent_train(self,gp):
        gp.likelihood.variance.assign(1e-5)
        gpflow.utilities.set_trainable(gp.likelihood.variance, False)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(gp.training_loss, 
                                gp.trainable_variables, 
                                options=dict(maxiter=100))  

    def pigp_hyperparameter_optimize(self):
        self.pigp.likelihood.variance.assign(1e-5)
        gpflow.utilities.set_trainable(self.pigp.likelihood.variance, False)
        # gpflow.utilities.set_trainable(self.pigp.data[1], True)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(self.loss, 
                                self.pigp.trainable_variables, 
                                options=dict(maxiter=1))

    #----------------------------------------------------------------#
    #------------------------Interfaces------------------------------#
    #----------------------------------------------------------------#

    @abstractmethod
    def loss():
        pass 

    def train(self, nepochs, freq):
        optimiser = tf.keras.optimizers.Adam()
        tf.print(f"Initial loss {self.loss()}")
        self.latent_grid["y"] = tf.Variable(self.latent_grid["y"])
        for epoch_id in range(1, nepochs+1):
            optimiser.minimize(self.loss, self.latent_grid["y"])
            self.pigp_hyperparameter_optimize()
            #####################################################################
            # TODO: Implement it in form of a function
            self.latent_grid["y"][0].assign(tf.Variable([0.], dtype=np.float64))
            self.latent_grid["y"][-1].assign(tf.Variable([0.], dtype=np.float64))
            #######################################################################
            if epoch_id % freq == 0:
                tf.print(f"Epoch {epoch_id}: Residual (train) {self.loss()}")