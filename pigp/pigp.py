from abc import abstractmethod
from mimetypes import init
from re import I
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt

tf.get_logger().setLevel("INFO")
np.random.seed(0)
tf.random.set_seed(10)


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
        self.pigp.data = (self.latent_X, latent_Y)
    
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
    
    def loss(self):
        self.update_pigp(self.latent_Y)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.samples_X)
            mean,_ = self.pigp.predict_f(self.samples_X)
            gradient = tape.gradient(mean,self.samples_X)
        laplacian = tape.gradient(gradient,self.samples_X)
        loss_term = tf.math.reduce_mean(tf.square(self.rhs_sample - laplacian))
        return loss_term
            