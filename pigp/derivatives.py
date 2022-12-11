from .pigp import PIGP
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow

"""First derivative Operator for one dimension.
"""
class D(PIGP):
    def __init__(self, data, sampler, nlatent, nsampling,f):
        super().__init__(data, sampler, nlatent, nsampling,f)

    def loss(self):
        self.pigp.data = (self.latent_grid["x"],self.latent_grid["y"])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.sampling_grid["x"])
            mean, _ = self.pigp.predict_f(self.sampling_grid["x"])

        grad_mean = tape.gradient(mean,self.sampling_grid["x"])
        loss_term = tf.math.reduce_mean(tf.square(self.f(self.sampling_grid["x"]) - grad_mean))
        return loss_term


"""Second derivative Operator for one dimension.
"""
class D2(PIGP):
    def __init__(self, data, sampler, nlatent, nsampling,f):
        super().__init__(data, sampler, nlatent, nsampling,f)

    def loss(self):
        pass