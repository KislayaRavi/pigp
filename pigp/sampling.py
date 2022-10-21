import numpy as np
from abc import abstractmethod
from scipy.stats import qmc

class Sampling():
    def __init__(self,min,max,n):
        self.lbounds = min
        self.ubounds = max
        assert len(min) == len(max) , "Upper and lower bounds should have the same dimensions."
        self.npoints = n

    @abstractmethod
    def sample(self):
        pass 

class QMC(Sampling):
    def __init__(self,min,max,n):
        super().__init__(min,max,n)

    def sample(self):
        pass

class Sobol(QMC):
    def __init__(self,min,max,n):
        assert np.mod(n,2) == 0 , "Number of samples should be powers of 2."
        pseudo_n = int(np.log2(n)) - 1
        super().__init__(min,max,pseudo_n)

    def sample(self):
        dims = len(list(self.lbounds))
        sampler = qmc.Sobol(d=dims,scramble=False)
        samples = sampler.random_base2(m=self.npoints)
        samples = qmc.scale(samples,self.lbounds,self.ubounds)
        return samples 

class Halton(QMC):
    def __init__(self,min,max,n):
        assert np.mod(n,2) == 0, "Number of points must be divisible by 2."
        pseudo_n = n//2
        super().__init__(min,max,pseudo_n)

    def sample(self):
        dims = len(list(self.lbounds))
        sampler = qmc.Halton(d=dims,scramble=False)
        samples = sampler.random(n=self.npoints)
        samples = qmc.scale(samples,self.lbounds,self.ubounds)
        return samples 

class LatinHypercube(QMC):
    def __init__(self,min,max,n):
        assert np.mod(n,2) == 0, "Number of points must be divisible by 2."
        pseudo_n = n//2
        super().__init__(min,max,pseudo_n)

    def sample(self):
        dims = len(list(self.lbounds))
        sampler = qmc.LatinHypercube(d=dims)
        samples = sampler.random(n=self.npoints)
        samples = qmc.scale(samples,self.lbounds,self.ubounds)
        return samples 
