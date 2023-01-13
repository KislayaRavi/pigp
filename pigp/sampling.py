import numpy as np
from abc import abstractmethod
from scipy.stats import qmc

class Sampling():
    def __init__(self,min,max):
        self.lower_bounds = np.array(min)
        self.upper_bounds = np.array(max)
        assert len(min) == len(max) , "Upper and lower bounds should have the same dimensions."
        self.epsilon = 0.001 * (self.upper_bounds - self.lower_bounds) # ensures that points are not on the boundary

    @abstractmethod
    def sample(self, n, dim, lower_bounds, upper_bounds):
        pass

    def sample_internal(self, n):
        dim = len(self.lower_bounds)
        samples = self.sample(n, dim, self.lower_bounds + self.epsilon, self.upper_bounds)
        return samples

    def sample_boundary(self, n):
        dim = len(self.lower_bounds)
        samples = None
        if dim == 1:
            return np.array([[self.lower_bounds[0]], [self.upper_bounds[0]]])
        elif dim == 2:
            length = self.upper_bounds - self.lower_bounds
            points = np.sort(self.sample(n, dim-1, [self.lower_bounds[0]], [self.lower_bounds[0] + 2*np.sum(length)]).ravel())
            samples = np.zeros((len(points), dim))
            idy0 = np.argmax(points > self.upper_bounds[0])
            idx1 = np.argmax(points > (self.upper_bounds[0] + length[1]))
            idy1 = np.argmax(points > (self.upper_bounds[0] + length[1] + length[0]))        
            samples[:idy0, 0], samples[:idy0, 1] = points[:idy0], self.lower_bounds[1]
            samples[idy0:idx1, 0], samples[idy0:idx1, 1] = self.upper_bounds[0], points[idy0:idx1] - self.upper_bounds[0]
            samples[idx1:idy1, 0], samples[idx1:idy1, 1] = points[idx1:idy1] - (self.upper_bounds[0] + length[1]), self.upper_bounds[1]
            samples[idy1:, 0], samples[idy1:, 1] = self.lower_bounds[0], points[idy1:] - (self.upper_bounds[0] + length[1] + length[0]) 
        elif dim == 3:
            num_boundaries = 6
            raise('Not prepared for 3 dimensional spatial domain')
        else:
            raise('Not prepared for more than three dimensional spatial domain')
        return samples


class Sobol(Sampling):
    def __init__(self,min,max):
        super().__init__(min,max)

    def sample(self, n, dim, lower_bounds, upper_bounds):
        assert np.mod(n,2) == 0 , "Number of samples should be powers of 2."
        pseudo_n = int(np.log2(n)) #- 1
        sampler = qmc.Sobol(d=dim,scramble=False)
        samples = sampler.random_base2(m=pseudo_n)
        samples = qmc.scale(samples, lower_bounds, upper_bounds)
        return np.array(samples) 

class Halton(Sampling):
    def __init__(self,min,max):
        super().__init__(min,max)

    def sample(self, n, dim, lower_bounds, upper_bounds):
        assert np.mod(n,2) == 0, "Number of points must be divisible by 2."
        pseudo_n = 2*(n//2)
        sampler = qmc.Halton(d=dim,scramble=False)
        samples = sampler.random(n=pseudo_n)
        samples = qmc.scale(samples, lower_bounds, upper_bounds)
        return samples 

class LatinHypercube(Sampling):
    def __init__(self,min,max):
        super().__init__(min,max)

    def sample(self, n, dim, lower_bounds, upper_bounds):
        assert np.mod(n,2) == 0, "Number of points must be divisible by 2."
        pseudo_n = 2*(n//2)
        sampler = qmc.LatinHypercube(d=dim)
        samples = sampler.random(n=pseudo_n)
        samples = qmc.scale(samples, lower_bounds, upper_bounds)
        return samples 

class UniformRandom(Sampling):
    def __init__(self,min,max):
        super().__init__(min,max)

    def sample(self, n, dim, lower_bounds, upper_bounds):
        return np.random.uniform(lower_bounds, upper_bounds, (n, dim))
