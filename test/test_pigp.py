from pigp import *
import numpy as np

p = 3
n = 100
X = np.random.rand(n,p)
Y = np.random.rand(n,1)
data = (X,Y)
sampler = Sobol
nlatent = 10
nsampling = 100
f = lambda x : np.sin(x)
pigp = PIGP(data,sampler,nlatent,nsampling,f)