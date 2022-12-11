# %%
from pigp import *
import numpy as np

# %%
p = 1
n = 100
X = np.random.rand(n,1)
Y = np.random.rand(n,1)
# %%
data = (X,Y)
sampler = Sobol
# %%
nlatent = 16
nsampling = 128
f = lambda x : np.sin(x)

# %%
pigp = PIGP(data,sampler,nlatent,nsampling,f)
