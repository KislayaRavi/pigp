using DiffEqOperators
using Plots
using HDF5

dx = 0.01
axis = collect(-1:dx:1);
umesh = [(i, j) for i in axis, j in axis];

sines((x,y)) = sin(pi*x)*sin(pi*y)
u = sines.(umesh)

f1 = heatmap(u,xlabel="x",ylabel="y",title="Source Term")

N = size(u,1)
Qx, Qy = Dirichlet0BC(Float64, size(u));
Dxx = CenteredDifference{1}(2,2,dx,N);
Dyy = CenteredDifference{2}(2,2,dx,N);
A = Dxx + Dyy

