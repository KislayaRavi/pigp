using Plots
using OrdinaryDiffEq
using LinearAlgebra

# Parameters
xmin = 0
xmax = π
ymin = 0
ymax = π
nx = 100
ny = 100
x = xmin:(xmax-xmin)/nx:xmax
y = ymin:(ymax-ymin)/ny:ymax

# Forcing function
f(x,y) = sin.(x*y)

# Zeros
F = [[f(xi,yi) for xi in x] for yi in y]
F = reduce(hcat,F)
heatmap(x,y,F)
savefig("figures/poisson.png")


niters = 10
tol = 1e-4
r = 100
iter = 1

u = zero(F)
du = zero(F)
res = zero(F)
nx = length(x)
ny = length(y)
dx = x[2]-x[1]
dy = y[2]-y[1]
kx = 1/dx^2
ky = 1/dy^2

while (iter < niters)
    print("Iteration = $(iter) \n")
    # Update du
    for i=2:nx-1
        for j=2:ny-1
            du[i,j] = kx*(u[i+1,j]+u[i-1,j]-2*u[i,j]) + ky*(u[i,j+1]+u[i,j+1]-2*u[i,j]) - F[i,j]
        end 
    end 
    # @show du

    # Compute residual
    for i=2:nx-1
        for j=2:ny-1
            res[i,j] = du[i,j] - kx*(u[i+1,j]+u[i-1,j]-2*u[i,j]) - ky*(u[i,j+1]+u[i,j+1]-2*u[i,j]) + F[i,j]
        end 
    end
    # @show res
    r = norm(res)    

    iter += 1
    u .= du .- 0.1*res
    
    print("Residual = $(r) \n\n") 
end 

heatmap(x,y,du)