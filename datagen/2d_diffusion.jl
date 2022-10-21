using OrdinaryDiffEq
using Plots
using HDF5
using FFTW
gr()

# Two dimension
xmin = 0.0
xmax = 2π
ymin = 0.0
ymax = 2π
tmin = 0.0
tmax = 1.0
N = 100
dx = 2π/N
dy = 2π/N

x = xmin:dx:xmax
y = ymin:dy:ymax

function gaussian(x,y,loc)
    nx = length(x)
    ny = length(y)
    A = zeros(nx,ny)
    σ = 1
    for i=1:nx
        for j=1:ny
            A[i,j] = exp(-(((x[i]-loc)^2)/σ^2 + ((y[j]-loc)^2)/σ^2))
        end 
    end 
    A
end 

u0 = gaussian(x,y,π)
source = rfft(gaussian(x,y,π/2))
# heatmap(u0)
# heatmap(irfft(source,N))

function k!(du,u,p,t)
    nx,ny = size(u)
    for i=1:nx
        for j=1:ny
            du[i,j] =  -i^2*u[i,j] -j^2*u[i,j] + source[i,j]
        end 
    end 
end 

# Solve problem in Fourier space
û = rfft(u0)
tspan = (tmin,tmax)
prob = ODEProblem(k!,û,tspan,[])
sol = solve(prob,Tsit5())


# Convert problem back to real space.
solution = Array{eltype(sol[1][1].re),3}(undef,N,N,length(sol))
for (i,elem) in enumerate(sol.u)
    solution[:,:,i] = irfft(sol[i],N) 
end 

# Write the file to  a HDF5 file
filename= "data/diffusion2d"
file = h5open(filename,"w")
file["readme"] = "Output is a three dimensional array. Third dimension indexes the time. The other two are space dimensions."
file["data"] = solution
close(file)