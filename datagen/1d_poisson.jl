using DiffEqOperators
using Plots
using HDF5

begin
    f(x) = -sin.(4 .* x) .* cos.(6 .* x)
    x = 0:1e-2:4π
    y = f(x)
    fig1 = plot(x,y,title="Forcing function - 1D Poisson",label="y",xlabel="x",ylabel="y")

    nknots = length(x)
    dx = x[2]-x[1]
    ord_deriv = 2
    ord_approx = 2
    const Δ = CenteredDifference(ord_deriv,ord_approx,dx,nknots)
    const bc= Dirichlet0BC(Float64)

    # We are solving the BVP
    #Δu = y
    # u = inv(Δ)y

    u = inv(Array(Δ*bc)[1])*y
    fig2 = plot(x,u,title="Solution - 1D Poisson",label="u",xlabel="x",ylabel="u")
    plot(fig1,fig2,size=(1000,700))
    savefig("datagen/1d_poisson.png")
end 

begin
    filename = "data/1d_poisson.hdf5"
    file = h5open(filename,"w")
    file["label"] = "1D Poisson equation Δu(x) = f(x)"
    file["x"] = Array(x)
    file["f"] = "-sin.(4 .* x) .* cos.(6 .* x)"
    file["u"] = u
    close(file)
end 