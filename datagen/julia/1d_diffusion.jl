using HDF5
using Plots
gr()
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets

begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    ## Governing equation
    eq  = Dt(u(t, x)) ~ Dxx(u(t, x)) - sin(8*x) - cos(6*x)
    
    bcs = [u(0, x) ~ sin(x),
            u(t, 0) ~ 0,
            u(t, 2π) ~ 0]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 5.0),
            x ∈ Interval(0.0, 2π)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 0.05
    order = 5
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.2)
    solution = sol.u[u(t,x)]

    anim = @animate for i=1:size(solution,1)
        plot(solution[i,:],label="Iterate = $(i)",ylim=[-1,1])
    end 
    gif(anim,"figures/1d_Diffusion.gif",fps=20)

    file = h5open("data/1d_diffusion","w")
    file["readme"] = "u is matrix with timesteps along the rows and gridsteps along the columns"
    file["x"] = Vector(0.0:dx:2π)
    file["t"] = sol.t
    file["u"] = solution
    close(file)
 end 
