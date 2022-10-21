using HDF5
using Plots
gr()
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets

begin
    # Parameters, variables, and derivatives
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 1D PDE and boundary conditions
    ## Governing equation
    eq  = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) - sin(8*x) - cos(6*y)
    
    bcs = [u(0, x, y) ~ sin(x*y),
            u(t, 0, y) ~ 0,
            u(t, 2π, y) ~ 0,
            u(t, x, 0) ~ 0,
            u(t, x, 2π) ~ 0
            ]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 5.0),
            x ∈ Interval(0.0, 2π),
            y ∈ Interval(0.0, 2π)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    # Method of lines discretization
    dx = 0.05
    dy = 0.05
    order = 5
    discretization = MOLFiniteDifference([x => dx,y => dy], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.2)
    solution = sol.u[u(t,x,y)]

    anim = @animate for i=1:size(solution,1)
        heatmap(solution[i,:,:],label="Iterate $(i)",cmap=(-2,2))
    end 
    gif(anim,"figures/1d_Diffusion.gif",fps=20)

    file = h5open("data/1d_diffusion","w")
    file["readme"] = "Index u along the first index to get the u value over the domain with respect to time."
    file["x"] = Vector(0.0:dx:2π)
    file["y"] = Vector(0.0:dy:2π)
    file["t"] = sol.t
    file["u"] = solution
    close(file)
 end 
