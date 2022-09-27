using OrdinaryDiffEq
using Plots
using HDF5
using FFTW
gr()

# One dimension spectral method 
begin
    xmin = 0.0
    xmax = 2π
    tmin = 0.0
    tmax = 1.0
    N = 100
    dx = 2π/N

    f(x) = sin.(x)
    x = xmin:dx:xmax 
    t = tmin:0.2:tmax
    source = rfft(f(x))

    function k!(du,u,p,t)
        p = x[1]
        for (i,elem) in enumerate(u)
            du[i] = -i^2*elem + source[i]
        end 
    end 

    # Change the function to Fourier space
    u0 = f(x)
    û = rfft(u0)
    tspan = (tmin,tmax)
    prob = ODEProblem(k!,û,tspan,[x])
    sol = solve(prob,Tsit5(),saveat=0.2)

    anim = @animate for i=1:length(t)
        u = irfft(sol.u[i],N)
        plot(u,title="Diffusion time step = $(i)",ylim=[-1,1])
    end 
    gif(anim,"diffusion.gif",fps=20)
end 


using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets

begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t, x)) ~ Dxx(u(t, x)) 
    bcs = [u(0, x) ~ sin(x),
            u(t, 0) ~ 0,
            u(t, 2π) ~ 0]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
            x ∈ Interval(0.0, 2π)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.2)
    solution = sol.u[u(t,x)]

    anim = @animate for i=1:size(solution,1)
        plot(solution[i,:],label="Iterate = $(i)",ylim=[-1,1])
    end 
    gif(anim,"finite_differences.gif",fps=20)
end 
