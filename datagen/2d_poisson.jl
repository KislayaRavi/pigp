using HDF5
using Plots
gr()
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets

# Parameters, variables, and derivatives
@parameters x, y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 1D PDE and boundary conditions
## Governing equation
eq  = Dyy(u(x, y)) + Dxx(u(x, y)) ~ - sin(8*x) - cos(6*y)

bcs = [ u(x, 0) ~ 0,
        u(x,6) ~ sin(x)
        u(0,y) ~ 0,
        u(6,y) ~ sin(y)
        ]

# Space and time domains
domains = [x ∈ Interval(0.0, 6.0),
        y ∈ Interval(0.0, 6.0)]

# Solve this equation using a boundary value problem framework.