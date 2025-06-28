using Ferrite
using SparseArrays

include("Problem.jl")
include("CircuitCoupling.jl")
include("Solver.jl")
include("PostProcessing.jl")
include("Homogenization.jl")

include("FerriteAdditions.jl")

μ0 = 4π * 1e-7