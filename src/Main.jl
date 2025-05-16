using Ferrite
using SparseArrays

include("FerriteAdditions.jl")

include("Problem.jl")
include("CircuitCoupling.jl")
include("Solver2D.jl")
include("PostProcessing2D.jl")

μ0 = 4π * 1e-7