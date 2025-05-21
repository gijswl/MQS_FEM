using Ferrite
using SparseArrays

include("Problem.jl")
include("CircuitCoupling.jl")
include("Solver2D.jl")
include("PostProcessing2D.jl")

include("FerriteAdditions.jl")

μ0 = 4π * 1e-7