# Boundary conditions
abstract type AbstractBC{T} end

struct BoundaryA{T} <: AbstractBC{T}
    A::T
end

struct BoundaryFlux{T} <: AbstractBC{T}
    B::T
end

BoundaryInfty(T) = BoundaryA(zero(T))


# Materials


# Sources


# Problem definition
abstract type TimeVariation end
struct TimeStatic <: TimeVariation end
@kwdef struct TimeHarmonic <: TimeVariation
    ω::Real # Frequency in rad/s
end
# struct TimeTransient <: TimeVariation end

# Two-dimensional problem symmetry
abstract type Symmetry2D end
struct Axi2D <: Symmetry2D end
struct Planar2D <: Symmetry2D
    depth::Real
end

@kwdef struct Problem{T}
    symmetry::Symmetry2D
    time::TimeVariation
    fe_order::Integer
    qr_order::Integer

    materials::Dict
    sources::Dict
    boundaries::Dict{String,<:AbstractBC}
end

# Cell parameters
abstract type ConductorType end
struct ConductorTypeNone <: ConductorType end
struct ConductorTypeSolid <: ConductorType end
struct ConductorTypeStranded <: ConductorType end

struct CellParams
    J0            # Source current density [A/m^2]
    σ             # Conductivity [S/m]
    ν::Tensor{2,2} # Reluctivity tensor [m/H]
    cond_type::ConductorType
end

function get_frequency(problem::Problem)
    if (typeof(problem.time) <: TimeHarmonic)
        return problem.time.ω
    else
        return 0
    end
end