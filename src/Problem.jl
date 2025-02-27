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
abstract type AbstractProblem end

abstract type TimeVariation end
struct TimeStatic <: TimeVariation end
@kwdef struct TimeHarmonic <: TimeVariation
    ω::Real # Frequency in rad/s
end
# struct TimeTransient <: TimeVariation end

## Two-dimensional problem
abstract type Symmetry2D end
struct Axi2D <: Symmetry2D end
struct Planar2D <: Symmetry2D end

@kwdef struct Problem2D{T} <: AbstractProblem
    symmetry::Symmetry2D
    time::TimeVariation
    fe_order::Integer
    qr_order::Integer

    materials::Dict
    sources::Dict
    boundaries::Dict{String,<:AbstractBC}
end

## Three-dimensional problem
abstract type FEFormulation end
struct FormulationA <: FEFormulation end
#struct FormulationAϕ <: FEFormulation end
#struct FormulationTΩ <: FEFormulation end

@kwdef struct Problem3D <: AbstractProblem
    formulation::FEFormulation
    time::TimeVariation
    datatype::Type
    fe_order::Integer
    qr_order::Integer

    materials::Dict
    sources::Dict
    boundaries::Dict
end

# Cell parameters
struct CellParams{T}
    J0::Vector{T}    # Source current density [A/m^2]
    σ::Vector{T}     # Conductivity [S/m]
    ν::Vector{T}     # Reluctivity [m/H]
    # TODO allow for anisotropic permeability/reluctivity
end