abstract type AbstractFECoupling end

mutable struct ConductorSolid <: AbstractFECoupling
    domain::String
    symm_factor::Real
    area::Real
    G::Real # Conductance
    natural::Bool # natural = true if the solid conductor is directly driven by a voltage source (no extra unknown required)
end

mutable struct ConductorStranded <: AbstractFECoupling
    domain::String
    symm_factor::Real
    area::Real
    N::Int # Number of strands
    R::Real # Resistance
    natural::Bool # natural = true if the stranded conductor is directly driven by a current source (no extra unknown required)
end

mutable struct CircuitHandler
    const data_type::DataType
    const dh::Ferrite.AbstractDofHandler

    cond_sol::Vector{ConductorSolid}
    cond_str::Vector{ConductorStranded}
    cond_type::Dict{String,AbstractConductorType}
    fecoupling::Dict{String,AbstractFECoupling}

    circuit::Circuit

    closed::Bool
    ndofs::Int
end

function CircuitHandler(dh::Ferrite.AbstractDofHandler, val_type::DataType)
    @assert Ferrite.isclosed(dh)

    circuit = Circuit()
    CircuitHandler(val_type, dh, ConductorSolid[], ConductorStranded[], Dict{String,AbstractConductorType}(), Dict{String,AbstractFECoupling}(), circuit, false, -1)
end

isclosed(ch::CircuitHandler) = ch.closed

function close!(ch::CircuitHandler)
    for conductor ∈ ch.cond_str
        conductor.area = get_domain_area(ch.dh, conductor.domain) / conductor.symm_factor
        ch.cond_type[conductor.domain] = ConductorTypeStranded()
        ch.fecoupling[conductor.domain] = conductor
    end
    for conductor ∈ ch.cond_sol
        conductor.area = get_domain_area(ch.dh, conductor.domain) / conductor.symm_factor
        ch.cond_type[conductor.domain] = ConductorTypeSolid()
        ch.fecoupling[conductor.domain] = conductor
    end

    close!(ch.circuit)

    ch.closed = true
    ch.ndofs = size(ch.circuit.S, 1)
end

function Base.show(io::IO, ::MIME"text/plain", ch::CircuitHandler)
    println(io, "CircuitHandler:")
    print(io, "  Solid conductors:")
    for c in ch.cond_sol
        print(io, "\n    ", "Domain: ", c.domain)
    end

    print(io, "\n  Stranded conductors:")
    for c in ch.cond_str
        print(io, "\n    ", "Domain: ", c.domain, ", N: ", c.N)
    end

    print(io, "\n  Circuit elements:")
    for elem ∈ circuit_elements(ch)
        print(io, "\n    ", elem)
    end
end

function add_conductor_solid!(ch::CircuitHandler, domain::String)
    add_conductor_solid!(ch, domain, 1)
end

function add_conductor_solid!(ch::CircuitHandler, domain::String, symm_factor::Real)
    @assert (symm_factor >= 0) && (symm_factor <= 1) "Symmetry factor must be between 0 and 1"
    push!(ch.cond_sol, ConductorSolid(domain, symm_factor, -1, -1, true))
end

function add_conductor_stranded!(ch::CircuitHandler, domain::String, N::Integer)
    add_conductor_stranded!(ch, domain, N, 1)
end

function add_conductor_stranded!(ch::CircuitHandler, domain::String, N::Integer, symm_factor::Real)
    @assert (symm_factor >= 0) && (symm_factor <= 1) "Symmetry factor must be between 0 and 1"
    push!(ch.cond_str, ConductorStranded(domain, symm_factor, -1, N, -1, true))
end

ncond_sol(ch::CircuitHandler) = length(ch.cond_sol)
ncond_str(ch::CircuitHandler) = length(ch.cond_str)
nconductors(ch::CircuitHandler) = ncond_sol(ch) + ncond_str(ch)

function check_circuit(ch::CircuitHandler)
    v = []
    for elem ∈ ch.circuit_elements
        !(elem.pin1 ∈ v) && push!(v, elem.pin1)
        !(elem.pin2 ∈ v) && push!(v, elem.pin2)

        if (typeof(elem) <: CircuitCoil)
            @assert haskey(ch.cond_type, elem.domain) "Coil component on $(elem.domain) does not have a corresponding conductor defined"
        end
    end
    sort!(v)
    @assert all(diff(v) .== 1) "Electrical circuit nodes must be numbered consecutively"
    @assert v[1] == 0 "Electrical circuit must have a ground node numbered 0"

    return v
end

function get_ndofs(ch::CircuitHandler)
    @assert isclosed(ch)
    return ch.ndofs
end

function get_conductor_type(ch::CircuitHandler, domain::String)
    if (haskey(ch.cond_type, domain))
        return ch.cond_type[domain]
    end

    return ConductorTypeNone()
end