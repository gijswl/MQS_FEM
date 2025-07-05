using LinearAlgebra
using SparseArrays

abstract type AbstractCircuitElement end

struct CircuitImpedance <: AbstractCircuitElement
    Z::Complex
    pin1::Int
    pin2::Int
end

struct CircuitCurrentSource <: AbstractCircuitElement
    I # <: Real or Complex
    pin1::Int
    pin2::Int
end

struct CircuitVoltageSource <: AbstractCircuitElement
    V # <: Real or Complex
    pin1::Int
    pin2::Int
end

ordnum(::CircuitVoltageSource) = 1
ordnum(::CircuitImpedance) = 2
ordnum(::CircuitCurrentSource) = 3

function Base.isless(a::AbstractCircuitElement, b::AbstractCircuitElement)
    return ordnum(a) < ordnum(b)
end

mutable struct Circuit
    elements::Vector{AbstractCircuitElement}
    nodes::Vector{Int}
    tree::Vector{AbstractCircuitElement}
    cotree::Vector{AbstractCircuitElement}
    branches::Vector{Int}
    links::Vector{Int}
    vsrc::Vector{Int}
    isrc::Vector{Int}
    closed::Bool

    function Circuit()
        new(AbstractCircuitElement[], Int[], AbstractCircuitElement[], AbstractCircuitElement[], Int[], Int[], Int[], Int[], false)
    end
end

function add_circuit_R!(c::Circuit, R::Real, pin1::Int, pin2::Int)
    @assert !c.closed "Components cannot be added to a closed circuit."
    add_circuit_Z!(c, R + 0im, pin1, pin2)
end

function add_circuit_G!(c::Circuit, G::Real, pin1::Int, pin2::Int)
    @assert !c.closed "Components cannot be added to a closed circuit."
    add_circuit_Z!(c, 1 / G + 0im, pin1, pin2)
end

function add_circuit_Z!(c::Circuit, Z::Complex, pin1::Int, pin2::Int)
    @assert !c.closed "Components cannot be added to a closed circuit."
    push!(c.elements, CircuitImpedance(Z, pin1, pin2))
end

function add_circuit_Y!(c::Circuit, Y::Complex, pin1::Int, pin2::Int)
    @assert !c.closed "Components cannot be added to a closed circuit."
    add_circuit_Z!(c, 1 / Y, pin1, pin2)
end

function add_circuit_I!(c::Circuit, I::T, pin1::Int, pin2::Int) where {T<:Union{Real,Complex}}
    @assert !c.closed "Components cannot be added to a closed circuit."
    push!(c.elements, CircuitCurrentSource(I, pin1, pin2))
end

function add_circuit_V!(c::Circuit, V::T, pin1::Int, pin2::Int) where {T<:Union{Real,Complex}}
    @assert !c.closed "Components cannot be added to a closed circuit."
    push!(c.elements, CircuitVoltageSource(V, pin1, pin2))
end

function close!(c::Circuit)
    @assert !c.closed "Circuit is already closed."

    process_nodes!(c)
    process_tree!(c)

    c.closed = true
end

function process_nodes!(c::Circuit)
    nodes = c.nodes
    for elem ∈ c.elements
        !(elem.pin1 ∈ nodes) && push!(nodes, elem.pin1)
        !(elem.pin2 ∈ nodes) && push!(nodes, elem.pin2)
    end

    sort!(nodes)
end

function process_tree!(c::Circuit)
    elements = sort(c.elements)
    conn_nodes = Set{Int}()

    for e ∈ elements
        connected = (e.pin1 ∈ conn_nodes && e.pin2 ∈ conn_nodes)
        e_type = typeof(e)
        is_current_source = (e_type <: CircuitCurrentSource)

        if !connected && !is_current_source
            push!(conn_nodes, e.pin1, e.pin2)
            push!(c.tree, e)

            if e_type <: CircuitImpedance
                push!(c.branches, length(c.tree))
            elseif e_type <: CircuitVoltageSource
                push!(c.vsrc, length(c.tree))
            end
        else
            push!(c.cotree, e)

            if e_type <: CircuitImpedance
                push!(c.links, length(c.cotree))
            elseif e_type <: CircuitCurrentSource
                push!(c.isrc, length(c.cotree))
            end
        end
    end

    reverse!(c.cotree)
    idx = reverse(1:length(c.cotree))
    c.links = idx[c.links]
    c.isrc = idx[c.isrc]
end

function find_cycle(c::Circuit, pin1::Int, pin2::Int)
    node_to_index = Dict([node => idx for (idx, node) ∈ enumerate(c.nodes)])

    connected = [Set{Int}() for _ ∈ 1:length(c.nodes)]
    for (idx, elem) ∈ enumerate(c.tree)
        idx1 = node_to_index[elem.pin1]
        idx2 = node_to_index[elem.pin2]
        push!(connected[idx1], idx)
        push!(connected[idx2], idx)
    end


    r = pin1

    stack = [r]
    visited = Set(r)
    gnodes = Set(c.nodes)
    pred = Dict(r => r)
    edge = Dict(r => -1)
    while !isempty(stack)
        z = pop!(stack)
        for edge_nr ∈ connected[node_to_index[z]]
            element = c.tree[edge_nr]
            dest = element.pin1 == z ? element.pin2 : element.pin1
            if !(dest ∈ visited)
                push!(stack, dest)

                pred[dest] = z
                edge[dest] = edge_nr
            end
        end
        push!(visited, z)
    end
    setdiff!(gnodes, visited)
    if !isempty(gnodes)
        error("Graph not structured properly")
    end

    node = pin2
    cycle = Int[]
    while true
        prev = pred[node]

        e = edge[node]
        e == -1 && break
        element = c.tree[e]
        if (node == element.pin1 && prev == element.pin2)
            orientation = 1
        else
            orientation = -1
        end

        push!(cycle, orientation * e)
        node = prev
    end

    return cycle
end

function build_matrix(c::Circuit)
    @assert c.closed "Circuit must be closed before building the circuit matrix."

    Nvsrc = length(c.vsrc)
    Nbranch = length(c.branches)
    Nisrc = length(c.isrc)
    Nlink = length(c.links)

    vsrc_idx = 1:Nvsrc
    branch_idx = Nvsrc .+ (1:Nbranch)
    isrc_idx = 1:Nisrc
    link_idx = Nisrc .+ c.links[(1:Nlink)] # TODO correct the ordering of links vs the cotree... this is not sufficient

    vsrc = ComplexF64[c.tree[src].V for src ∈ c.vsrc]
    isrc = ComplexF64[c.cotree[src].I for src ∈ c.isrc]
    Slink = ComplexF64[-c.cotree[link].Z for link ∈ c.links]
    Sbranch = ComplexF64[1 / c.tree[branch].Z for branch ∈ c.branches]

    Ntree = length(c.tree)
    Ncotree = length(c.cotree)
    B = zeros(Ncotree, Ntree)
    for (i, link) ∈ enumerate(c.cotree)
        cycle = find_cycle(c, link.pin1, link.pin2)
        for (j, branch) ∈ enumerate(c.tree)
            if (j ∈ cycle)
                B[i, j] = 1
            elseif (-j ∈ cycle)
                B[i, j] = -1
            end
        end
    end

    println(B)

    BLT = B[link_idx, branch_idx]
    BLv = B[link_idx, vsrc_idx]
    BiT = B[isrc_idx, branch_idx]

    S = [Diagonal(Slink) -BLT; -transpose(BLT) Diagonal(Sbranch)]
    S = sparse(S)
    W = [BLv * vsrc; -transpose(BiT) * isrc]

    return S, W
end

c = Circuit()
add_circuit_V!(c, 230 * exp(0im * 2π / 3), 1, 0)
add_circuit_V!(c, 230 * exp(+1im * 2π / 3), 2, 0)
add_circuit_V!(c, 230 * exp(-1im * 2π / 3), 3, 0)
add_circuit_R!(c, 1e3, 1, 4)
add_circuit_R!(c, 1e3, 2, 4)
add_circuit_R!(c, 1e3, 3, 4)
add_circuit_Z!(c, 2π * 50im * 100e-6, 1, 2)
close!(c)

S, W = build_matrix(c)
u = S \ W