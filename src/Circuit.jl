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

struct CircuitCoilStranded <: AbstractCircuitElement
    domain::String
    R::Real # Series resistance
    pin1::Int
    pin2::Int
end

struct CircuitCoilSolid <: AbstractCircuitElement
    domain::String
    R::Real # Parallel resistance
    pin1::Int
    pin2::Int
end

ordnum(::CircuitVoltageSource) = 1
ordnum(::CircuitCoilSolid) = 2
ordnum(::CircuitImpedance) = 3
ordnum(::CircuitCoilStranded) = 4
ordnum(::CircuitCurrentSource) = 5

function Base.isless(a::AbstractCircuitElement, b::AbstractCircuitElement)
    return ordnum(a) < ordnum(b)
end

mutable struct Circuit
    elements::Vector{AbstractCircuitElement}
    nodes::Vector{Int}
    tree::Vector{AbstractCircuitElement}
    cotree::Vector{AbstractCircuitElement}

    S::Matrix
    W::Vector
    coupling::Dict

    closed::Bool

    function Circuit()
        new(AbstractCircuitElement[], Int[], AbstractCircuitElement[], AbstractCircuitElement[], zeros(2, 2), ComplexF64[], Dict(), false)
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

function add_circuit_str!(c::Circuit, domain::String, R::Real, pin1::Int, pin2::Int)
    @assert !c.closed "Components cannot be added to a closed circuit."
    push!(c.elements, CircuitCoilStranded(domain, R, pin1, pin2))
end

function add_circuit_sol!(c::Circuit, domain::String, R::Real, pin1::Int, pin2::Int)
    @assert !c.closed "Components cannot be added to a closed circuit."
    push!(c.elements, CircuitCoilSolid(domain, R, pin1, pin2))
end

function close!(c::Circuit)
    @assert !c.closed "Circuit is already closed."

    process_nodes!(c)
    process_tree!(c)
    c.closed = true

    S, W, coupling = build_matrix(c)
    c.S = S
    c.W = W
    c.coupling = coupling
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
        else
            push!(c.cotree, e)
        end
    end
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

    tree_vsrc = filter(e -> typeof(e) <: CircuitVoltageSource, c.tree)
    tree_sol = filter(e -> typeof(e) <: CircuitCoilSolid, c.tree)
    tree_bra = filter(e -> typeof(e) <: CircuitImpedance, c.tree)
    tree_str = filter(e -> typeof(e) <: CircuitCoilStranded, c.tree)
    cotree_str = filter(e -> typeof(e) <: CircuitCoilStranded, c.cotree)
    cotree_lin = filter(e -> typeof(e) <: CircuitImpedance, c.cotree)
    cotree_sol = filter(e -> typeof(e) <: CircuitCoilSolid, c.cotree)
    cotree_isrc = filter(e -> typeof(e) <: CircuitCurrentSource, c.cotree)

    Nvsrc = length(tree_vsrc)
    Ntree_sol = length(tree_sol)
    Ntree_bra = length(tree_bra)
    Ntree_str = length(tree_str)
    Ncotree_sol = length(cotree_sol)
    Ncotree_lin = length(cotree_lin)
    Ncotree_str = length(cotree_str)
    Nisrc = length(cotree_isrc)

    idx_vsrc = 1:Nvsrc
    idx_tree_sol = Nvsrc .+ (1:Ntree_sol)
    idx_tree_bra = Nvsrc + Ntree_sol .+ (1:Ntree_bra)
    idx_tree_str = Nvsrc + Ntree_sol + Ntree_bra .+ (1:Ntree_str)
    idx_cotree_sol = (1:Ncotree_sol)
    idx_cotree_lin = Ncotree_sol .+ (1:Ncotree_lin)
    idx_cotree_str = Ncotree_sol + Ncotree_lin .+ (1:Ncotree_str)
    idx_isrc = Ncotree_sol + Ncotree_lin + Ncotree_str .+ (1:Nisrc)

    # Sources
    vsrc = ComplexF64[src.V for src ∈ tree_vsrc]
    isrc = ComplexF64[-src.I for src ∈ cotree_isrc]

    # Diagonal impedances
    Rstr = ComplexF64[link.R for link ∈ cotree_str]
    Rstra = ComplexF64[branch.R for branch ∈ tree_str]
    Gsol = ComplexF64[1 / branch.R for branch ∈ tree_sol]
    Gsola = ComplexF64[1 / link.R for link ∈ cotree_sol]
    Slink = ComplexF64[link.Z for link ∈ cotree_lin]
    Sbranch = ComplexF64[1 / branch.Z for branch ∈ tree_bra]

    B = zeros(length(c.cotree), length(c.tree))
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

    BLT = B[idx_cotree_lin, idx_tree_bra]
    BLsol = B[idx_cotree_lin, idx_tree_sol]
    BstrT = B[idx_cotree_str, idx_tree_bra]
    Bstrsol = B[idx_cotree_str, idx_tree_sol]
    DTL = -transpose(BLT)
    DsolL = -transpose(BLsol)
    DTstr = -transpose(BstrT)
    Dsolstr = -transpose(Bstrsol)

    BLv = B[idx_cotree_lin, idx_vsrc]
    Bstrv = B[idx_cotree_str, idx_vsrc]
    Bsolv = B[idx_cotree_sol, idx_vsrc]
    BiT = B[idx_isrc, idx_tree_bra]
    Bisol = B[idx_isrc, idx_tree_sol]
    Bistr = B[idx_isrc, idx_tree_str]
    DTi = -transpose(BiT)
    Dsoli = -transpose(Bisol)
    Dstri = -transpose(Bistr)

    Bstrstr = B[idx_cotree_str, idx_tree_str]
    Dstrstr = -transpose(Bstrstr)
    if (length(Bstrstr) != 0)
        Rstr = Diagonal(Rstr) - Bstrstr * Rstra * Dstrstr
        vstr = Bstrv * vsrc - Bstrstr * Rstra * Dstri * isrc
    else
        Rstr = Diagonal(Rstr)
        vstr = Bstrv * vsrc
    end
    Bsolsol = B[idx_cotree_sol, idx_tree_sol]
    Dsolsol = -transpose(Bsolsol)
    if (length(Bsolsol) != 0)
        Gsol = Diagonal(Gsol) - Dsolsol * Gsola * Bsolsol
        isol = -Dsoli * isrc + Dsolsol * Gsola * Bsolv * vsrc
    else
        Gsol = Diagonal(Gsol)
        isol = -Dsoli * isrc
    end

    Slink2 = length(Slink) == 0 ? 0 : size(Slink, 2)
    Rstr2 = length(Rstr) == 0 ? 0 : size(Rstr, 2)
    Sbranch2 = length(Sbranch) == 0 ? 0 : size(Sbranch, 2)
    Gsol2 = length(Gsol) == 0 ? 0 : size(Gsol, 2)

    ZstrL = zeros(size(Rstr, 1), Slink2) # Not correct size(Slink, 2)
    ZLstr = zeros(size(Slink, 1), Rstr2)
    ZsolT = zeros(size(Gsol, 1), Sbranch2)
    ZTsol = zeros(size(Sbranch, 1), Gsol2)

    S = [
        -Rstr ZstrL -Bstrsol -BstrT;
        ZLstr Diagonal(-Slink) -BLsol -BLT;
        Dsolstr DsolL Gsol ZsolT;
        DTstr DTL ZTsol Diagonal(Sbranch)
    ]
    W = [
        vstr; BLv * vsrc; isol; -DTi * isrc
    ]

    coupling = Dict(
        "Bsolv" => Bsolv,
        "Dstri" => Dstri,
        "Bsolsol" => Bsolsol,
        "Bstrstr" => Bstrstr,
        "Dstrstr" => Dstrstr,
        "Dsolsol" => Dsolsol,
        "Vv" => vsrc,
        "Ii" => isrc,
        "segmentation" => [size(Rstr, 1), size(Slink, 1), size(Gsol, 1), size(Sbranch, 1)]
    )

    return S, W, coupling
end

# c = Circuit()
# # add_circuit_V!(c, 230 * exp(0im * 2π / 3), 1, 0)
# # add_circuit_V!(c, 230 * exp(+1im * 2π / 3), 2, 0)
# # add_circuit_V!(c, 230 * exp(-1im * 2π / 3), 3, 0)
# # add_circuit_str!(c, "Conductor1", 1, 1, 4)
# # add_circuit_str!(c, "Conductor2", 1, 2, 4)
# # add_circuit_str!(c, "Conductor3", 1, 3, 4)
# add_circuit_V!(c, 1, 1, 0)
# add_circuit_sol!(c, "Conductor", 1, 1, 0)
# close!(c)

# S, W, coupling = build_matrix(c)
# #u = S \ W