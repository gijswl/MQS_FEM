abstract type CircuitCoupling end

struct CurrentCoupling{T} <: CircuitCoupling
    domain::String
    area::Real
    symm_factor::Real
    current::T
end

struct VoltageCoupling{T} <: CircuitCoupling
    domain::String
    area::Real
    symm_factor::Real
    voltage::T
end

struct CircuitHandler{DH<:Ferrite.AbstractDofHandler}
    coupling::Vector{CircuitCoupling}
    dh::DH
    data_type::DataType
end

function CircuitHandler(dh::Ferrite.AbstractDofHandler, val_type::DataType)
    @assert Ferrite.isclosed(dh)

    CircuitHandler(CircuitCoupling[], dh, val_type)
end

function Base.show(io::IO, ::MIME"text/plain", ch::CircuitHandler)
    println(io, "CircuitHandler:")
    print(io, "  Circuit constraints:")
    for c in ch.coupling
        print(io, "\n    ", "Domain: ", c.domain, ", ")
        if (typeof(c) <: CurrentCoupling)
            print(io, "Current: ", c.current, " A")
        elseif (typeof(c) <: VoltageCoupling)
            print(io, "Voltage: ", c.current, " V")
        else
            error("Invalid CircuitCoupling subtype $(typeof(c))")
        end
    end
end

function add_current_coupling!(ch::CircuitHandler, dom_name::String, current, area::Real)
    push!(ch.coupling, CurrentCoupling{ch.data_type}(dom_name, area, 1, current))
end

function add_current_coupling!(ch::CircuitHandler, dom_name::String, current, area::Real, symm_factor::Real)
    push!(ch.coupling, CurrentCoupling{ch.data_type}(dom_name, area, symm_factor, current))
end

function add_voltage_coupling!(ch::CircuitHandler, dom_name::String, voltage, area::Real)
    push!(ch.coupling, VoltageCoupling{ch.data_type}(dom_name, area, 1, voltage))
end

function add_voltage_coupling!(ch::CircuitHandler, dom_name::String, voltage, area::Real, symm_factor::Real)
    push!(ch.coupling, VoltageCoupling{ch.data_type}(dom_name, area, symm_factor, voltage))
end

function ncouplings(ch::CircuitHandler)
    return length(ch.coupling)
end

function add_sparsity_circuit!(sp::SparsityPattern, dh::DofHandler, ch::CircuitHandler)
    for (i, coupling) ∈ enumerate(ch.coupling)
        coupling_idx = ndofs(dh) + i

        Ferrite.add_entry!(sp, coupling_idx, coupling_idx)

        cells = getcellset(dh.grid, coupling.domain)
        for cell ∈ cells
            dofs = celldofs(dh, cell)
            for dof ∈ dofs
                Ferrite.add_entry!(sp, coupling_idx, dof)
                Ferrite.add_entry!(sp, dof, coupling_idx)
            end
        end
    end
end

function apply_circuit_couplings!(problem::Problem, time::TimeHarmonic, params::Vector{CellParams}, K::SparseMatrixCSC, f::Vector, cv::CV, dh::DofHandler, ch::CircuitHandler) where {CV<:NamedTuple}
    for sdh ∈ dh.subdofhandlers
        cell_type = getcelltype(sdh)
        cv_ = get_cellvalues(cv, cell_type)

        apply_circuit_couplings!(problem, time, params, K, f, cv_, sdh, ch)
    end
end

function apply_circuit_couplings!(problem::Problem, time::TimeHarmonic, params::Vector{CellParams}, K::SparseMatrixCSC, f::Vector, cv::CellValues, sdh::SubDofHandler, ch::CircuitHandler)
    for (i, coupling) ∈ enumerate(ch.coupling)
        coupling_idx = ndofs(sdh.dh) + i

        if (typeof(coupling) <: CurrentCoupling)
            apply_current_coupling!(problem, time, params, K, f, cv, sdh, coupling, coupling_idx)
            #elseif (typeof(coupling) <: VoltageCoupling)
            #    apply_voltage_coupling!(K, f, cv, sdh, coupling, coupling_idx, params)
        else
            error("Coupling $(typeof(c)) not implemented")
        end
    end

    return K, f
end

function apply_current_coupling!(::Problem, time::TimeHarmonic, params::Vector{CellParams}, K::SparseMatrixCSC, f::Vector, cv::CellValues, sdh::SubDofHandler, coupling::CircuitCoupling, coupling_idx::Int)
    # Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(Complex{Float64}, n_basefuncs)
    KTe = zeros(Complex{Float64}, n_basefuncs)

    ω = time.ω

    domain_set = getcellset(sdh.dh.grid, coupling.domain)
    for cell ∈ CellIterator(sdh)
        cell_id = cellid(cell)
        if (cell_id ∉ domain_set)
            continue
        end
        reinit!(cv, cell)

        # Reset to 0
        fill!(Ke, 0)
        fill!(KTe, 0)

        # Retrieve physical parameters
        param = params[cell_id]
        x = getcoordinates(sdh.dh.grid, cell_id)

        # Loop over quadrature points
        for q_point in 1:getnquadpoints(cv)
            # Get the quadrature weight
            coord = spatial_coordinate(cv, q_point, x)
            dΩ = getdetJdV(cv, q_point)

            # Loop over test shape functions
            for i in 1:n_basefuncs
                v = shape_value(cv, q_point, i)

                Ke[i] += -1im * ω * param.σ * v * dΩ
                KTe[i] += -1 / (coupling.area * coupling.symm_factor) * v * dΩ
            end
        end

        K[coupling_idx, celldofs(cell)] .+= Ke
        K[celldofs(cell), coupling_idx] .+= KTe
    end

    K[coupling_idx, coupling_idx] = 1
    f[coupling_idx] = coupling.current * coupling.symm_factor

    return K, f
end