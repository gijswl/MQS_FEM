abstract type FECoupling end

struct ConductorSolid <: FECoupling
    domain::String
    area::Real
    symm_factor::Real
end

struct ConductorStranded <: FECoupling
    domain::String
    area::Real
    symm_factor::Real
    N::Int # Number of strands
end

struct CircuitHandler
    data_type::DataType
    dh::Ferrite.AbstractDofHandler
    cond_sol::Vector{ConductorSolid}
    cond_str::Vector{ConductorStranded}
end

function CircuitHandler(dh::Ferrite.AbstractDofHandler, val_type::DataType)
    @assert Ferrite.isclosed(dh)

    CircuitHandler(val_type, dh, ConductorSolid[], ConductorStranded[])
end

function Base.show(io::IO, ::MIME"text/plain", ch::CircuitHandler)
    println(io, "CircuitHandler:")
    print(io, "  Solid conductors:")
    for c in ch.cond_sol
        print(io, "\n    ", "Domain: ", c.domain)
    end

    print(io, "  Stranded conductors:")
    for c in ch.cond_str
        print(io, "\n    ", "Domain: ", c.domain, ", N: ", c.N)
    end
end

function add_conductor_solid!(ch::CircuitHandler, domain::String)
    add_conductor_solid!(ch, domain, 1)
end

function add_conductor_solid!(ch::CircuitHandler, domain::String, symm_factor::Real)
    @assert (symm_factor >= 0) && (symm_factor <= 1) "Symmetry factor must be between 0 and 1"
    area = get_domain_area(ch.dh, domain) / symm_factor
    push!(ch.cond_sol, ConductorSolid(domain, area, symm_factor))
end

function add_conductor_stranded!(ch::CircuitHandler, domain::String, N::Integer)
    add_conductor_stranded!(ch, domain, N, 1)
end

function add_conductor_stranded!(ch::CircuitHandler, domain::String, N::Integer, symm_factor::Real)
    @assert (symm_factor >= 0) && (symm_factor <= 1) "Symmetry factor must be between 0 and 1"
    area = get_domain_area(ch.dh, domain) / symm_factor
    push!(ch.cond_str, ConductorStranded(domain, area, symm_factor, N))
end

function ncond_sol(ch::CircuitHandler)
    return length(ch.cond_sol)
end

function ncond_str(ch::CircuitHandler)
    return length(ch.cond_str)
end

function nconductors(ch::CircuitHandler)
    return ncond_sol(ch) + ncond_str(ch)
end

function get_ndofs(ch::CircuitHandler)
    return nconductors(ch)
end

function get_conductor_type(ch::CircuitHandler, domain::String)
    for conductor ∈ ch.cond_str
        if(conductor.domain == domain)
            return ConductorTypeStranded()
        end
    end
    for conductor ∈ ch.cond_sol
        if(conductor.domain == domain)
            return ConductorTypeSolid()
        end
    end

    return ConductorTypeNone()
end

function get_domain_area(dh::DofHandler, domain::String)
    Sdom = 0
    for sdh ∈ dh.subdofhandlers
        cell_type = getcelltype(sdh)
        ref_shape = getrefshape(cell_type)

        ip = Lagrange{ref_shape,1}()
        qr = QuadratureRule{ref_shape}(2)
        cv = CellValues(qr, ip)

        Sdom += get_domain_area(sdh, cv, domain)
    end

    return Sdom
end

function get_domain_area(sdh::SubDofHandler, cv::CellValues, domain::String)
    Sdom = 0
    domain_set = getcellset(sdh.dh.grid, domain)
    for cell ∈ CellIterator(sdh)
        cell_id = cellid(cell)
        if (cell_id ∉ domain_set)
            continue
        end
        reinit!(cv, cell)

        for q_point in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q_point)
            Sdom += dΩ
        end
    end

    return Sdom
end

function add_sparsity_circuit!(sp::SparsityPattern, dh::DofHandler, ch::CircuitHandler)
    for (p, conductor) ∈ enumerate(ch.cond_str)
        coupling_idx = ndofs(dh) + p
        add_sparsity_fecoupling!(sp, dh, conductor, coupling_idx)
    end

    for (q, conductor) ∈ enumerate(ch.cond_sol)
        coupling_idx = ndofs(dh) + ncond_str(ch) + q
        add_sparsity_fecoupling!(sp, dh, conductor, coupling_idx)
    end
end

function add_sparsity_fecoupling!(sp::SparsityPattern, dh::DofHandler, fe_coupling::FECoupling, coupling_idx::Integer)
    Ferrite.add_entry!(sp, coupling_idx, coupling_idx)

    cells = getcellset(dh.grid, fe_coupling.domain)
    for cell ∈ cells
        dofs = celldofs(dh, cell)
        for dof ∈ dofs
            Ferrite.add_entry!(sp, coupling_idx, dof)
            Ferrite.add_entry!(sp, dof, coupling_idx)
        end
    end
end

function apply_circuit_couplings!(K::SparseMatrixCSC, f::Vector, dh::DofHandler, cv::CV, ch::CircuitHandler, problem::Problem, params::Vector{CellParams}) where {CV<:NamedTuple}
    for (p, conductor) ∈ enumerate(ch.cond_str)
        coupling_idx = ndofs(dh) + p
        conductor.symm_factor != 1 && error("Non-unity symmetry factor not yet supported for stranded conductor $(conductor.domain)")
        apply_conductor!(K, f, dh, cv, problem.time, problem.symmetry, params, conductor, coupling_idx)
    end

    for (q, conductor) ∈ enumerate(ch.cond_sol)
        coupling_idx = ndofs(dh) + ncond_str(ch) + q
        conductor.symm_factor != 1 && error("Non-unity symmetry factor not yet supported for solid conductor $(conductor.domain)")
        apply_conductor!(K, f, dh, cv, problem.time, problem.symmetry, params, conductor, coupling_idx)
    end

    return K, f
end

function apply_conductor!(K::SparseMatrixCSC, f::Vector, dh::DofHandler, cv::CV, time::TimeHarmonic, symmetry::Symmetry2D, params::Vector{CellParams}, conductor::ConductorSolid, coupling_idx::Int) where {CV<:NamedTuple}
    G = 0
    ℓ = get_modeldepth(symmetry, 0) # TODO axisymmetric
    for sdh ∈ dh.subdofhandlers
        cell_type = getcelltype(sdh)
        cv_ = get_cellvalues(cv, cell_type)

        K, f, G_ = apply_conductor!(K, f, sdh, cv_, symmetry, params, conductor, coupling_idx)
        G += G_
    end

    χ = 1 / (1im * time.ω * ℓ)

    K[coupling_idx, coupling_idx] = χ * G
    f[coupling_idx] = χ
end

function apply_conductor!(K::SparseMatrixCSC, f::Vector, sdh::SubDofHandler, cv::CellValues, symm::Symmetry2D, params::Vector{CellParams}, conductor::ConductorSolid, coupling_idx::Int)
    n_basefuncs = getnbasefunctions(cv)
    Qe = zeros(Complex{Float64}, n_basefuncs)
    G = 0

    domain_set = getcellset(sdh.dh.grid, conductor.domain)
    for cell ∈ CellIterator(sdh)
        cell_id = cellid(cell)
        if (cell_id ∉ domain_set)
            continue
        end
        reinit!(cv, cell)

        # Reset to 0
        fill!(Qe, 0)

        # Retrieve physical parameters
        param = params[cell_id]
        x = getcoordinates(sdh.dh.grid, cell_id)

        # Loop over quadrature points
        for q_point in 1:getnquadpoints(cv)
            # Get the quadrature weight
            coord = spatial_coordinate(cv, q_point, x)
            dΩ = getdetJdV(cv, q_point)

            σe = param.σ
            ℓe = get_modeldepth(symm, coord[1])
            G += σe / ℓe * dΩ

            # Loop over test shape functions
            for i in 1:n_basefuncs
                v = shape_value(cv, q_point, i)

                Qe[i] += σe / ℓe * v * dΩ
            end
        end

        K[coupling_idx, celldofs(cell)] .-= Qe
        K[celldofs(cell), coupling_idx] .-= Qe
    end

    return K, f, G
end

function apply_conductor!(K::SparseMatrixCSC, f::Vector, dh::DofHandler, cv::CV, time::TimeHarmonic, symmetry::Symmetry2D, params::Vector{CellParams}, conductor::ConductorStranded, coupling_idx::Int) where {CV<:NamedTuple}
    G = 0
    ℓ = get_modeldepth(symmetry, 0) # TODO axisymmetric
    for sdh ∈ dh.subdofhandlers
        cell_type = getcelltype(sdh)
        cv_ = get_cellvalues(cv, cell_type)

        K, f, G_ = apply_conductor!(K, f, sdh, cv_, symmetry, params, conductor, coupling_idx)
        G += G_
    end

    χ = 1 / (1im * time.ω * ℓ)
    R = conductor.N^2 / conductor.area / G

    K[coupling_idx, coupling_idx] = -χ * R
    f[coupling_idx] = χ
end

function apply_conductor!(K::SparseMatrixCSC, f::Vector, sdh::SubDofHandler, cv::CellValues, symm::Symmetry2D, params::Vector{CellParams}, conductor::ConductorStranded, coupling_idx::Int)
    n_basefuncs = getnbasefunctions(cv)
    Pe = zeros(Complex{Float64}, n_basefuncs)
    G = 0

    domain_set = getcellset(sdh.dh.grid, conductor.domain)
    for cell ∈ CellIterator(sdh)
        cell_id = cellid(cell)
        if (cell_id ∉ domain_set)
            continue
        end
        reinit!(cv, cell)

        # Reset to 0
        fill!(Pe, 0)

        # Retrieve physical parameters
        param = params[cell_id]
        x = getcoordinates(sdh.dh.grid, cell_id)

        # Loop over quadrature points
        for q_point in 1:getnquadpoints(cv)
            # Get the quadrature weight
            coord = spatial_coordinate(cv, q_point, x)
            dΩ = getdetJdV(cv, q_point)

            σe = param.σ
            ℓe = get_modeldepth(symm, coord[1])
            G += σe / ℓe * dΩ

            # Loop over test shape functions
            for i in 1:n_basefuncs
                v = shape_value(cv, q_point, i)

                Pe[i] += conductor.N / conductor.area * v * dΩ
            end
        end

        K[coupling_idx, celldofs(cell)] .-= Pe
        K[celldofs(cell), coupling_idx] .-= Pe
    end

    return K, f, G
end