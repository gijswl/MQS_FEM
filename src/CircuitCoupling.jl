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
    # idx0 = ndofs(dh)
    # for (p, conductor) ∈ enumerate(ch.cond_str)
    #     coupling_idx = idx0 + p
    #     add_sparsity_fecoupling!(sp, dh, conductor, coupling_idx)
    # end

    # idx0 = ndofs(dh) + ncond_str(ch)
    # for (q, conductor) ∈ enumerate(ch.cond_sol)
    #     coupling_idx = idx0 + q
    #     add_sparsity_fecoupling!(sp, dh, conductor, coupling_idx)
    # end
end

function add_sparsity_fecoupling!(sp::SparsityPattern, dh::DofHandler, fe_coupling::AbstractFECoupling, coupling_idx::Integer)
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

# for (p, conductor) ∈ enumerate(ch.cond_str)
#     coupling_idx = ndofs(dh) + p
#     conductor.symm_factor != 1 && error("Non-unity symmetry factor not yet supported for stranded conductor $(conductor.domain)")
#     apply_conductor!(K, f, dh, cv, problem.time, problem.symmetry, params, conductor, coupling_idx)
# end

# for (q, conductor) ∈ enumerate(ch.cond_sol)
#     coupling_idx = ndofs(dh) + ncond_str(ch) + q
#     conductor.symm_factor != 1 && error("Non-unity symmetry factor not yet supported for solid conductor $(conductor.domain)")
#     apply_conductor!(K, f, dh, cv, problem.time, problem.symmetry, params, conductor, coupling_idx)
# end
function apply_circuit_couplings!(K::SparseMatrixCSC, f::Vector, dh::DofHandler, cv::CV, ch::CircuitHandler, problem::Problem, params::Vector{CellParams}) where {CV<:NamedTuple}
    @assert isclosed(ch)
    circuit = ch.circuit

    ℓ = get_modeldepth(problem.symmetry, 0) # TODO axisymmetric
    χ = 1 / (1im * problem.time.ω * ℓ)

    idx = size(K, 1)-get_ndofs(ch)+1:size(K, 1)
    K[idx, idx] .= χ * circuit.S
    f[idx] .= χ * circuit.W

    tree_sol = filter(e -> typeof(e) <: CircuitCoilSolid, circuit.tree)
    tree_str = filter(e -> typeof(e) <: CircuitCoilStranded, circuit.tree)
    cotree_str = filter(e -> typeof(e) <: CircuitCoilStranded, circuit.cotree)
    cotree_sol = filter(e -> typeof(e) <: CircuitCoilSolid, circuit.cotree)

    segmentation = circuit.coupling["segmentation"]
    start_idx = cumsum(segmentation)
    start_idx = [1; 1 .+ start_idx[1:end-1]]

    F = zeros(size(K, 1), get_ndofs(ch))

    # Column 1: Stranded conductors
    Dstrstr = circuit.coupling["Dstrstr"]
    Dstri = circuit.coupling["Dstri"]
    Ii = circuit.coupling["Ii"]

    Pstr = zeros(size(K, 1))
    for (j, coil) ∈ enumerate(cotree_str)
        fill!(Pstr, 0)

        conductor = cch.fecoupling[coil.domain]
        apply_conductor!(Pstr, dh, cv, problem.symmetry, params, conductor)
        F[:, start_idx[1]+j-1] .+= Pstr
    end
    for (j, coil) ∈ enumerate(tree_str)
        fill!(Pstr, 0)

        conductor = cch.fecoupling[coil.domain]
        apply_conductor!(Pstr, dh, cv, problem.symmetry, params, conductor)
        F[:, start_idx[1]:start_idx[2]-1] .-= Pstr * transpose(Dstrstr[j, :])
        f .-= Pstr * transpose(Dstri[j, :]) * Ii
    end

    # Column 3: Solid conductors
    Bsolsol = circuit.coupling["Bsolsol"]
    Bsolv = circuit.coupling["Bsolv"]
    Vv = circuit.coupling["Vv"]

    Qsol = zeros(size(K, 1))
    for (j, coil) ∈ enumerate(tree_sol)
        fill!(Qsol, 0)

        conductor = cch.fecoupling[coil.domain]
        apply_conductor!(Qsol, dh, cv, problem.symmetry, params, conductor)
        F[:, start_idx[3]+j-1] .+= Qsol
    end
    for (j, coil) ∈ enumerate(cotree_sol)
        fill!(Qsol, 0)

        conductor = cch.fecoupling[coil.domain]
        apply_conductor!(Qsol, dh, cv, problem.symmetry, params, conductor)
        F[:, start_idx[3]:start_idx[4]-1] .-= Qsol * transpose(Bsolsol[j, :])
        f .-= Qsol * transpose(Bsolv[j, :]) * Vv
    end

    K[:, idx] .-= F
    K[idx, :] .-= transpose(F)

    return K, f
end

function apply_conductor!(Qsol::Vector, dh::DofHandler, cv::CV, symmetry::Symmetry2D, params::Vector{CellParams}, conductor::ConductorSolid) where {CV<:NamedTuple}
    for sdh ∈ dh.subdofhandlers
        cell_type = getcelltype(sdh)
        cv_ = get_cellvalues(cv, cell_type)

        apply_conductor!(Qsol, sdh, cv_, symmetry, params, conductor)
    end
end

function apply_conductor!(Qsol::Vector, sdh::SubDofHandler, cv::CellValues, symm::Symmetry2D, params::Vector{CellParams}, conductor::ConductorSolid)
    n_basefuncs = getnbasefunctions(cv)
    Qe = zeros(Float64, n_basefuncs)

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

            # Loop over test shape functions
            for i in 1:n_basefuncs
                v = shape_value(cv, q_point, i)

                Qe[i] += σe / ℓe * v * dΩ
            end
        end

        Qsol[celldofs(cell)] .+= Qe
    end

    return Qsol
end

function apply_conductor!(Pstr::Vector, dh::DofHandler, cv::CV, symmetry::Symmetry2D, params::Vector{CellParams}, conductor::ConductorStranded) where {CV<:NamedTuple}
    for sdh ∈ dh.subdofhandlers
        cell_type = getcelltype(sdh)
        cv_ = get_cellvalues(cv, cell_type)

        apply_conductor!(Pstr, sdh, cv_, symmetry, params, conductor)
    end
end

function apply_conductor!(Pstr::Vector, sdh::SubDofHandler, cv::CellValues, symm::Symmetry2D, params::Vector{CellParams}, conductor::ConductorStranded)
    n_basefuncs = getnbasefunctions(cv)
    Pe = zeros(Complex{Float64}, n_basefuncs)

    domain_set = getcellset(sdh.dh.grid, conductor.domain)
    for cell ∈ CellIterator(sdh)
        cell_id = cellid(cell)
        if (cell_id ∉ domain_set)
            continue
        end
        reinit!(cv, cell)

        # Reset to 0
        fill!(Pe, 0)

        # Loop over quadrature points
        for q_point in 1:getnquadpoints(cv)
            # Get the quadrature weight
            dΩ = getdetJdV(cv, q_point)

            # Loop over test shape functions
            for i in 1:n_basefuncs
                v = shape_value(cv, q_point, i)

                Pe[i] += conductor.N / conductor.area * v * dΩ
            end
        end

        Pstr[celldofs(cell)] .-= Pe
    end

    return Pstr
end