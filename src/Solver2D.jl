function get_modeldepth(problem::Problem2D, symmetry::Planar2D, ::Any)
    return symmetry.depth
end

function get_modeldepth(::Problem2D, ::Axi2D, x::Vec{2,<:Real})
    return 2π * x[1]
end

function create_bc(dh::DofHandler, field::Symbol, bc::BoundaryA, boundary::String)
    return Dirichlet(
        field,
        getfacetset(dh.grid, boundary),
        (x, t) -> bc.A
    )
end

function create_bc(dh::DofHandler, field::Symbol, bc::BoundaryFlux, boundary::String)
    return Dirichlet(
        field,
        getfacetset(dh.grid, boundary),
        (x, t) -> x[2] * bc.B[1] - x[1] * bc.B[2]
    )
end

function apply_material!(J0, σ, ν, cellset, material)
    if (haskey(material, "σ"))
        σ[cellset] .= material["σ"]
    end
    if (haskey(material, "μr"))
        μr_ = material["μr"]
        ν_ = 1 ./ (μ0 * μr_)
        if (length(ν_) == 2)
            for cell in cellset
                ν[cell] = Tensor{2,2,Complex{Float64}}((ν_[1], 0, 0, ν_[2]))
            end
        else
            for cell in cellset
                ν[cell] = Tensor{2,2,Complex{Float64}}((ν_, 0, 0, ν_))
            end
        end
    end
    if (haskey(material, "ν"))
        ν_ = material["ν"]
        if (length(ν_) == 2)
            for cell in cellset
                ν[cell] = Tensor{2,2,Complex{Float64}}((ν_[1], 0, 0, ν_[2]))
            end
        else
            for cell in cellset
                ν[cell] = Tensor{2,2,Complex{Float64}}((ν_, 0, 0, ν_))
            end
        end
    end
    if (haskey(material, "J0"))
        J0[cellset] .= material["J0"]
    end

    return J0, σ, ν
end

function init_problem(problem::Problem2D, grid::Grid{2})
    refshape = getrefshape(grid.cells[1])

    fe_order = problem.fe_order
    qr_order = problem.qr_order

    ip_fe = Lagrange{refshape,fe_order}()
    ip_geo = Lagrange{refshape,1}()
    qr = QuadratureRule{refshape}(qr_order)
    cv = CellValues(qr, ip_fe, ip_geo)

    dh = DofHandler(grid)
    add!(dh, :A, ip_fe)
    close!(dh)

    return cv, dh
end

function init_params(dh::DofHandler, problem::Problem2D{T}) where {T}
    grid = dh.grid

    Ncells = getncells(grid)
    J0 = zeros(T, Ncells)
    σ = zeros(T, Ncells)
    ν = [Tensor{2,2,T}((1 / μ0, 0, 0, 1 / μ0)) for _ in 1:Ncells]

    for (domain, material) ∈ problem.materials
        cellset = collect(getcellset(grid, domain))

        apply_material!(J0, σ, ν, cellset, material)
    end

    return CellParams(J0, σ, ν)
end

function init_constraints(dh::DofHandler, problem::Problem2D)
    # Define the boundary conditions using a constraint handler
    ch = ConstraintHandler(dh)

    # Add boundary conditions to the constraint handler
    for (boundary, condition) ∈ problem.boundaries
        bc = create_bc(dh, :A, condition, boundary)
        add!(ch, bc)
    end

    close!(ch)
    update!(ch, 0.0) # Since the BCs do not depend on time, update them once at t = 0.0

    return ch
end

function allocate(dh::DofHandler, ::Problem2D{T}) where {T}
    K = allocate_matrix(SparseMatrixCSC{T,Int}, dh)

    return K
end

function allocate(dh::DofHandler, cch::CircuitHandler, ::Problem2D{T}) where {T}
    ndof = ndofs(dh) + ncouplings(cch)
    sp = SparsityPattern(ndof, ndof; nnz_per_row=2 * ndofs_per_cell(dh.subdofhandlers[1])) # How to optimize nnz_per_row?
    add_sparsity_entries!(sp, dh)
    add_sparsity_circuit!(sp, dh, cch)

    K = allocate_matrix(SparseMatrixCSC{T,Int}, sp)

    return K
end

function assemble_global(K::SparseMatrixCSC, dh::DofHandler, cv::CellValues, problem::Problem2D{T}, cellparams::CellParams) where {T}
    # Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(T, n_basefuncs, n_basefuncs)
    fe = zeros(T, n_basefuncs)

    # Allocate global force vector f
    f = zeros(T, size(K, 1))

    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cels
    for cell ∈ CellIterator(dh)
        reinit!(cv, cell)
        cell_id = cellid(cell)

        Je = cellparams.J0[cell_id]
        σe = cellparams.σ[cell_id]
        νe = cellparams.ν[cell_id]
        x = getcoordinates(dh.grid, cell_id)

        # Compute element contribution
        assemble_element!(problem.symmetry, problem.time, Ke, fe, cv, Je, σe, νe, x)

        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

function assemble_element!(::Planar2D, time::TimeStatic, Ke::Matrix, fe::Vector, cv::CellValues, Je::T, σe::T, νe::Tensor{2,2,<:T}, x::Vector{<:Vec{2}}) where {T}
    n_basefuncs = getnbasefunctions(cv)

    # Reset local contribution to 0
    fill!(Ke, 0)
    fill!(fe, 0)

    # Loop over quadrature points
    for q_point ∈ 1:getnquadpoints(cv)
        # Get the quadrature weight
        dΩ = getdetJdV(cv, q_point)

        # Loop over test shape functions
        for i ∈ 1:n_basefuncs
            v = shape_value(cv, q_point, i)
            ∇v = shape_gradient(cv, q_point, i)

            # Add contribution to fe
            fe[i] += Je * v * dΩ

            # Loop over trial shape functions
            for j ∈ 1:n_basefuncs
                ∇u = shape_gradient(cv, q_point, j)

                # Add contribution to Ke
                Ke[i, j] += (∇v ⋅ νe ⋅ ∇u) * dΩ
            end
        end
    end

    return Ke, fe
end

function assemble_element!(::Planar2D, time::TimeHarmonic, Ke::Matrix, fe::Vector, cv::CellValues, Je::T, σe::T, νe::Tensor{2,2,<:T}, x::Vector{<:Vec{2}}) where {T}
    n_basefuncs = getnbasefunctions(cv)
    ω = time.ω

    # Reset local contribution to 0
    fill!(Ke, 0)
    fill!(fe, 0)

    # Loop over quadrature points
    for q_point ∈ 1:getnquadpoints(cv)
        # Get the quadrature weight
        dΩ = getdetJdV(cv, q_point)

        # Loop over test shape functions
        for i ∈ 1:n_basefuncs
            v = shape_value(cv, q_point, i)
            ∇v = shape_gradient(cv, q_point, i)

            # Add contribution to fe
            fe[i] += Je * v * dΩ

            # Loop over trial shape functions
            for j ∈ 1:n_basefuncs
                u = shape_value(cv, q_point, j)
                ∇u = shape_gradient(cv, q_point, j)

                # Add contribution to Ke
                Ke[i, j] += ((∇v ⋅ νe ⋅ ∇u) + 1im * ω * σe * (v * u)) * dΩ
            end
        end
    end

    return Ke, fe
end