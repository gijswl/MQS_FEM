function get_modeldepth(symmetry::Planar2D, ::Any)
    return symmetry.depth
end

function get_modeldepth(::Axi2D, x::Vec{2,<:Real})
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

function get_material_params(material)
    # Source current
    J0 = 0.0
    if (haskey(material, "J0"))
        J0 = material["J0"]
    end

    # Conductivity
    σ = 0.0
    if (haskey(material, "σ"))
        σ = material["σ"]
    end

    # Permeability / reluctivity
    ν_ = 1 / μ0
    if (haskey(material, "μr"))
        μr_ = material["μr"]
        ν_ = 1 ./ (μ0 * μr_)
    end
    if (haskey(material, "ν"))
        ν_ = material["ν"]
    end
    if (length(ν_) == 2)
        ν = Tensor{2,2}((ν_[1], 0, 0, ν_[2]))
    else
        ν = Tensor{2,2}((ν_, 0, 0, ν_))
    end

    return J0, σ, ν
end

function preprocess_grid(grid)
    cells_tri = Int64[]
    cells_quad = Int64[]
    for (cell_num, cell) ∈ enumerate(grid.cells)
        shape = getrefshape(cell)

        if (shape == RefTriangle)
            push!(cells_tri, cell_num)
        elseif (shape == RefQuadrilateral)
            push!(cells_quad, cell_num)
        else
            error("Reference shape $shape not implemented")
        end
    end

    addcellset!(grid, "cells_tri", cells_tri)
    addcellset!(grid, "cells_quad", cells_quad)
end

function init_problem(problem::Problem, grid::Grid{2})
    fe_order = problem.fe_order
    qr_order = problem.qr_order
    ip_tri = Lagrange{RefTriangle,fe_order}()
    ip_quad = Lagrange{RefQuadrilateral,fe_order}()

    cells_tri = getcellset(grid, "cells_tri")
    cells_quad = getcellset(grid, "cells_quad")

    dh = DofHandler(grid)
    if (!isempty(cells_tri))
        sdh_tri = SubDofHandler(dh, cells_tri)
        add!(sdh_tri, :A, ip_tri)
    end
    if (!isempty(cells_quad))
        sdh_quad = SubDofHandler(dh, cells_quad)
        add!(sdh_quad, :A, ip_quad)
    end
    close!(dh)

    qr_tri = QuadratureRule{RefTriangle}(qr_order)
    qr_quad = QuadratureRule{RefQuadrilateral}(qr_order)
    cv_tri = CellValues(qr_tri, ip_tri)
    cv_quad = CellValues(qr_quad, ip_quad)

    return (tri=cv_tri, quad=cv_quad), dh
end

function init_params(dh::DofHandler, ch::CircuitHandler, problem::Problem)
    ν0 = 1 / μ0
    default = CellParams(0.0, 0.0, Tensor{2,2}((ν0, 0.0, 0.0, ν0)), ConductorTypeNone())

    Ncells = getncells(grid)
    params = [default for _ ∈ 1:Ncells]
    for (domain, material) ∈ problem.materials
        J0, σ, ν = get_material_params(material)
        cond_type = get_conductor_type(ch, domain)
        param = CellParams(J0, σ, ν, cond_type)

        for cell ∈ getcellset(dh.grid, domain)
            params[cell] = param
        end
    end

    return params
end

function init_constraints(dh::DofHandler, problem::Problem)
    # Define the boundary conditions using a constraint handler
    ch = ConstraintHandler(dh)

    # Add boundary conditions to the constraint handler
    for (boundary, condition) ∈ problem.boundaries
        bc = create_bc(dh, :A, condition, boundary)
        add!(ch, bc)
    end

    close!(ch)
    Ferrite.update!(ch, 0.0) # Since the BCs do not depend on time, update them once at t = 0.0

    return ch
end

function allocate(dh::DofHandler, ::Problem{T}) where {T}
    K = allocate_matrix(SparseMatrixCSC{T,Int}, dh)

    return K
end

function allocate(dh::DofHandler, cch::CircuitHandler, ::Problem{T}) where {T}
    ndof = ndofs(dh) + get_ndofs(cch)
    sp = SparsityPattern(ndof, ndof; nnz_per_row=2 * ndofs_per_cell(dh.subdofhandlers[1])) # How to optimize nnz_per_row?
    add_sparsity_entries!(sp, dh)
    add_sparsity_circuit!(sp, dh, cch)

    K = allocate_matrix(SparseMatrixCSC{T,Int}, sp)

    return K
end

get_cellvalues(cv::CV, ::Type{Triangle}) where {CV<:NamedTuple} = cv.tri
get_cellvalues(cv::CV, ::Type{Quadrilateral}) where {CV<:NamedTuple} = cv.quad

function assemble_global(K::SparseMatrixCSC, dh::DofHandler, cv::CV, problem::Problem{T}, cellparams::Vector{CellParams}) where {T,CV<:NamedTuple}
    # Allocate global force vector f
    f = zeros(T, size(K, 1))

    # Create an assembler
    assembler = start_assemble(K, f)

    for sdh ∈ dh.subdofhandlers
        cell_type = getcelltype(sdh)
        cv_ = get_cellvalues(cv, cell_type)

        assemble_global!(assembler, sdh, cv_, problem, cellparams)
    end

    return K, f
end

function assemble_global!(assembler::Ferrite.AbstractAssembler, sdh::SubDofHandler, cv::CellValues, problem::Problem{T}, cellparams::Vector{CellParams}) where {T}
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(T, n_basefuncs, n_basefuncs)
    fe = zeros(T, n_basefuncs)

    # Loop over all cells in the sub dof handler
    for cell ∈ CellIterator(sdh)
        reinit!(cv, cell)
        cell_id = cellid(cell)

        param = cellparams[cell_id]
        x = getcoordinates(sdh.dh.grid, cell_id)

        # Compute element contribution
        assemble_element!(problem.symmetry, problem.time, Ke, fe, cv, param, x)

        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
end

function assemble_element!(::Planar2D, time::TimeStatic, Ke::Matrix, fe::Vector, cv::CellValues, param::CellParams, x::Vector{<:Vec{2}})
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
            fe[i] += param.J0 * v * dΩ

            # Loop over trial shape functions
            for j ∈ 1:n_basefuncs
                ∇u = shape_gradient(cv, q_point, j)

                # Add contribution to Ke
                Ke[i, j] += (∇v ⋅ param.ν ⋅ ∇u) * dΩ
            end
        end
    end

    return Ke, fe
end

function assemble_element!(::Planar2D, time::TimeHarmonic, Ke::Matrix, fe::Vector, cv::CellValues, param::CellParams, x::Vector{<:Vec{2}})
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
            fe[i] += param.J0 * v * dΩ

            # Loop over trial shape functions
            for j ∈ 1:n_basefuncs
                u = shape_value(cv, q_point, j)
                ∇u = shape_gradient(cv, q_point, j)

                # Add contribution to Ke
                if (typeof(param.cond_type) <: ConductorTypeStranded)
                    Ke[i, j] += (∇v ⋅ param.ν ⋅ ∇u) * dΩ
                else
                    Ke[i, j] += ((∇v ⋅ param.ν ⋅ ∇u) + 1im * ω * param.σ * (v * u)) * dΩ
                end
            end
        end
    end

    return Ke, fe
end