function write_postprocessed(vtk::VTKGridFile, dh::DofHandler{sdim}, ch::CircuitHandler, u::Vector{T}, problem::Problem, cellparams::Vector{CellParams}, quantity::Symbol) where {sdim,T}
    return write_postprocessed(vtk, dh, ch, u, problem, cellparams, quantity, string(quantity))
end

function write_postprocessed(vtk::VTKGridFile, dh::DofHandler{sdim}, ch::CircuitHandler, u::Vector{T}, problem::Problem, cellparams::Vector{CellParams}, quantity::Symbol, name::String) where {sdim,T}
    cellnodes = vtk.cellnodes
    fieldname = :A

    # Make sure the field exists
    fieldname ∈ Ferrite.getfieldnames(dh) || error("Field $fieldname not found.")
    # Figure out the return type (scalar or vector)
    field_idx = Ferrite.find_field(dh, fieldname)
    ip = Ferrite.getfieldinterpolation(dh, field_idx)

    get_vtk_dim(::Val{:B_norm}) = 1
    get_vtk_dim(::Val{:B_real}) = 2
    get_vtk_dim(::Val{:B_imag}) = 2
    get_vtk_dim(::Val{:J_norm}) = 1
    get_vtk_dim(::Val{:S_real}) = 1
    get_vtk_dim(::Val{:S_imag}) = 1

    vtk_dim = get_vtk_dim(Val{quantity}())
    n_vtk_nodes = maximum(maximum, cellnodes)
    data = fill(NaN * zero(Float64), vtk_dim, n_vtk_nodes)

    for sdh in dh.subdofhandlers
        # Check if this sdh contains this field, otherwise continue to the next
        field_idx = Ferrite._find_field(sdh, fieldname)
        field_idx === nothing && continue

        # Set up CellValues with the local node coords as quadrature points
        CT = Ferrite.getcelltype(sdh)
        ip = Ferrite.getfieldinterpolation(sdh, field_idx)
        ip_geo = Ferrite.geometric_interpolation(CT)
        local_node_coords = Ferrite.reference_coordinates(ip_geo)
        qr = QuadratureRule{getrefshape(ip)}(zeros(length(local_node_coords)), local_node_coords)
        cv = CellValues(qr, ip, ip_geo^sdim)
        drange = dof_range(sdh, field_idx)
        # Function barrier
        _evaluate_postprocessed_at_discontinuous_vtkgrid_nodes!(data, sdh, ch, u, cv, drange, cellnodes, problem, quantity, cellparams)
    end

    Ferrite._vtk_write_node_data(vtk.vtk, data, name)
    return vtk
end

function _post_process(problem, sdh::SubDofHandler, ch::CircuitHandler, ::Val{:B_norm}, cell_num, uq, uc, ∇uq, xq, param::CellParams)
    Bre_q, Bim_q = ComputeFluxDensity(∇uq, xq, problem.symmetry)
    return sqrt(Bre_q[1]^2 + Bre_q[2]^2 + Bim_q[1]^2 + Bim_q[2]^2)
end

function _post_process(problem, sdh::SubDofHandler, ch::CircuitHandler, ::Val{:B_real}, cell_num, uq, uc, ∇uq, xq, param::CellParams)
    Bre_q, _ = ComputeFluxDensity(∇uq, xq, problem.symmetry)
    return Ferrite.Vec{2,Float64}(Bre_q)
end

function _post_process(problem, sdh::SubDofHandler, ch::CircuitHandler, ::Val{:B_imag}, cell_num, uq, uc, ∇uq, xq, param::CellParams)
    _, Bim_q = ComputeFluxDensity(∇uq, xq, problem.symmetry)
    return Ferrite.Vec{2,Float64}(Bim_q)
end

function _post_process(problem, sdh::SubDofHandler, ch::CircuitHandler, ::Val{:J_norm}, cell_num, uq, uc, ∇uq, xq, param::CellParams)
    ω = get_frequency(problem)

    Jsource = param.J0
    Jeddy = -1im * param.σ * ω * uq

    for (i, coupling) ∈ enumerate(ch.coupling)
        if cell_num ∈ getcellset(sdh.dh.grid, coupling.domain)
            Jsource += uc[i] / (coupling.symm_factor * coupling.area)
        end
    end

    return norm(Jsource + Jeddy)
end

function _post_process(problem, sdh::SubDofHandler, ch::CircuitHandler, ::Val{:S_real}, cell_num, uq, uc, ∇uq, xq, param::CellParams)
    ω = get_frequency(problem)

    B_real = _post_process(problem, sdh, ch, Val{:B_real}(), cell_num, uq, uc, ∇uq, xq, param)
    B_imag = _post_process(problem, sdh, ch, Val{:B_imag}(), cell_num, uq, uc, ∇uq, xq, param)
    J_norm = _post_process(problem, sdh, ch, Val{:J_norm}(), cell_num, uq, uc, ∇uq, xq, param)

    B_q = B_q = Vec{2}((B_real[1] + 1im * B_imag[1], B_real[2] + 1im * B_imag[2]))

    sm = 0.5im * ω * B_q ⋅ Vec{2}(conj(param.ν ⋅ B_q))
    if (norm(J_norm) > 0)
        se = norm(J_norm)^2 / (2 * param.σ)
    else
        se = 0
    end

    return real(sm + se)
end

function _post_process(problem, sdh::SubDofHandler, ch::CircuitHandler, ::Val{:S_imag}, cell_num, uq, uc, ∇uq, xq, param::CellParams)
    ω = get_frequency(problem)

    B_real = _post_process(problem, sdh, ch, Val{:B_real}(), cell_num, uq, uc, ∇uq, xq, param)
    B_imag = _post_process(problem, sdh, ch, Val{:B_imag}(), cell_num, uq, uc, ∇uq, xq, param)
    J_norm = _post_process(problem, sdh, ch, Val{:J_norm}(), cell_num, uq, uc, ∇uq, xq, param)

    B_q = B_q = Vec{2}((B_real[1] + 1im * B_imag[1], B_real[2] + 1im * B_imag[2]))

    sm = 0.5im * ω * B_q ⋅ Vec{2}(conj(param.ν ⋅ B_q))
    if (norm(J_norm) > 0)
        se = norm(J_norm)^2 / (2 * param.σ)
    else
        se = 0
    end

    return imag(sm + se)
end

function _evaluate_postprocessed_at_discontinuous_vtkgrid_nodes!(
    data::Matrix, sdh::SubDofHandler, ch::CircuitHandler,
    u::Vector{T}, cv::CellValues, drange::UnitRange, cellnodes, problem::Problem, quantity::Symbol, cellparams::Vector{CellParams}
) where {T}
    ue = zeros(T, length(drange))
    uc = u[end-get_ndofs(ch)+1:end]
    for cell in CellIterator(sdh)
        reinit!(cv, cell)
        @assert getnquadpoints(cv) == length(cell.nodes)

        cell_num = cellid(cell)
        xe = getcoordinates(sdh.dh.grid, cell_num)
        param = cellparams[cell_num]

        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]]
        end
        for (qp, nodeid) in pairs(cellnodes[cellid(cell)])
            uq = function_value(cv, qp, ue)
            ∇uq = function_gradient(cv, qp, ue)
            xq = spatial_coordinate(cv, qp, xe)
            val = _post_process(problem, sdh, ch, Val{quantity}(), cell_num, uq, uc, ∇uq, xq, param)

            dataview = @view data[:, nodeid]
            fill!(dataview, 0) # purge the NaN
            Ferrite.toparaview!(dataview, val)
        end
    end
    return data
end