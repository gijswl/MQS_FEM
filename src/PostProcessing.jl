function get_nquadpoints(dh::DofHandler, cv::CV) where {CV<:NamedTuple}
    n_quadpts = 0
    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        n_quadpts = max(n_quadpts, getnquadpoints(cv_))
    end

    return n_quadpts
end

function ComputeFluxDensity(∇Aq::Vec{2,T}, ::Vec{2}, ::Planar2D) where {T}
    ∇Aq_re = real(∇Aq)
    ∇Aq_im = imag(∇Aq)

    Bre_x = ∇Aq_re[2]
    Bre_y = -∇Aq_re[1]
    Bim_x = ∇Aq_im[2]
    Bim_y = -∇Aq_im[1]

    return Ferrite.Vec{2,Float64}((Bre_x, Bre_y)), Ferrite.Vec{2,Float64}((Bim_x, Bim_y))
end

function ComputeFluxDensity(dh::DofHandler, cv::CV, u::AbstractVector{T}, problem::Problem{T}, cellparams::Vector{CellParams}) where {T,CV<:NamedTuple}
    Be = zeros(Ferrite.Vec{2,T}, getncells(dh.grid), get_nquadpoints(dh, cv))
    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        ComputeFluxDensity!(Be, sdh, cv_, u, problem, cellparams)
    end

    return Be
end

function ComputeFluxDensity!(Be, sdh::SubDofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem{T}, cellparams::Vector{CellParams}) where {T}
    # Allocate temporary storage
    drange = dof_range(sdh, :A)
    ue = zeros(T, length(drange))

    for cell ∈ CellIterator(sdh)
        cell_num = cellid(cell)
        reinit!(cv, cell)

        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]]
        end
        xe = getcoordinates(sdh.dh.grid, cell_num)

        for q_point ∈ 1:getnquadpoints(cv)
            ∇uq = function_gradient(cv, q_point, ue)
            xq = spatial_coordinate(cv, q_point, xe)
            Bre_q, Bim_q = ComputeFluxDensity(∇uq, xq, problem.symmetry)

            Be[cell_num, q_point] = Bre_q + 1im * Bim_q
        end
    end
end

function ComputeCurrentDensity(dh::DofHandler, cv::CV, u::AbstractVector{T}, problem::Problem{T}, cellparams::Vector{CellParams}) where {T,CV<:NamedTuple}
    # Allocate temporary storage
    Je = zeros(T, getncells(dh.grid), get_nquadpoints(dh, cv))

    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        ComputeCurrentDensity!(Je, sdh, cv_, u, problem, cellparams)
    end

    return Je
end

function ComputeCurrentDensity!(Je, sdh::SubDofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem{T}, cellparams::Vector{CellParams}) where {T}
    ω = get_frequency(problem)

    drange = dof_range(sdh, :A)
    ue = zeros(T, length(drange))

    for cell ∈ CellIterator(sdh)
        cell_num = cellid(cell)
        reinit!(cv, cell)

        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]] # TODO eddy currents in axisymmetric model
        end

        param = cellparams[cell_num]

        for q_point ∈ 1:getnquadpoints(cv)
            uq = function_value(cv, q_point, ue)

            Je[cell_num, q_point] += param.J0 - 1im * param.σ * ω * uq
        end
    end
end

function ComputeCurrentDensity(dh::DofHandler, cv::CV, ch::CircuitHandler, u::AbstractVector{T}, problem::Problem{T}, cellparams::Vector{CellParams}) where {T,CV<:NamedTuple}
    Je = ComputeCurrentDensity(dh, cv, u, problem, cellparams)

    for (p, conductor) ∈ enumerate(ch.cond_str)
        coupling_idx = ndofs(dh) + p
        domain_set = getcellset(dh.grid, conductor.domain)

        for cell_num ∈ domain_set
            Je[cell_num, :] .= conductor.N * u[coupling_idx] / conductor.area
        end
    end

    for (q, conductor) ∈ enumerate(ch.cond_sol)
        coupling_idx = ndofs(dh) + ncond_str(ch) + q
        domain_set = getcellset(dh.grid, conductor.domain)

        Gdc = ComputeDCConductance(dh, cv, problem, cellparams, conductor.domain)
        for cell_num ∈ domain_set
            Je[cell_num, :] .+= Gdc * u[coupling_idx] / conductor.area
        end
    end

    return Je
end

function ComputeDCResistance(dh::DofHandler, cv::CV, problem::Problem, cellparams::Vector{CellParams}, domain::String) where {CV<:NamedTuple}
    return 1 / ComputeDCConductance(dh, cv, problem, cellparams, domain)
end

function ComputeDCConductance(dh::DofHandler, cv::CV, problem::Problem, cellparams::Vector{CellParams}, domain::String) where {CV<:NamedTuple}
    Gdc = 0
    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        Gdc += ComputeDCConductance(sdh, cv_, problem.symmetry, cellparams, domain)
    end

    return Gdc
end

function ComputeDCConductance(sdh::SubDofHandler, cv::CellValues, symmetry::Symmetry2D, cellparams::Vector{CellParams}, domain::String)
    Gdc = 0

    domain_set = getcellset(sdh.dh.grid, domain)
    for cell ∈ CellIterator(sdh)
        cell_id = cellid(cell)
        if (cell_id ∉ domain_set)
            continue
        end
        reinit!(cv, cell)

        param = cellparams[cell_id]
        x = getcoordinates(sdh.dh.grid, cell_id)

        for q_point ∈ 1:getnquadpoints(cv)
            coord = spatial_coordinate(cv, q_point, x)
            ℓe = get_modeldepth(symmetry, coord[1])

            dΩ = getdetJdV(cv, q_point)
            Gdc += param.σ / ℓe * dΩ
        end
    end

    return Gdc
end

function ComputeLossDensity(dh::DofHandler, cv::CV, J::AbstractMatrix{T}, B::AbstractMatrix{U}, problem::Problem{T}, cellparams::Vector{CellParams}) where {T,U,CV<:NamedTuple}
    n_quadpts = get_nquadpoints(dh, cv)
    S_cell = zeros(Complex{Float64}, getncells(dh.grid), n_quadpts)
    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        ComputeLossDensity!(S_cell, sdh, cv_, J, B, problem, cellparams)
    end

    return S_cell
end

function ComputeLossDensity!(S_cell, sdh::SubDofHandler, cv::CellValues, J::AbstractMatrix{T}, B::AbstractMatrix{U}, problem::Problem{T}, cellparams::Vector{CellParams}) where {T,U}
    ω = get_frequency(problem)

    for cell ∈ CellIterator(sdh)
        cell_num = cellid(cell)
        param = cellparams[cell_num]

        for q_point ∈ 1:getnquadpoints(cv)
            Je_q = J[cell_num, q_point]
            Be_q = B[cell_num, q_point]

            sm = 0.5im * ω * Be_q ⋅ Vec{2}(conj(param.ν ⋅ Be_q))
            if (norm(Je_q) > 0)
                se = norm(Je_q)^2 / (2 * param.σ)
            else
                se = 0
            end

            S_cell[cell_num, q_point] += se + sm
        end
    end
end

function ComputeLoss(dh::DofHandler, cv::CV, ch::CircuitHandler, J::AbstractMatrix{T}, B::AbstractMatrix{U}, problem::Problem{T}, cellparams::Vector{CellParams}) where {T,U,CV<:NamedTuple}
    # Result storage
    I_circ = Dict()
    S_circ = Dict()
    R_circ = Dict()

    # Calculate the complex loss density for each cell
    S_cell = ComputeLossDensity(dh, cv, J, B, problem, cellparams)

    # Calculate the loss and current for each circuit
    coupling = vcat(ch.cond_str, ch.cond_sol)
    for (i, coupling) ∈ enumerate(coupling)
        domain_set = getcellset(dh.grid, coupling.domain)
        I_ = zero(T)
        S_ = zero(T)

        for sdh ∈ dh.subdofhandlers
            cv_ = get_cellvalues(cv, getcelltype(sdh))
            for cell ∈ CellIterator(sdh)
                cell_num = cellid(cell)
                if (cell_num ∉ domain_set)
                    continue
                end
                reinit!(cv_, cell)

                xe = getcoordinates(dh.grid, cell_num)

                for q_point ∈ 1:getnquadpoints(cv_)
                    dΩ = getdetJdV(cv_, q_point)
                    xq = spatial_coordinate(cv_, q_point, xe)

                    depth = get_modeldepth(problem.symmetry, xq)

                    Je = J[cell_num, q_point]
                    Se = S_cell[cell_num, q_point]

                    I_ += Je * dΩ
                    S_ += Se * depth * dΩ
                end
            end
        end

        I_circ[coupling.domain] = I_
        S_circ[coupling.domain] = S_
        R_circ[coupling.domain] = real(2 * S_ / norm(I_)^2)
    end

    return (I_circ, S_circ, R_circ)
end