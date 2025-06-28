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

function ComputeFluxDensity(dh::DofHandler, cv::CV, u::AbstractVector{T}, problem::Problem{T}, cellparams::CellParams) where {T,CV<:NamedTuple}
    Be = zeros(Ferrite.Vec{2,T}, getncells(dh.grid), get_nquadpoints(dh, cv))
    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        ComputeFluxDensity!(Be, sdh, cv_, u, problem, cellparams)
    end

    return Be
end

function ComputeFluxDensity!(Be, sdh::SubDofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem{T}, cellparams::CellParams) where {T}
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

function ComputeCurrentDensity(dh::DofHandler, cv::CV, u::AbstractVector{T}, problem::Problem{T}, cellparams::CellParams) where {T,CV<:NamedTuple}
    # Allocate temporary storage
    Je = zeros(T, getncells(dh.grid), get_nquadpoints(dh, cv))

    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        ComputeCurrentDensity!(Je, sdh, cv_, u, problem, cellparams)
    end

    return Je
end

function ComputeCurrentDensity!(Je, sdh::SubDofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem{T}, cellparams::CellParams) where {T}
    ω = get_frequency(problem)

    drange = dof_range(sdh, :A)
    ue = zeros(T, length(drange))

    for cell ∈ CellIterator(sdh)
        cell_num = cellid(cell)
        reinit!(cv, cell)

        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]] # TODO eddy currents in axisymmetric model
        end
        σe = cellparams.σ[cell_num]
        J0e = cellparams.J0[cell_num]

        for q_point ∈ 1:getnquadpoints(cv)
            uq = function_value(cv, q_point, ue)

            Je[cell_num, q_point] += J0e - 1im * σe * ω * uq
        end
    end
end

function ComputeCurrentDensity(dh::DofHandler, cv::CV, ch::CircuitHandler, u::AbstractVector{T}, problem::Problem{T}, cellparams::CellParams) where {T,CV<:NamedTuple}
    Je = ComputeCurrentDensity(dh, cv, u, problem, cellparams)

    for (i, coupling) ∈ enumerate(ch.coupling)
        coupling_idx = ndofs(dh) + i
        domain_set = getcellset(dh.grid, coupling.domain)

        for sdh ∈ dh.subdofhandlers
            for cell ∈ CellIterator(sdh)
                cell_num = cellid(cell)
                if (cell_num ∉ domain_set)
                    continue
                end

                Je[cell_num, :] .+= u[coupling_idx] / (coupling.symm_factor * coupling.area)
            end
        end
    end

    return Je
end

function ComputeLossDensity(dh::DofHandler, cv::CV, J::AbstractMatrix{T}, B::AbstractMatrix{U}, problem::Problem{T}, cellparams::CellParams) where {T,U,CV<:NamedTuple}
    n_quadpts = get_nquadpoints(dh, cv)
    S_cell = zeros(Complex{Float64}, getncells(dh.grid), n_quadpts)
    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        ComputeLossDensity!(S_cell, sdh, cv_, J, B, problem, cellparams)
    end

    return S_cell
end

function ComputeLossDensity!(S_cell, sdh::SubDofHandler, cv::CellValues, J::AbstractMatrix{T}, B::AbstractMatrix{U}, problem::Problem{T}, cellparams::CellParams) where {T,U}
    ω = get_frequency(problem)

    for cell ∈ CellIterator(sdh)
        cell_num = cellid(cell)
        σe = cellparams.σ[cell_num]
        νe = cellparams.ν[cell_num]

        for q_point ∈ 1:getnquadpoints(cv)
            Je_q = J[cell_num, q_point]
            Be_q = B[cell_num, q_point]

            sm = 0.5im * ω * Be_q ⋅ Vec{2}(conj(νe ⋅ Be_q))
            if (norm(Je_q) > 0)
                se = norm(Je_q)^2 / (2 * σe)
            else
                se = 0
            end

            S_cell[cell_num, q_point] += se + sm
        end
    end
end

function ComputeLoss(dh::DofHandler, cv::CV, ch::CircuitHandler, J::AbstractMatrix{T}, B::AbstractMatrix{U}, problem::Problem{T}, cellparams::CellParams) where {T,U,CV<:NamedTuple}
    # Result storage
    ## Cell quantities
    S_cell = zeros(Complex{Float64}, getncells(dh.grid))

    ## Circuit quantites
    I_circ = zeros(Complex{Float64}, length(ch.coupling))
    S_circ = zeros(Complex{Float64}, length(ch.coupling))
    R_circ = zeros(Float64, length(ch.coupling))
    A_circ = zeros(Float64, length(ch.coupling))

    # Calculate the complex loss density for each cell
    S_cell = ComputeLossDensity(dh, cv, J, B, problem, cellparams)

    # Calculate the loss and current for each circuit
    for (i, coupling) ∈ enumerate(ch.coupling)
        domain_set = getcellset(dh.grid, coupling.domain)

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

                    depth = get_modeldepth(problem, problem.symmetry, xq)

                    Je = J[cell_num, q_point]
                    Se = S_cell[cell_num, q_point]

                    I_circ[i] += Je * dΩ
                    S_circ[i] += Se * depth * dΩ
                    A_circ[i] += dΩ
                end
            end
        end

        I_circ[i] = I_circ[i] / coupling.symm_factor
        S_circ[i] = S_circ[i] / coupling.symm_factor
        R_circ[i] = real(2 * S_circ[i] / norm(I_circ[i])^2)
    end

    return (I_circ, S_circ, R_circ, A_circ)
end