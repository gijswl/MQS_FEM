function ComputeFluxDensity(∇Aq::Vec{2,T}, ::Vec{2}, ::Problem2D{T}, ::Planar2D) where {T}
    ∇Aq_re = real(∇Aq)
    ∇Aq_im = imag(∇Aq)

    Bre_x = ∇Aq_re[2]
    Bre_y = -∇Aq_re[1]
    Bim_x = ∇Aq_im[2]
    Bim_y = -∇Aq_im[1]

    return Ferrite.Vec{2,Float64}((Bre_x, Bre_y)), Ferrite.Vec{2,Float64}((Bim_x, Bim_y))
end

function ComputeFluxDensity(dh::DofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem2D{T}, cellparams::CellParams) where {T}
    n_quadpts = getnquadpoints(cv)

    # Allocate temporary storage
    drange = dof_range(dh, :A)
    ue = zeros(T, length(drange))
    Be = zeros(Ferrite.Vec{2,T}, getncells(dh.grid), n_quadpts)

    for cell ∈ CellIterator(dh)
        cell_num = cellid(cell)
        reinit!(cv, cell)

        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]]
        end
        xe = getcoordinates(dh.grid, cell_num)

        for q_point ∈ 1:n_quadpts
            ∇uq = function_gradient(cv, q_point, ue)
            xq = spatial_coordinate(cv, q_point, xe)
            Bre_q, Bim_q = ComputeFluxDensity(∇uq, xq, problem, problem.symmetry)

            Be[cell_num, q_point] = Bre_q + 1im * Bim_q
        end
    end

    return Be
end

function ComputeCurrentDensity(dh::DofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem2D{T}, cellparams::CellParams) where {T}
    n_quadpts = getnquadpoints(cv)

    ω = get_frequency(problem)

    # Allocate temporary storage
    drange = dof_range(dh, :A)
    ue = zeros(T, length(drange))
    Je = zeros(T, getncells(dh.grid), n_quadpts)

    for cell ∈ CellIterator(dh)
        cell_num = cellid(cell)
        reinit!(cv, cell)

        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]] # TODO eddy currents in axisymmetric model
        end
        σe = cellparams.σ[cell_num]
        J0e = cellparams.J0[cell_num]

        for q_point ∈ 1:n_quadpts
            uq = function_value(cv, q_point, ue)

            Je[cell_num, q_point] += J0e - 1im * σe * ω * uq
        end
    end

    return Je
end

function ComputeCurrentDensity(dh::DofHandler, cv::CellValues, ch::CircuitHandler, u::AbstractVector{T}, problem::Problem2D{T}, cellparams::CellParams) where {T}
    Je = ComputeCurrentDensity(dh, cv, u, problem, cellparams)

    for (i, coupling) ∈ enumerate(ch.coupling)
        coupling_idx = ndofs(dh) + i

        for cell ∈ CellIterator(dh, getcellset(dh.grid, coupling.domain))
            cell_num = cellid(cell)

            Je[cell_num, :] .+= u[coupling_idx] / (coupling.symm_factor * coupling.area)
        end
    end

    return Je
end

function ComputeLossDensity(dh::DofHandler, cv::CellValues, J::AbstractMatrix{T}, B::AbstractMatrix{U}, problem::Problem2D{T}, cellparams::CellParams) where {T,U}
    ω = get_frequency(problem)

    n_quadpts = getnquadpoints(cv)
    S_cell = zeros(Complex{Float64}, getncells(dh.grid), n_quadpts)

    for cell ∈ CellIterator(dh)
        cell_num = cellid(cell)
        σe = cellparams.σ[cell_num]
        νe = cellparams.ν[cell_num]

        for q_point ∈ 1:n_quadpts
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

    return S_cell
end

function ComputeLoss(dh::DofHandler, cv::CellValues, ch::CircuitHandler, J::AbstractMatrix{T}, B::AbstractMatrix{U}, problem::Problem2D{T}, cellparams::CellParams) where {T,U}
    n_quadpts = getnquadpoints(cv)

    # Result storage
    ## Cell quantities
    S_cell = zeros(Complex{Float64}, getncells(dh.grid))

    ## Circuit quantites
    I_circ = zeros(Complex{Float64}, length(ch.coupling))
    S_circ = zeros(Complex{Float64}, length(ch.coupling))
    R_circ = zeros(Float64, length(ch.coupling))

    # Calculate the complex loss density for each cell
    S_cell = ComputeLossDensity(dh, cv, J, B, problem, cellparams)

    # Calculate the loss and current for each circuit
    for (i, coupling) ∈ enumerate(ch.coupling)
        for cell ∈ CellIterator(dh, getcellset(dh.grid, coupling.domain))
            cell_num = cellid(cell)
            reinit!(cv, cell)

            xe = getcoordinates(dh.grid, cell_num)

            for q_point ∈ 1:n_quadpts
                dΩ = getdetJdV(cv, q_point)
                xq = spatial_coordinate(cv, q_point, xe)

                depth = get_modeldepth(problem, problem.symmetry, xq)

                Je = J[cell_num, q_point]
                Se = S_cell[cell_num, q_point]

                I_circ[i] += Je * dΩ
                S_circ[i] += Se * depth * dΩ
            end
        end

        I_circ[i] = I_circ[i] / coupling.symm_factor
        S_circ[i] = S_circ[i] / coupling.symm_factor
        R_circ[i] = real(2 * S_circ[i] / norm(I_circ[i])^2)
    end

    return (I_circ, S_circ, R_circ)
end