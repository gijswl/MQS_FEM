function ComputeFluxDensity(∇Aq::Vec{2,T}, ::Vec{2}, ::Problem2D{T}, ::Planar2D) where {T}
    ∇Aq_re = real(∇Aq)
    ∇Aq_im = imag(∇Aq)

    Bre_x = ∇Aq_re[2]
    Bre_y = -∇Aq_re[1]
    Bim_x = ∇Aq_im[2]
    Bim_y = -∇Aq_im[1]

    return (Bre_x, Bre_y), (Bim_x, Bim_y)
end

# function ComputeFluxDensity(∇uq::Vec{2,<:Complex}, xq::Vec{2,<:Real}, ::SimParamsAxi)
#     ∇Aq = ∇uq / xq[1]
#     ∇Aq_re = real(∇Aq)
#     ∇Aq_im = imag(∇Aq)

#     Bre_ρ = -∇Aq_re[2]
#     Bre_z = ∇Aq_re[1]
#     Bim_ρ = -∇Aq_im[2]
#     Bim_z = ∇Aq_im[1]

#     return (Bre_ρ, Bre_z), (Bim_ρ, Bim_z)
# end

# """
#     ComputeFluxDensity(cv::CellValues, dh::DofHandler, projector::L2Projector, qr::QuadratureRule, u::AbstractVector{T}, params::SimParams) where {T}

# Compute the nodal flux density vectors (Bx, By) or (Bρ, Bz). 
# Separately returns the real and imaginary parts.
# """
# function ComputeFluxDensity(cv::CellValues, dh::DofHandler, projector::L2Projector, qr::QuadratureRule, u::AbstractVector{T}, params::SimParams) where {T}
#     n_basefuncs = getnbasefunctions(cv)
#     n_quadpts = getnquadpoints(cv)
#     cell_dofs = zeros(Int, n_basefuncs)

#     # Allocate storage for the flux density vectors
#     Bre = [Ferrite.Vec{2,Float64}[] for _ ∈ 1:getncells(dh.grid)]
#     Bim = [Ferrite.Vec{2,Float64}[] for _ ∈ 1:getncells(dh.grid)]

#     for (cell_num, cell) ∈ enumerate(CellIterator(dh))
#         celldofs!(cell_dofs, dh, cell_num)
#         reinit!(cv, cell)

#         ue = u[cell_dofs]
#         Bre_cell = Bre[cell_num]
#         Bim_cell = Bim[cell_num]

#         xe = getcoordinates(dh.grid, cell_num)

#         for q_point ∈ 1:n_quadpts
#             ∇uq = function_gradient(cv, q_point, ue)
#             xq = spatial_coordinate(cv, q_point, xe)
#             Bre_q, Bim_q = ComputeFluxDensity(∇uq, xq, params)

#             push!(Bre_cell, Ferrite.Vec{2,Float64}(Bre_q))
#             push!(Bim_cell, Ferrite.Vec{2,Float64}(Bim_q))
#         end
#     end

#     Bre_proj = project(projector, Bre, qr)
#     Bim_proj = project(projector, Bim, qr)

#     return Bre_proj, Bim_proj
# end

function ComputeFluxDensity(dh::DofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem2D{T}, cellparams::CellParams) where {T}
    n_basefuncs = getnbasefunctions(cv)
    n_quadpts = getnquadpoints(cv)
    cell_dofs = zeros(Int, n_basefuncs)

    # Allocate storage for the flux density vectors
    Bre = zeros(Ferrite.Vec{2,Float64}, getncells(dh.grid), n_quadpts)
    Bim = zeros(Ferrite.Vec{2,Float64}, getncells(dh.grid), n_quadpts)

    for (cell_num, cell) ∈ enumerate(CellIterator(dh))
        celldofs!(cell_dofs, dh, cell_num)
        reinit!(cv, cell)

        ue = u[cell_dofs]
        xe = getcoordinates(dh.grid, cell_num)

        for q_point ∈ 1:n_quadpts
            ∇uq = function_gradient(cv, q_point, ue)
            xq = spatial_coordinate(cv, q_point, xe)
            Bre_q, Bim_q = ComputeFluxDensity(∇uq, xq, problem, problem.symmetry)

            Bre[cell_num, q_point] = Ferrite.Vec{2,Float64}(Bre_q)
            Bim[cell_num, q_point] = Ferrite.Vec{2,Float64}(Bim_q)
        end
    end

    return Bre, Bim
end

function ComputeCurrentDensity(dh::DofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem2D{T}, cellparams::CellParams) where {T}
    n_basefuncs = getnbasefunctions(cv)
    n_quadpts = getnquadpoints(cv)
    cell_dofs = zeros(Int, n_basefuncs)

    ω = get_frequency(problem)

    # Allocate storage for the current density vectors
    Ncell = getncells(dh.grid)
    Jsource = zeros(T, Ncell)
    Jeddy = zeros(T, Ncell)

    for (cell_num, cell) ∈ enumerate(CellIterator(dh))
        celldofs!(cell_dofs, dh, cell_num)
        reinit!(cv, cell)

        ue = u[cell_dofs] # TODO eddy currents in axisymmetric model
        σe = cellparams.σ[cell_num]

        AvgAz = zero(T)
        cell_area = 0

        for q_point ∈ 1:n_quadpts
            uq = function_value(cv, q_point, ue)
            dΩ = getdetJdV(cv, q_point)

            AvgAz += uq * dΩ
            cell_area += dΩ
        end
        AvgAz /= cell_area

        Jeddy[cell_num] += -1im * σe * ω * AvgAz
        Jsource[cell_num] += cellparams.J0[cell_num]
    end

    return Jsource + Jeddy
end

function ComputeCurrentDensity(dh::DofHandler, cv::CellValues, ch::CircuitHandler, u::AbstractVector{T}, problem::Problem2D{T}, cellparams::CellParams) where {T}
    n_basefuncs = getnbasefunctions(cv)
    n_quadpts = getnquadpoints(cv)
    cell_dofs = zeros(Int, n_basefuncs)

    ω = get_frequency(problem)

    # Allocate storage for the current density vectors
    Ncell = getncells(dh.grid)
    Jsource = zeros(T, Ncell, n_quadpts)
    Jeddy = zeros(T, Ncell, n_quadpts)

    for (cell_num, cell) ∈ enumerate(CellIterator(dh))
        celldofs!(cell_dofs, dh, cell_num)
        reinit!(cv, cell)

        ue = u[cell_dofs] # TODO eddy currents in axisymmetric model
        σe = cellparams.σ[cell_num]

        for q_point ∈ 1:n_quadpts
            uq = function_value(cv, q_point, ue)

            Jeddy[cell_num, q_point] += -1im * σe * ω * uq
            Jsource[cell_num, q_point] += cellparams.J0[cell_num]
        end
    end

    for (i, coupling) ∈ enumerate(ch.coupling)
        coupling_idx = ndofs(dh) + i

        for cell ∈ CellIterator(dh, getcellset(dh.grid, coupling.domain))
            cell_num = cellid(cell)

            Jsource[cell_num, :] .+= u[coupling_idx] / (coupling.symm_factor * coupling.area)
        end
    end

    return Jsource + Jeddy
end

function ComputeLossDensity(dh::DofHandler, cv::CellValues, J::AbstractMatrix{T}, Bre::AbstractMatrix{U}, Bim::AbstractMatrix{U}, problem::Problem2D{T}, cellparams::CellParams) where {T,U}
    ω = get_frequency(problem)

    n_quadpts = getnquadpoints(cv)
    S_cell = zeros(Complex{Float64}, getncells(dh.grid), n_quadpts)

    for cell ∈ CellIterator(dh)
        cell_num = cellid(cell)
        σe = cellparams.σ[cell_num]
        νe = cellparams.ν[cell_num]

        for q_point ∈ 1:n_quadpts
            Je_q = J[cell_num, q_point]
            Bre_q = Bre[cell_num, q_point]
            Bim_q = Bim[cell_num, q_point]

            B_q = Vec{2,T}((Bre_q[1] + 1im * Bim_q[1], Bre_q[2] + 1im * Bim_q[2]))

            sm = 0.5im * ω * B_q ⋅ Vec{2}(conj(νe ⋅ B_q))
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

function ComputeLoss(dh::DofHandler, cv::CellValues, ch::CircuitHandler, J::AbstractMatrix{T}, Bre::AbstractMatrix{U}, Bim::AbstractMatrix{U}, problem::Problem2D{T}, cellparams::CellParams) where {T,U}
    n_quadpts = getnquadpoints(cv)

    # Result storage
    ## Cell quantities
    S_cell = zeros(Complex{Float64}, getncells(dh.grid))

    ## Circuit quantites
    I_circ = zeros(Complex{Float64}, length(ch.coupling))
    S_circ = zeros(Complex{Float64}, length(ch.coupling))
    R_circ = zeros(Float64, length(ch.coupling))

    # Calculate the complex loss density for each cell
    S_cell = ComputeLossDensity(dh, cv, J, Bre, Bim, problem, cellparams)

    # Calculate the loss and current for each circuit
    for (i, coupling) ∈ enumerate(ch.coupling)
        for cell ∈ CellIterator(dh, getcellset(dh.grid, coupling.domain))
            reinit!(cv, cell)
            cell_num = cellid(cell)

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

    return (S_cell, I_circ, S_circ, R_circ)
end