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
    Bre = zeros(Ferrite.Vec{2,Float64}, getncells(dh.grid))
    Bim = zeros(Ferrite.Vec{2,Float64}, getncells(dh.grid))

    for (cell_num, cell) ∈ enumerate(CellIterator(dh))
        celldofs!(cell_dofs, dh, cell_num)
        reinit!(cv, cell)

        ue = u[cell_dofs]
        xe = getcoordinates(dh.grid, cell_num)

        Bre_e = Ferrite.Vec{2,Float64}([0, 0])
        Bim_e = Ferrite.Vec{2,Float64}([0, 0])
        cell_area = 0

        for q_point ∈ 1:n_quadpts
            ∇uq = function_gradient(cv, q_point, ue)
            xq = spatial_coordinate(cv, q_point, xe)
            Bre_q, Bim_q = ComputeFluxDensity(∇uq, xq, problem, problem.symmetry)

            dΩ = getdetJdV(cv, q_point)
            cell_area += dΩ

            Bre_e += Ferrite.Vec{2,Float64}(Bre_q) * dΩ
            Bim_e += Ferrite.Vec{2,Float64}(Bim_q) * dΩ
        end

        Bre[cell_num] += Bre_e / cell_area
        Bim[cell_num] += Bim_e / cell_area
    end

    return Bre, Bim
end

function ComputeCurrentDensity(dh::DofHandler, cv::CellValues, u::AbstractVector{T}, problem::Problem2D{T}, cellparams::CellParams) where {T}
    n_basefuncs = getnbasefunctions(cv)
    n_quadpts = getnquadpoints(cv)
    cell_dofs = zeros(Int, n_basefuncs)

    if(typeof(problem.time) <: TimeHarmonic)
        ω = problem.time.ω
    else
        ω = 0
    end

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

function ComputeLossDensity(dh::DofHandler, cv::CellValues, u::AbstractVector{T}, J::AbstractVector{T}, Bre::AbstractVector{U}, Bim::AbstractVector{U}, problem::Problem2D{T}, cellparams::CellParams) where {T, U}
    if(typeof(problem.time) <: TimeHarmonic)
        ω = problem.time.ω
    else
        ω = 0
    end

    σ = cellparams.σ
    ν = cellparams.ν

    S_cell = zeros(Complex{Float64}, getncells(dh.grid))

    for cell ∈ CellIterator(dh)
        cell_num = cellid(cell)
        Je = J[cell_num]
        σe = σ[cell_num]
        νe = ν[cell_num]

        Bre_e = Bre[cell_num]
        Bim_e = Bim[cell_num]

        B_e = Vec{2, T}((Bre_e[1] + 1im * Bim_e[1], Bre_e[2] + 1im * Bim_e[2]))

        sm = 0.5im * ω * B_e ⋅ Vec{2}(conj(νe ⋅ B_e))
        if (norm(Je) > 0)
            se = norm(Je)^2 / (2 * σe)
        else
            se = 0
        end

        S_cell[cell_num] += se + sm
    end

    return S_cell
end