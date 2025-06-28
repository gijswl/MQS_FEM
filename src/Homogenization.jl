"""
    ComputeHomogenization(cv::CellValues, dh::DofHandler, domain::String, J::AbstractVector{T}, B::AbstractVector{U}, S::AbstractVector{T}, problem::Problem{T}) where {T,U}

Compute the homogenized reluctivity `ν` (in the x and y directions) and conductivity `σ` for the homogenization `domain`.
Depending on which simulation is being used as input, one or more of the parameters may not be relevant.
"""
function ComputeHomogenization(dh::DofHandler, cv::CV, domain::String, J::AbstractMatrix{T}, B::AbstractMatrix{U}, S::AbstractMatrix{T}, problem::Problem{T}) where {T,U,CV<:NamedTuple}
    ω = get_frequency(problem)
    depth = get_modeldepth(problem, problem.symmetry, 0)

    # Result storage
    AvgB = Ferrite.Vec{2,Complex{Float64}}([0, 0])
    Itot = 0 + 0im
    Stot = 0 + 0im
    domain_area = 0

    domain_set = getcellset(dh.grid, domain)

    # Loop over the cells in the domain for numerical integration
    for sdh ∈ dh.subdofhandlers
        cv_ = get_cellvalues(cv, getcelltype(sdh))
        for cell ∈ CellIterator(sdh)
            cell_num = cellid(cell)
            if (cell_num ∉ domain_set)
                continue
            end

            reinit!(cv_, cell)

            for q_point ∈ 1:getnquadpoints(cv_)
                dΩ = getdetJdV(cv_, q_point)

                Je = J[cell_num, q_point]
                Se = S[cell_num, q_point]
                Be = B[cell_num, q_point]

                # Integrate dΩ to obtain the domain area: Ω = ∫ dΩ
                domain_area += dΩ

                # Integrate the quantities of interest over the domain
                AvgB += Be * dΩ # Averaged flux density AvB = 1/Ω * ∫ B dΩ
                Itot += Je * dΩ # Total current I = ∫ J dΩ
                Stot += Se * dΩ # Complex loss S = ∫ s dΩ
            end
        end
    end

    AvgB = AvgB / domain_area

    # Homogenized reluctivity ν and conductivity σ 
    νx = conj(2 * Stot / (1im * ω * norm(AvgB[1])^2 * depth * domain_area))
    νy = conj(2 * Stot / (1im * ω * norm(AvgB[2])^2 * depth * domain_area))
    σz = depth * norm(Itot)^2 / (2 * Stot * domain_area)

    return (νx, νy, σz)
end