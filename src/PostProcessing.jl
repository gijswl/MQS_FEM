function ComputeFluxDensity(cellvalues::CellValues, dh::DofHandler, u::AbstractVector{T}) where {T}
    n_basefuncs = getnbasefunctions(cellvalues)
    n_quadpts = getnquadpoints(cellvalues)
    cell_dofs = zeros(Int, n_basefuncs)

    # Allocate storage for the flux density vectors
    B = zeros(Ferrite.Vec{3,Complex{Float64}}, getncells(dh.grid))

    for (cell_num, cell) ∈ enumerate(CellIterator(dh))
        celldofs!(cell_dofs, dh, cell_num)
        reinit!(cellvalues, cell)

        ue = u[cell_dofs]
        Be = zero(Ferrite.Vec{3,Complex{Float64}})

        cell_area = 0

        for q_point ∈ 1:n_quadpts
            curl_uq = function_curl(cellvalues, q_point, ue)
            
            dΩ = getdetJdV(cellvalues, q_point)
            cell_area += dΩ

            Be += curl_uq * dΩ
        end

        B[cell_num] += Be / cell_area
    end

    return B
end

function ComputeCurrentDensity(cellvalues::CellValues, dh::DofHandler, u::AbstractVector{T}) where {T}
    n_basefuncs = getnbasefunctions(cellvalues)
    n_quadpts = getnquadpoints(cellvalues)
    cell_dofs = zeros(Int, n_basefuncs)

    # Allocate storage for the current density vectors
    Jsource = zeros(Vec{3,Complex{Float64}}, getncells(dh.grid))
    Jeddy = zeros(Vec{3,Complex{Float64}}, getncells(dh.grid))

    for (cell_num, cell) ∈ enumerate(CellIterator(dh))
        celldofs!(cell_dofs, dh, cell_num)
        reinit!(cellvalues, cell)

        ue = u[cell_dofs]
        σe = σ[cell_num]

        AvgAz = zero(Vec{3,Complex{Float64}})
        cell_area = 0

        for q_point ∈ 1:n_quadpts
            uq = function_value(cellvalues, q_point, ue)
            dΩ = getdetJdV(cellvalues, q_point)

            AvgAz += uq * dΩ
            cell_area += dΩ
        end
        AvgAz /= cell_area

        Jeddy[cell_num] += -1im * σe * ω * AvgAz
        Jsource[cell_num] += J0[cell_num]
    end

    return Jsource + Jeddy
end