# RefTetrahedron, 1st order Lagrange
# https://defelement.org/elements/examples/tetrahedron-nedelec1-lagrange-1.html
function Ferrite.reference_shape_value(ip::Nedelec{RefTetrahedron,1}, ξ::Vec{3}, i::Int)
    x, y, z = ξ

    i == 1 && return Vec(1 - y - z, x, x)
    i == 2 && return Vec(-y, x, zero(x)) # DefElement φ2
    i == 3 && return Vec(-y, x + z - 1, -y) # DefElement -φ4
    i == 4 && return Vec(z, z, 1 - x - y) # DefElement φ3
    i == 5 && return Vec(-z, zero(x), x) # DefElement φ1
    i == 6 && return Vec(zero(x), -z, y) # DefElement φ0
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

Ferrite.getnbasefunctions(::Nedelec{RefTetrahedron,1}) = 6
Ferrite.edgedof_interior_indices(::Nedelec{RefTetrahedron,1}) = ((1,), (2,), (3,), (4,), (5,), (6,))
Ferrite.facedof_indices(::Nedelec{RefTetrahedron,1}) = ((1, 2, 3), (1, 4, 5), (2, 5, 6), (3, 4, 6))
Ferrite.adjust_dofs_during_distribution(::Nedelec{RefTetrahedron,1}) = false

function Ferrite.get_direction(::Nedelec{RefTetrahedron,1}, j, cell)
    edge = Ferrite.edges(cell)[j]
    return ifelse(edge[2] > edge[1], 1, -1)
end



function Ferrite._evaluate_at_grid_nodes!(
    data::Union{Vector,Matrix}, sdh::SubDofHandler,
    u::AbstractVector{T}, cv::CellValues, drange::UnitRange
) where {T}
    ue = zeros(T, length(drange))
    for cell in CellIterator(sdh)
        reinit!(cv, cell)
        @assert getnquadpoints(cv) == length(cell.nodes)
        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]]
        end
        for (qp, nodeid) in pairs(cell.nodes)
            val = function_value(cv, qp, ue)
            if data isa Matrix # VTK
                data[1:length(val), nodeid] .= val
                data[(length(val)+1):end, nodeid] .= 0 # purge the NaN
            else
                data[nodeid] = val
            end
        end
    end
    return data
end