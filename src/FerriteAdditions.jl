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