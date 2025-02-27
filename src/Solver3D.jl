
function init_problem(problem::Problem3D, grid::Grid{3})
    refshape = getrefshape(grid.cells[1])

    fe_order = problem.fe_order
    qr_order = problem.qr_order

    dh = DofHandler(grid)

    formulation = typeof(problem.formulation)
    if (formulation <: FormulationA)
        ip_fe = Nedelec{refshape,fe_order}()
        ip_geo = Lagrange{refshape,1}()
        qr = QuadratureRule{refshape}(qr_order)
        cv = CellValues(qr, ip_fe, ip_geo)

        add!(dh, :A, ip_fe)
    else
        error("Unknown 3D problem formulation $(formulation)")
    end
    close!(dh)

    return cv, dh
end