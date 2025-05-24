using Gmsh

function CalculateHexPacking(d::Real, λ::Real)
    A = π / 4 * d^2
    d0 = √(2 * A / (λ * √(3)))

    return d0
end

function CalculateRectPacking(d::Real, λ::Real)
    if (λ > π / 4)
        error("Packing factor for rectangular packing too high: $λ > π/4.")
    end
    d0 = √(π / 4 * d^2 / λ)
    h = d0 - d

    return (h, h)
end

function DrawCircle(geo, x0::Tuple{Real,Real}, r::Real, mesh_density::Real)
    p1 = geo.addPoint(x0[1], x0[2], 0, 1)
    p2 = geo.addPoint(x0[1] + r, x0[2], 0, mesh_density)
    p3 = geo.addPoint(x0[1] - r, x0[2], 0, mesh_density)

    l1 = geo.addCircleArc(p2, p1, p3)
    l2 = geo.addCircleArc(p3, p1, p2)

    cl = geo.addCurveLoop([l1, l2])

    return (cl, [l1, l2])
end

function GenerateGeometryRect(d::Real, λ::Real, name::String; path="examples/geo/")
    (v, h) = CalculateRectPacking(d, λ)

    GenerateGeometryRect(d, v, h, name; path)
end

function GenerateGeometryRect(d::Real, v::Real, h::Real, name::String; path="examples/geo/")
    dx = d + h
    dy = d + v

    gmsh.initialize()

    model = gmsh.model
    geo = model.geo
    mesh = model.mesh

    mshd_cond = d / 15
    mshd_domain = min(dx, dy) / 15

    (cl1, l) = DrawCircle(geo, (0, 0), d / 2, mshd_cond)
    ps1 = geo.addPlaneSurface([cl1])

    p1 = geo.addPoint(-dx / 2, -dy / 2, 0, mshd_cond)
    p2 = geo.addPoint(+dx / 2, -dy / 2, 0, mshd_cond)
    p3 = geo.addPoint(+dx / 2, +dy / 2, 0, mshd_cond)
    p4 = geo.addPoint(-dx / 2, +dy / 2, 0, mshd_cond)
    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p1)
    cl = geo.addCurveLoop([l1, l2, l3, l4])
    ps = geo.addPlaneSurface([cl, cl1])

    cl = [cl]
    l = [l]
    ps = [ps]

    coords = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for coord ∈ coords
        x = coord[1] * dx
        y = coord[2] * dy

        (cl_, l_) = DrawCircle(geo, (x, y), d / 2, mshd_cond)
        ps_ = geo.addPlaneSurface([cl_])

        push!(cl, cl_)
        push!(l, l_)
        push!(ps, ps_)
    end

    (cl_dom, l_dom) = DrawCircle(geo, (0, 0), 2 * max(dx, dy), mshd_domain)
    ps_dom = geo.addPlaneSurface(vcat(cl_dom, cl))

    conductors = vcat(ps1, ps[2:end])

    # Physical groups 
    geo.synchronize()
    geo.addPhysicalGroup(1, l_dom, 1)
    geo.addPhysicalGroup(2, [ps_dom, ps[1]], 1)
    geo.addPhysicalGroup(2, conductors, 2)
    geo.addPhysicalGroup(2, [ps[1], conductors[1]], 3)
    model.setPhysicalName(1, 1, "Domain")
    model.setPhysicalName(2, 1, "Air")
    model.setPhysicalName(2, 2, "Conductors")
    model.setPhysicalName(2, 3, "Cell0")

    for (i, ps_) ∈ enumerate(conductors)
        pg = geo.addPhysicalGroup(2, [ps_])
        model.setPhysicalName(2, pg, "Conductor" * string(i))
    end

    # Mesh
    geo.synchronize()
    mesh.generate(2)

    gmsh.write(path * name * ".msh")

    gmsh.finalize()
end

function GenerateGeometryHex(d::Real, λ::Real, name::String; path="examples/geo/")
    A = π / 4 * d^2
    d0 = √(2 * A / (λ * √(3)))

    gmsh.initialize()

    model = gmsh.model
    geo = model.geo
    mesh = model.mesh

    mshd_cond = d / 15
    mshd_domain = d0 / 15

    (cl1, l) = DrawCircle(geo, (0, 0), d / 2, mshd_cond)
    ps1 = geo.addPlaneSurface([cl1])

    r = d0 / √(3)
    p1 = geo.addPoint(r, 0, 0, mshd_cond)
    p2 = geo.addPoint(r * cos(π / 3), r * sin(π / 3), 0, mshd_cond)
    p3 = geo.addPoint(r * cos(2π / 3), r * sin(2π / 3), 0, mshd_cond)
    p4 = geo.addPoint(r * cos(3π / 3), r * sin(3π / 3), 0, mshd_cond)
    p5 = geo.addPoint(r * cos(4π / 3), r * sin(4π / 3), 0, mshd_cond)
    p6 = geo.addPoint(r * cos(5π / 3), r * sin(5π / 3), 0, mshd_cond)
    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p5)
    l5 = geo.addLine(p5, p6)
    l6 = geo.addLine(p6, p1)
    cl = geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
    ps = geo.addPlaneSurface([cl, cl1])

    cl = [cl]
    l = [l]
    ps = [ps]

    for i ∈ range(0, 5)
        θ = π / 3 * i
        x = d0 * sin(θ)
        y = d0 * cos(θ)
        (cl_, l_) = DrawCircle(geo, (x, y), d / 2, mshd_cond)
        ps_ = geo.addPlaneSurface([cl_])

        push!(cl, cl_)
        push!(l, l_)
        push!(ps, ps_)
    end

    (cl_dom, l_dom) = DrawCircle(geo, (0, 0), 2 * d0, mshd_domain)
    ps_dom = geo.addPlaneSurface(vcat(cl_dom, cl))

    conductors = vcat(ps1, ps[2:end])

    # Physical groups 
    geo.synchronize()
    geo.addPhysicalGroup(1, l_dom, 1)
    geo.addPhysicalGroup(2, [ps_dom, ps[1]], 1)
    geo.addPhysicalGroup(2, conductors, 2)
    geo.addPhysicalGroup(2, [ps[1], conductors[1]], 3)
    model.setPhysicalName(1, 1, "Domain")
    model.setPhysicalName(2, 1, "Air")
    model.setPhysicalName(2, 2, "Conductors")
    model.setPhysicalName(2, 3, "Cell0")

    for (i, ps_) ∈ enumerate(conductors)
        pg = geo.addPhysicalGroup(2, [ps_])
        model.setPhysicalName(2, pg, "Conductor" * string(i))
    end

    # Mesh
    geo.synchronize()
    mesh.generate(2)

    gmsh.write(path * name * ".msh")

    gmsh.finalize()
end