using Gmsh

function GenerateGeometryRoundWire(d_cond::Real, d_domain::Real, name::String; path="geo/")
    mshd_cond = d_cond / 20
    mshd_dom = d_domain / 20

    gmsh.initialize()
    gmsh.model.add(name)

    model = gmsh.model
    geo = model.geo
    mesh = model.mesh

    # Vertices
    geo.addPoint(0, 0, 0, 1, 1) # Center point
    geo.addPoint(+d_cond / 2, 0, 0, mshd_cond, 2)
    geo.addPoint(-d_cond / 2, 0, 0, mshd_cond, 3)
    geo.addPoint(+d_domain / 2, 0, 0, mshd_dom, 4)
    geo.addPoint(-d_domain / 2, 0, 0, mshd_dom, 5)

    # Edges
    geo.addCircleArc(2, 1, 3, 1)
    geo.addCircleArc(3, 1, 2, 2)
    geo.addCircleArc(4, 1, 5, 3)
    geo.addCircleArc(5, 1, 4, 4)

    # Faces
    geo.addCurveLoop([1, 2], 1)
    geo.addCurveLoop([3, 4], 2)

    geo.addPlaneSurface([1], 1)
    geo.addPlaneSurface([2, 1], 2)

    # Physical groups 
    geo.synchronize()
    geo.addPhysicalGroup(1, [3, 4], 1)
    geo.addPhysicalGroup(2, [1], 1)
    geo.addPhysicalGroup(2, [2], 2)

    model.setPhysicalName(1, 1, "Domain")
    model.setPhysicalName(2, 1, "Conductor")
    model.setPhysicalName(2, 2, "Air")

    # Mesh
    geo.synchronize()
    mesh.generate(2)

    gmsh.write(path * name * ".msh")

    gmsh.finalize()
end

function GenerateGeometryRectWire(w::Real, t::Real, d_domain::Real, name::String; path="geo/")
    mshd_cond = w / 20
    mshd_dom = d_domain / 20

    gmsh.initialize()
    gmsh.model.add(name)

    model = gmsh.model
    geo = model.geo
    mesh = model.mesh

    # Vertices
    geo.addPoint(+w/2, +t/2, 0, mshd_cond, 1)
    geo.addPoint(-w/2, +t/2, 0, mshd_cond, 2)
    geo.addPoint(-w/2, -t/2, 0, mshd_cond, 3)
    geo.addPoint(+w/2, -t/2, 0, mshd_cond, 4)
    geo.addPoint(+d_domain / 2, 0, 0, mshd_dom, 5)
    geo.addPoint(-d_domain / 2, 0, 0, mshd_dom, 6)
    geo.addPoint(0, 0, 0, mshd_dom, 7)

    # Edges
    geo.addLine(1, 2, 1)
    geo.addLine(2, 3, 2)
    geo.addLine(3, 4, 3)
    geo.addLine(4, 1, 4)
    geo.addCircleArc(5, 7, 6, 5)
    geo.addCircleArc(6, 7, 5, 6)

    # Faces
    geo.addCurveLoop([1, 2, 3, 4], 1)
    geo.addCurveLoop([5, 6], 2)

    geo.addPlaneSurface([1], 1)
    geo.addPlaneSurface([2, 1], 2)

    # Physical groups 
    geo.synchronize()
    geo.addPhysicalGroup(1, [5, 6], 1)
    geo.addPhysicalGroup(2, [1], 1)
    geo.addPhysicalGroup(2, [2], 2)

    model.setPhysicalName(1, 1, "Domain")
    model.setPhysicalName(2, 1, "Conductor")
    model.setPhysicalName(2, 2, "Air")

    # Mesh
    geo.synchronize()

    mesh.setTransfiniteCurve(1, 35, "Bump", 0.1)
    mesh.setTransfiniteCurve(3, 35, "Bump", 0.1)
    mesh.setTransfiniteCurve(2, 15, "Bump", 0.1)
    mesh.setTransfiniteCurve(4, 15, "Bump", 0.1)
    mesh.setTransfiniteSurface(1, "Left")
    mesh.setRecombine(2, 1)

    mesh.generate(2)

    gmsh.write(path * name * ".msh")

    gmsh.finalize()
end