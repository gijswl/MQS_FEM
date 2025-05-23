using Gmsh

ri = 19.1e-3;   # Cable inner radius
ro = 37.5e-3;   # Insulation outer radius
rj = 46.5e-3;   # Cable outer radius

lc1 = 5e-3;     # Mesh density at outside of cable
lc2 = 1e-3;   # Mesh density at the conductor edge
lc3 = 30e-3;

x1 =   0; y1 =  2/3 * sqrt(3) * rj;
x2 = -rj*1.001; y2 = -1/3 * sqrt(3) * rj;
x3 =  rj*1.001; y3 = -1/3 * sqrt(3) * rj;
     

function gmsh_add_circle(mid, radius, lc)
    geo = gmsh.model.geo;
    
    # Corner points
    p1 = geo.addPoint(mid[1] + radius, mid[2], 0, lc);
    p2 = geo.addPoint(mid[1] - radius, mid[2], 0, lc);
    p3 = geo.addPoint(mid[1], mid[2], 0, 1);
    points = [p1, p2];
    
    # Lines
    l1 = geo.addCircleArc(p1, p3, p2);
    l2 = geo.addCircleArc(p2, p3, p1);
    lines = [l1, l2];
    
    # Curve loop
    loop = geo.addCurveLoop(lines);
    
    return loop, lines, points;
end
     

gmsh.initialize()
gmsh.model.add("cable_trefoil")
model = gmsh.model
geo = model.geo
mesh = model.mesh

## Domain
domain_lp, domain_lines, _ = gmsh_add_circle([0, 0], 10*rj, lc3);

## Cables
con1_lp, con1_edges, _ = gmsh_add_circle([x1, y1], ri, lc2);
ins1_lp, _, _ = gmsh_add_circle([x1, y1], ro, lc1);
jac1_lp, _, _ = gmsh_add_circle([x1, y1], rj, lc1);

con2_lp, con2_edges, _ = gmsh_add_circle([x2, y2], ri, lc2);
ins2_lp, _, _ = gmsh_add_circle([x2, y2], ro, lc1);
jac2_lp, _, _ = gmsh_add_circle([x2, y2], rj, lc1);

con3_lp, con3_edges, _ = gmsh_add_circle([x3, y3], ri, lc2);
ins3_lp, _, _ = gmsh_add_circle([x3, y3], ro, lc1);
jac3_lp, _, _ = gmsh_add_circle([x3, y3], rj, lc1);

## Plane surfaces
domain_surf = geo.addPlaneSurface([domain_lp, jac1_lp, jac2_lp, jac3_lp])

jac1_surf = geo.addPlaneSurface([jac1_lp, ins1_lp])
jac2_surf = geo.addPlaneSurface([jac2_lp, ins2_lp])
jac3_surf = geo.addPlaneSurface([jac3_lp, ins3_lp])

ins1_surf = geo.addPlaneSurface([ins1_lp, con1_lp])
ins2_surf = geo.addPlaneSurface([ins2_lp, con2_lp])
ins3_surf = geo.addPlaneSurface([ins3_lp, con3_lp])

con1_surf = geo.addPlaneSurface([con1_lp])
con2_surf = geo.addPlaneSurface([con2_lp])
con3_surf = geo.addPlaneSurface([con3_lp])

geo.synchronize();

## Physical domains
boundary = geo.addPhysicalGroup(1, domain_lines)
domain   = geo.addPhysicalGroup(2, [domain_surf])

con1 = geo.addPhysicalGroup(2, [con1_surf])
con2 = geo.addPhysicalGroup(2, [con2_surf])
con3 = geo.addPhysicalGroup(2, [con3_surf])

insulator = geo.addPhysicalGroup(2, [ins1_surf, ins2_surf, ins3_surf])
jacket    = geo.addPhysicalGroup(2, [jac1_surf, jac2_surf, jac3_surf])

model.setPhysicalName(1, boundary, "Boundary")
model.setPhysicalName(2, domain, "Air")
model.setPhysicalName(2, con1, "Conductor1")
model.setPhysicalName(2, con2, "Conductor2")
model.setPhysicalName(2, con3, "Conductor3")
model.setPhysicalName(2, insulator, "Insulator")
model.setPhysicalName(2, jacket, "Jacket")

# Generate mesh and save
bl = mesh.field.add("BoundaryLayer")
mesh.field.setNumbers(bl, "CurvesList", con1_edges)
mesh.field.setNumbers(bl, "ExcludedSurfacesList", [ins1_surf])
mesh.field.setNumber(bl, "Quads", 1)
mesh.field.setNumber(bl, "Ratio", 1.1)
mesh.field.setNumber(bl, "Size", 0.1e-3)
mesh.field.setNumber(bl, "SizeFar", 1e-3)
mesh.field.setNumber(bl, "IntersectMetrics", 1)
mesh.field.setNumber(bl, "NbLayers", 10)
mesh.field.setNumber(bl, "Thickness", 5e-3)
mesh.field.setAsBoundaryLayer(bl)

bl = mesh.field.add("BoundaryLayer")
mesh.field.setNumbers(bl, "CurvesList", con2_edges)
mesh.field.setNumbers(bl, "ExcludedSurfacesList", [ins2_surf])
mesh.field.setNumber(bl, "Quads", 1)
mesh.field.setNumber(bl, "Ratio", 1.1)
mesh.field.setNumber(bl, "Size", 0.1e-3)
mesh.field.setNumber(bl, "SizeFar", 1e-3)
mesh.field.setNumber(bl, "IntersectMetrics", 1)
mesh.field.setNumber(bl, "NbLayers", 10)
mesh.field.setNumber(bl, "Thickness", 5e-3)
mesh.field.setAsBoundaryLayer(bl)

bl = mesh.field.add("BoundaryLayer")
mesh.field.setNumbers(bl, "CurvesList", con3_edges)
mesh.field.setNumbers(bl, "ExcludedSurfacesList", [ins3_surf])
mesh.field.setNumber(bl, "Quads", 1)
mesh.field.setNumber(bl, "Ratio", 1.1)
mesh.field.setNumber(bl, "Size", 0.1e-3)
mesh.field.setNumber(bl, "SizeFar", 1e-3)
mesh.field.setNumber(bl, "IntersectMetrics", 1)
mesh.field.setNumber(bl, "NbLayers", 10)
mesh.field.setNumber(bl, "Thickness", 5e-3)
mesh.field.setAsBoundaryLayer(bl)

geo.synchronize()
mesh.generate(2);

gmsh.write("examples/hv_cable/geo/cable_trefoil.msh")

gmsh.fltk.run()
gmsh.finalize()