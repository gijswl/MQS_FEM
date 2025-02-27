using Gmsh

R_cond = 19.1e-3;
R_ins  = 18.4e-3;
R_sh   = 1e-3;
R_jac  = 8e-3;

r_cond = R_cond;             # Conductor
r_ins  = r_cond + R_cond;    # Insulator
r_sh   = r_ins + R_sh;       # Sheath
r_jac  = r_sh + R_jac;       # Jacket

# Mesh density
mshd_cond = 0.5e-3; 
mshd_ins  = R_ins / 10;
mshd_sh   = R_sh / 5;
mshd_jac  = 2e-3;
    
#gmsh.finalize()
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("cable_geo")

geo = gmsh.model.geo;

## Points
geo.addPoint(0, 0, 0, mshd_cond, 1)
geo.addPoint(r_cond, 0, 0, mshd_cond, 2)
geo.addPoint(0, r_cond, 0, mshd_cond, 3)
geo.addPoint(r_ins, 0, 0, mshd_sh, 4)
geo.addPoint(0, r_ins, 0, mshd_sh, 5)
geo.addPoint(r_sh, 0, 0, mshd_sh, 6)
geo.addPoint(0, r_sh, 0, mshd_sh, 7)
geo.addPoint(r_jac, 0, 0, mshd_jac, 8)
geo.addPoint(0, r_jac, 0, mshd_jac, 9)

## Curves
geo.addCircleArc(2, 1, 3, 1)
geo.addCircleArc(4, 1, 5, 2)
geo.addCircleArc(6, 1, 7, 3)
geo.addCircleArc(8, 1, 9, 4)

geo.addLine(1, 2, 5)
geo.addLine(2, 4, 6)
geo.addLine(4, 6, 7)
geo.addLine(6, 8, 8)

geo.addLine(3, 1, 9)
geo.addLine(5, 3, 10)
geo.addLine(7, 5, 11)
geo.addLine(9, 7, 12)

## Surfaces
geo.addCurveLoop([5, 1, 9], 1)
geo.addCurveLoop([6, 2, 10, -1], 2)
geo.addCurveLoop([7, 3, 11, -2], 3)
geo.addCurveLoop([8, 4, 12, -3], 4)

geo.addPlaneSurface([1], 1)
geo.addPlaneSurface([2], 2)
geo.addPlaneSurface([3], 3)
geo.addPlaneSurface([4], 4)

## Define domains
geo.addPhysicalGroup(2, [1], 1) # Conductor
geo.addPhysicalGroup(2, [2], 2) # Dielectric
geo.addPhysicalGroup(2, [3], 3) # Sheath
geo.addPhysicalGroup(2, [4], 4) # Jacket
gmsh.model.setPhysicalName(2, 1, "Conductor")
gmsh.model.setPhysicalName(2, 2, "Dielectric")
gmsh.model.setPhysicalName(2, 3, "Sheath")
gmsh.model.setPhysicalName(2, 4, "Jacket")

geo.addPhysicalGroup(1, [4], 1) # Jacket boundary
gmsh.model.setPhysicalName(1, 1, "jacket")

# Generate mesh and save
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

gmsh.write("examples/mesh/cable_single.msh")
gmsh.finalize()