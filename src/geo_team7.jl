using Gmsh

gmsh.initialize()
model = gmsh.model
occ = model.occ
mesh = model.mesh

model.add("TEAM7")

r1 = occ.addRectangle(0, 0, 0, 0.294, 0.294)
r2 = occ.addRectangle(0.018, 0.018, 0, 0.108, 0.108)
r3 = occ.cut([(2, r1)], [(2, r2)])
p1 = occ.extrude(r3[1], 0, 0, 0.019)

r4 = occ.addRectangle(0.094, 0, 0.049, 0.200, 0.200, -1, 0.050)
r5 = occ.addRectangle(0.094 + 0.025, + 0.025, 0.049, 0.150, 0.150, -1, 0.025)
r6 = occ.cut([(2, r4)], [(2, r5)])
p2 = occ.extrude(r6[1], 0, 0, 0.100)

domain = occ.addBox(-1.353, -1.353, -0.300, 1.647 + 1.353, 1.647 + 1.353, 0.449 + 0.300)
p3 = occ.fragment([(3, 3)], [(3, 1), (3, 2)])

occ.synchronize()

# Set mesh sizes
nodes_all = model.getEntities(0);
mesh.setSize(nodes_all, 0.2);

nodes_plate = model.getBoundary((3, 1), false, false, true)
mesh.setSize(nodes_plate, 0.02);

nodes_coil = model.getBoundary((3, 2), false, false, true)
mesh.setSize(nodes_coil, 0.02);

mesh.generate(3)

# Physical domains
faces_outer = model.getBoundary([(3, 1), (3, 2), (3, 3)], true, false)
model.addPhysicalGroup(2, [f[2] for f ∈ faces_outer], 1, "outer")

gmsh.write("test/mesh/team7.msh")