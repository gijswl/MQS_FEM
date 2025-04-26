include("../src/Main.jl")
using FerriteGmsh

grid = saved_file_to_grid("examples/mesh/cable_single.msh")

f = 50
T = Complex{Float64}

materials = Dict(
    "Conductor" => Dict(
        "μr" => 1,
        "σ" => 36.9e6, 
        "J0" => 0.8e6
    ),
    "Sheath" => Dict(
        "μr" => 1,
        "σ" => 59.6e6
    )
)

boundaries = Dict(
    "jacket" => BoundaryInfty(T)
)

prob = Problem2D{T}(
    symmetry = Planar2D(),
    time = TimeHarmonic(ω = 2π * f),
    #time = TimeStatic(),
    fe_order = 2,
    qr_order = 4,
    materials = materials,
    sources = Dict(),
    boundaries = boundaries
)

cv, dh = init_problem(prob, grid)
ch = init_constraints(dh, prob)
cellparams = init_params(dh, prob)

K = allocate(dh, prob)
K, f = assemble_global(K, dh, cv, prob, cellparams)

apply!(K, f, ch)

# Solve the linear system
u = K \ f;
Bre_cell, Bim_cell = ComputeFluxDensity(dh, cv, u, prob, cellparams)
J_cell = ComputeCurrentDensity(dh, cv, u, prob, cellparams)
S_cell = ComputeLossDensity(dh, cv, u, J_cell, Bre_cell, Bim_cell, prob, cellparams)

VTKGridFile("examples/results/cable_single", dh) do vtk
    write_solution(vtk, dh, abs.(u), "_abs")
    write_cell_data(vtk, Bre_cell, "B_real")
    write_cell_data(vtk, Bim_cell, "B_imag")
    write_cell_data(vtk, real.(J_cell), "J_real")
    write_cell_data(vtk, imag.(J_cell), "J_imag")
    write_cell_data(vtk, norm.(J_cell), "J_norm")
    write_cell_data(vtk, imag.(S_cell), "S_imag")
    write_cell_data(vtk, real.(S_cell), "S_real")
end