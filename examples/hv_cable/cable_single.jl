include("../../src/Main.jl")
using FerriteGmsh
using TimerOutputs

using Krylov: BicgstabWorkspace, bicgstab!
using Krylov: GmresWorkspace, gmres!
using KrylovPreconditioners: ilu

reset_timer!()

@timeit "setup" begin
    grid = saved_file_to_grid("examples/hv_cable/geo/cable_single.msh")

    f = 50
    T = Complex{Float64}
    A_cond = π * 19.1e-3^2
    I_cond = 1000

    materials = Dict(
        "Conductor" => Dict(
            "μr" => 1,
            "σ" => 36.9e6
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
        symmetry=Planar2D(1),
        time=TimeHarmonic(ω=2π * f),
        #time = TimeStatic(),
        fe_order=2,
        qr_order=4,
        materials=materials,
        sources=Dict(),
        boundaries=boundaries
    )

    cv, dh = init_problem(prob, grid)
    cch = CircuitHandler(dh, T)
    add_current_coupling!(cch, "Conductor", I_cond, A_cond, 0.25)

    ch = init_constraints(dh, prob)
    cellparams = init_params(dh, prob)

    K = allocate(dh, cch, prob)
end

@timeit "assemble" begin
    K, f = assemble_global(K, dh, cv, prob, cellparams)
    apply_circuit_couplings!(prob, prob.time, cellparams, K, f, cv, dh, cch)

    apply!(K, f, ch)
end

@timeit "solve" begin
    # # Direct
    # u = K \ f

    # Iterative
    ilu_τ = 1e-3 * maximum(norm.(K))

    Pℓ = ilu(K, τ=ilu_τ)
    #workspace = BicgstabWorkspace(K, f)
    #workspace = bicgstab!(workspace, K, f; M=Pℓ, ldiv=true, itmax=1000, verbose=5, history=true)

    workspace = GmresWorkspace(K, f)
    workspace = gmres!(workspace, K, f; M=Pℓ, ldiv=true, itmax=1000, verbose=5, history=true)

    u = workspace.x
    stats = workspace.stats
end

@timeit "post-processing" begin
    Bre_cell, Bim_cell = ComputeFluxDensity(dh, cv, u, prob, cellparams)
    J_cell = ComputeCurrentDensity(dh, cv, cch, u, prob, cellparams)
    S_cell = ComputeLossDensity(dh, cv, J_cell, Bre_cell, Bim_cell, prob, cellparams)
    (_, I_circ, _, R_circ) = ComputeLoss(dh, cv, cch, J_cell, Bre_cell, Bim_cell, prob, cellparams)

    VTKGridFile("examples/hv_cable/results/cable_single", dh) do vtk
        write_solution(vtk, dh, abs.(u), "_abs")
        write_cell_data(vtk, Bre_cell, "B_real")
        write_cell_data(vtk, Bim_cell, "B_imag")
        write_cell_data(vtk, real.(J_cell), "J_real")
        write_cell_data(vtk, imag.(J_cell), "J_imag")
        write_cell_data(vtk, norm.(J_cell), "J_norm")
        write_cell_data(vtk, imag.(S_cell), "S_imag")
        write_cell_data(vtk, real.(S_cell), "S_real")
    end
end

print_timer()

## Plot convergence history
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], title="Convergence", xlabel="Iteration", ylabel="Residual norm", yscale=log10)
lines!(ax, stats.residuals)
display(fig)


σ = materials["Conductor"]["σ"]
d = 2 * 19.1e-3
Rdc = 1 / (σ * π/4 * d^2)

println("DC resistance: $(Rdc * 1e6) mΩ/km")
println("AC resistance: $(R_circ[1] * 1e6) mΩ/km @ f = $(prob.time.ω / 2π) Hz")