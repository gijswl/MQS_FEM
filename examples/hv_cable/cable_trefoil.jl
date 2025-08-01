include("../../src/Main.jl")
using FerriteGmsh
using TimerOutputs

using Krylov: BicgstabWorkspace, bicgstab!
using Krylov: GmresWorkspace, gmres!
using KrylovPreconditioners: ilu

reset_timer!()

@timeit "setup" begin
    grid = saved_file_to_grid("examples/hv_cable/geo/cable_trefoil.msh")
    preprocess_grid(grid)

    f = 50
    T = Complex{Float64}
    A_cond = π * 19.1e-3^2
    I_cond = 1000

    materials = Dict(
        "Conductor1" => Dict(
            "σ" => 36.9e6
        ),
        "Conductor2" => Dict(
            "σ" => 36.9e6
        ),
        "Conductor3" => Dict(
            "σ" => 36.9e6
        )
    )

    boundaries = Dict(
        "Boundary" => BoundaryInfty(T)
    )

    prob = Problem{T}(
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
    add_conductor_solid!(cch, "Conductor1")
    add_conductor_solid!(cch, "Conductor2")
    add_conductor_solid!(cch, "Conductor3")

    add_circuit_sol!(cch.circuit, "Conductor1", 2.3645964141395557e-5, 1, 0)
    add_circuit_sol!(cch.circuit, "Conductor2", 2.3645964141395557e-5, 2, 0)
    add_circuit_sol!(cch.circuit, "Conductor3", 2.3645964141395557e-5, 3, 0)
    add_circuit_I!(cch.circuit, I_cond * exp(0im * 2π / 3), 1, 0)
    add_circuit_I!(cch.circuit, I_cond * exp(+1im * 2π / 3), 2, 0)
    add_circuit_I!(cch.circuit, I_cond * exp(-1im * 2π / 3), 3, 0)
    close!(cch)

    cellparams = init_params(dh, cch, prob)
    ch = init_constraints(dh, prob)

    K = allocate(dh, cch, prob)
end

@timeit "assemble" begin
    K, f = assemble_global(K, dh, cv, prob, cellparams)
    apply_circuit_couplings!(K, f, dh, cv, cch, prob, cellparams)

    apply!(K, f, ch)
end

@timeit "solve" begin
    # Direct
    # u = K \ f

    # Iterative
    ilu_τ = 1e-4 * maximum(norm.(K))

    Pℓ = ilu(K, τ=ilu_τ)
    # workspace = BicgstabWorkspace(K, f)
    # workspace = bicgstab!(workspace, K, f; M=Pℓ, ldiv=true, itmax=1000, verbose=5, history=true)

    workspace = GmresWorkspace(K, f)
    workspace = gmres!(workspace, K, f; M=Pℓ, ldiv=true, itmax=1000, verbose=5, atol = 1e-12, rtol = 1e-12, history=true)

    u = workspace.x
    stats = workspace.stats
end

@timeit "post-processing" begin
    B = ComputeFluxDensity(dh, cv, u, prob, cellparams)
    J = ComputeCurrentDensity(dh, cv, cch, u, prob, cellparams)
    (I_circ, S_circ, R_circ) = ComputeLoss(dh, cv, cch, J, B, prob, cellparams)

    VTKGridFile("examples/hv_cable/results/cable_trefoil", dh; write_discontinuous=true) do vtk
        write_solution(vtk, dh, abs.(u), "_abs")
        write_postprocessed(vtk, dh, cch, u, prob, cellparams, :B_real)
        write_postprocessed(vtk, dh, cch, u, prob, cellparams, :B_imag)
        write_postprocessed(vtk, dh, cch, u, prob, cellparams, :B_norm)
        #write_postprocessed(vtk, dh, cch, u, prob, cellparams, :J_norm)
        #write_postprocessed(vtk, dh, cch, u, prob, cellparams, :S_real)
        #write_postprocessed(vtk, dh, cch, u, prob, cellparams, :S_imag)
        write_cell_data(vtk, norm.(J[:, 1]), "J_norm")
    end
end

print_timer()

# Plot convergence history
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], title="Convergence", xlabel="Iteration", ylabel="Residual norm", yscale=log10)
lines!(ax, stats.residuals)
display(fig)


σ = materials["Conductor1"]["σ"]
d = 2 * 19.1e-3
Rdc = 1 / (σ * π / 4 * d^2)

println("DC resistance: $(Rdc * 1e6) mΩ/km")
println("AC resistance: $(R_circ["Conductor1"] * 1e6) mΩ/km @ f = $(prob.time.ω / 2π) Hz")
println("AC resistance: $(R_circ["Conductor2"] * 1e6) mΩ/km @ f = $(prob.time.ω / 2π) Hz")
println("AC resistance: $(R_circ["Conductor3"] * 1e6) mΩ/km @ f = $(prob.time.ω / 2π) Hz")