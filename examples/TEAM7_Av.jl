using Ferrite
using FerriteGmsh
using SparseArrays

include("../src/FerriteAdditions.jl")
include("../src/PostProcessing3D.jl")

function assemble_global(cv, K::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cv.A) + getnbasefunctions(cv.ϕ)
    Ke = zeros(Complex{Float64}, n_basefuncs, n_basefuncs)
    fe = zeros(Complex{Float64}, n_basefuncs)

    # Allocate global force vector f
    f = zeros(Complex{Float64}, size(K, 1))

    # Create an assembler
    assembler = start_assemble(K, f)

    dofs_A = pairs(dof_range(dh, :A))
    dofs_ϕ = pairs(dof_range(dh, :ϕ))

    # Loop over all cells
    for cell ∈ CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cv.A, cell)
        reinit!(cv.ϕ, cell)

        cell_id = cellid(cell)

        Je = J0[cell_id]
        σe = σ[cell_id]
        νe = ν[cell_id]

        # Compute element contribution
        assemble_element!(Ke, fe, cv, dofs_A, dofs_ϕ, ω, νe, σe, Je)

        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

function assemble_element!(Ke::Matrix, fe::Vector, cv, dofs_A, dofs_ϕ, ω::Real, νe::Complex, σe::Complex, Je::Vec{3, <:Complex})
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)

    for q_point ∈ 1:getnquadpoints(cv.A)
        dΩ = getdetJdV(cv.A, q_point)

        # Loop over test shape functions
        for (i, dof_i) ∈ dofs_A
            v = shape_value(cv.A, q_point, i)
            curl_v = shape_curl(cv.A, q_point, i)

            # Add contribution to fe
            fe[dof_i] += Je ⋅ v * dΩ

            # Loop over trial shape functions
            for (j, dof_j) ∈ dofs_A
                u = shape_value(cv.A, q_point, j)
                curl_u = shape_curl(cv.A, q_point, j)

                Ke[dof_i, dof_j] += (νe * curl_u ⋅ curl_v + 1im * ω * σe * u ⋅ v) * dΩ
            end

            for (j, dof_j) ∈ dofs_ϕ
                ∇p = shape_gradient(cv.ϕ, q_point, j)

                Ke[dof_i, dof_j] += σe * ∇p ⋅ v * dΩ
            end
        end

        for (i, dof_i) ∈ dofs_ϕ
            ∇q = shape_gradient(cv.ϕ, q_point, i)

            # Loop over trial shape functions
            for (j, dof_j) ∈ dofs_A
                u = shape_value(cv.A, q_point, j)

                Ke[dof_i, dof_j] += 1im * ω * σe * u ⋅ ∇q * dΩ
            end

            for (j, dof_j) ∈ dofs_ϕ
                ∇p = shape_gradient(cv.ϕ, q_point, j)

                Ke[dof_i, dof_j] += σe * ∇p ⋅ ∇q * dΩ
            end
        end
    end

    return Ke, fe
end

grid = saved_file_to_grid("examples/mesh/team7.msh");

order = 1
shape = RefTetrahedron

ip_A = Nedelec{shape,order}()
ip_ϕ = Lagrange{shape,order}()
ip_geo = Lagrange{shape,1}()

qr = QuadratureRule{shape}(2 * order)
cv = (A = CellValues(qr, ip_A, ip_geo), ϕ = CellValues(qr, ip_ϕ, ip_geo))

dh = DofHandler(grid)
add!(dh, :A, ip_A)
add!(dh, :ϕ, ip_ϕ)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, WeakDirichlet(:A, getfacetset(dh.grid, "outer"), (x, _, n) -> zero(Vec{3}))) # WeakDirichlet requires kam/WeakDirichlet branch of Ferrite.jl
add!(ch, Dirichlet(:ϕ, getfacetset(dh.grid, "outer"), Returns(0.0)))
close!(ch)

## Simulation settings
ω = 2π * 200

materials = Dict(
    "Coil" => Dict(
    ),
    "Plate" => Dict(
        "σ" => 3.526e7,
    )
)

Ncells = getncells(dh.grid)
J0 = zeros(Vec{3,Complex{Float64}}, Ncells)
σ = 1e-3 * ones(Complex{Float64}, Ncells)
μr = ones(Complex{Float64}, Ncells)

for (domain, material) ∈ materials
    cellset = collect(getcellset(dh.grid, domain))

    if (haskey(material, "σ"))
        σ[cellset] .= material["σ"]
    end
    if (haskey(material, "μr"))
        μr[cellset] .= material["μr"]
    end
    if (haskey(material, "J0"))
        for cell ∈ cellset
            J0[cell] = material["J0"]
        end
    end
end

It = 2742
w = 0.025
h = 0.100
cx_min = 0.294 - 0.150
cx_max = 0.294 - 0.050
cy_min = 0.050
cy_max = 0.150

function proj(x, x_min, x_max)
    if (x - x_min > 0)
        if (x - x_max > 0)
            return x_max
        else
            return x
        end
    else
        return x_min
    end
end

# Loop over all cells in the coil
for cell ∈ CellIterator(dh, getcellset(dh.grid, "Coil"))
    # Reinitialize cellvalues for this cell
    reinit!(cv.A, cell)
    reinit!(cv.ϕ, cell)

    cell_id = cellid(cell)
    xe = getcoordinates(dh.grid, cell_id)

    Jcell = zero(Vec{3,Complex{Float64}})
    for x ∈ xe
        proj_x = proj(x[1], cx_min, cx_max)
        proj_y = proj(x[2], cy_min, cy_max)
        τ = Vec{3}((proj_y - x[2], x[1] - proj_x, 0))
        Jcell += It / (w * h) / length(xe) * τ / norm(τ)
    end

    J0[cell_id] = Jcell
end

μ0 = 4π * 1e-7
ν = 1 ./ (μ0 * μr)


ndof = ndofs(dh)
sp = SparsityPattern(ndof, ndof; nnz_per_row=2 * ndofs_per_cell(dh.subdofhandlers[1])) # How to optimize nnz_per_row?
add_sparsity_entries!(sp, dh)

K = allocate_matrix(SparseMatrixCSC{Complex{Float64},Int}, sp)
K, f = assemble_global(cv, K, dh)

apply!(K, f, ch)

# Direct solvers
# umfpack_control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int64) # read Julia default configuration for a Float64 sparse matrix
# umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 2.0 # reenable iterative refinement (2 is UMFPACK default max iterative refinement steps)
# Klu = lu(K; control = umfpack_control)

# u = Klu \ f

# Iterative solvers
p = IterativeSolvers.Identity()
# p = DiagonalPreconditioner(K)
# (u, hist) = bicgstabl(K, f, 4; Pl = p, log = true, verbose = true, max_mv_products = 2000)
(u, hist) = gmres(K, f; Pl = p, log = true, verbose = true, maxiter = 1000, reltol = 1e-8)

fig = Figure()
ax = Axis(fig[1,1], title = "Convergence", xlabel = "Iteration", ylabel = "Residual norm", yscale = log10)
lines!(ax, hist[:resnorm])
display(fig)

B = ComputeFluxDensity(cv.A, cv.ϕ, dh, u)
Babs = [Vec{3}(abs.(Bel)) for Bel ∈ B]
Breal = [Vec{3}(real.(Bel)) for Bel ∈ B]
Bimag = [Vec{3}(imag.(Bel)) for Bel ∈ B]

J = ComputeCurrentDensity(cv.A, cv.ϕ, dh, u)
Jabs = [Vec{3}(abs.(Jel)) for Jel ∈ J]
Jreal = [Vec{3}(real.(Jel)) for Jel ∈ J]
Jimag = [Vec{3}(imag.(Jel)) for Jel ∈ J]

VTKGridFile("examples/results/team7_200Hz", dh) do vtk
    write_solution(vtk, dh, abs.(u))
    write_cell_data(vtk, Babs, "abs(B)")
    write_cell_data(vtk, Breal, "real(B)")
    write_cell_data(vtk, Bimag, "imag(B)")
    write_cell_data(vtk, Jabs, "abs(J)")
    write_cell_data(vtk, Jreal, "real(J)")
    write_cell_data(vtk, Jimag, "imag(J)")
end