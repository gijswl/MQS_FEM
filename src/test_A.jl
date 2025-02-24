using Ferrite
using FerriteGmsh
using SparseArrays

# RefTetrahedron, 1st order Lagrange
# https://defelement.org/elements/examples/tetrahedron-nedelec1-lagrange-1.html
#Ferrite.facedof_indices(::Nedelec{RefTetrahedron, 1}) = ((1, 3, 2), (1, 2, 4), (2, 3, 4), (1, 4, 3))
Ferrite.edgedof_indices(::Nedelec{RefTetrahedron, 1}) = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))

function Ferrite.reference_coordinates(::Nedelec{RefTetrahedron, 1})
    return [
        Vec{3, Float64}((0.0, 0.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
    ]
end

function Ferrite.reference_shape_value(ip::Nedelec{RefTetrahedron,1}, ξ::Vec{3}, i::Int)
    x, y, z = ξ

    i == 1 && return Vec(0 * x, -z, y)
    i == 2 && return Vec(-z, 0 * x, x)
    i == 3 && return Vec(-y, x, 0 * x)
    i == 4 && return Vec(z, z, -x - y + 1)
    i == 5 && return Vec(y, -x - z + 1, y)
    i == 6 && return Vec(-y - z + 1, x, x)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

Ferrite.getnbasefunctions(::Nedelec{RefTetrahedron,1}) = 6
Ferrite.adjust_dofs_during_distribution(::Nedelec{RefTetrahedron,1}) = false

function assemble_global(cv, K::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(Complex{Float64}, n_basefuncs, n_basefuncs)
    fe = zeros(Complex{Float64}, n_basefuncs)

    # Allocate global force vector f
    f = zeros(Complex{Float64}, size(K, 1))

    # Create an assembler
    assembler = start_assemble(K, f)

    dofs_A = pairs(dof_range(dh, :A))

    # Loop over all cells
    for cell ∈ CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cv, cell)

        cell_id = cellid(cell)

        Je = J0[cell_id]
        σe = σ[cell_id]
        νe = ν[cell_id]

        # Compute element contribution
        assemble_element!(Ke, fe, cv, dofs_A, ω, νe, σe, Je)

        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

function assemble_element!(Ke::Matrix, fe::Vector, cv, dofs_A, ω::Real, νe::Complex, σe::Complex, Je::Vec{2,<:Complex})
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)

    for q_point ∈ 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)

        # Loop over test shape functions
        for (i, dof_i) ∈ dofs_A
            v = shape_value(cv, q_point, i)
            curl_v = shape_curl(cv, q_point, i)

            # Add contribution to fe
            fe[dof_i] += Je ⋅ v * dΩ

            # Loop over trial shape functions
            for (j, dof_j) ∈ dofs_A
                u = shape_value(cv, q_point, j)
                curl_u = shape_curl(cv, q_point, j)

                Ke[dof_i, dof_j] += (νe * curl_u ⋅ curl_v + 1im * ω * σe * u ⋅ v) * dΩ
            end
        end
    end

    return Ke, fe
end

grid = saved_file_to_grid("test/mesh/team7.msh");

order = 1
shape = RefTetrahedron

ip_A = Nedelec{shape,order}()
ip_geo = Lagrange{shape,1}()

qr = QuadratureRule{shape}(2 * order)
cv = (A = CellValues(qr, ip_A, ip_geo))

dh = DofHandler(grid)
add!(dh, :A, ip_A)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, WeakDirichlet(:A, getfacetset(dh.grid, "outer"), (x, _, n) -> zero(Vec{3}))) # WeakDirichlet requires kam/WeakDirichlet branch of Ferrite.jl
close!(ch)

## Simulation settings
ω = 2π * 50

materials = Dict(
    "Conductor" => Dict(
        "μr" => 1,
        "σ" => 36.9e6,
        "J0" => Vec{2}([1, 0])),
    "Sheath" => Dict(
        "μr" => 1,
        "σ" => 59.6e6
    )
)

Ncells = getncells(dh.grid)
J0 = zeros(Vec{2,Complex{Float64}}, Ncells)
σ = 1e-9 * ones(Complex{Float64}, Ncells)
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

μ0 = 4π * 1e-7
ν = 1 ./ (μ0 * μr)


ndof = ndofs(dh)
sp = SparsityPattern(ndof, ndof; nnz_per_row=2 * ndofs_per_cell(dh.subdofhandlers[1])) # How to optimize nnz_per_row?
add_sparsity_entries!(sp, dh)

K = allocate_matrix(SparseMatrixCSC{Complex{Float64},Int}, sp)
K, f = assemble_global(cv, K, dh)

apply!(K, f, ch)

u = K \ f


VTKGridFile("test/results/test_cable_single", dh) do vtk
    write_solution(vtk, dh, abs.(u))
end