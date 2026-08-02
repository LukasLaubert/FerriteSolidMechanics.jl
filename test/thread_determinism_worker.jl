# Worker for test_thread_determinism.jl; spawned as a subprocess, not run by
# runtests.jl. Prints one hash per case; the parent requires them to match

using Ferrite
using FerriteSolidMechanics
using Tensors
using LinearAlgebra
using Printf

# Sized so the assembler spawns several tasks (ntasks = min(nworkspaces, ncells))
const N3D = 6      # 216 hexes
const N2D = 16     # 256 quads

hashof(K, r) = hash((K.nzval, r))

function block3d(n)
    grid = generate_grid(Hexahedron, (n, n, n))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]))
    close!(ch)
    return grid, dh, ch
end

function plate2d(n)
    grid = generate_grid(Quadrilateral, (n, n))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0], [1, 2]))
    close!(ch)
    return grid, dh, ch
end

# Mixed SubDofHandlers with different ndofs_per_cell, plus two materials
function mixed2d(n)
    grid = generate_grid(Quadrilateral, (n, n))
    ncells = getncells(grid)
    half = ncells ÷ 2
    addcellset!(grid, "lo", Set(1:half))
    addcellset!(grid, "hi", Set((half + 1):ncells))
    dh = DofHandler(grid)
    sdh_lo = SubDofHandler(dh, getcellset(grid, "lo"))
    add!(sdh_lo, :u, Lagrange{RefQuadrilateral,2}()^2)
    sdh_hi = SubDofHandler(dh, getcellset(grid, "hi"))
    add!(sdh_hi, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)
    ch = ConstraintHandler(dh)
    close!(ch)
    return grid, dh, ch
end

wobble(n) = [1.0e-3 * sin(3.1 * i) for i in 1:n]

results = Pair{String,UInt64}[]

# 1. Single nonlinear material, 3D, generic material_response path
let (grid, dh, ch) = block3d(N3D)
    fem = create_assembler(NeoHooke(100.0, 0.3), dh, ch; quadrature_order=2)
    u = wobble(ndofs(dh))
    K, r = stiffness_matrix(fem, u; dt=0.0)
    push!(results, "neo_hooke_3d" => hashof(K, r))
    push!(results, "neo_hooke_3d_stress" => hash(compute_stresses(fem, u)))
end

# 2. History material: trial-state writes happen inside the threaded loop
let (grid, dh, ch) = block3d(N3D)
    fem = create_assembler(J2Plasticity(100.0, 0.3, 1.0e-3, 10.0), dh, ch; quadrature_order=2)
    u = wobble(ndofs(dh))
    K, r = stiffness_matrix(fem, u; dt=1.0)
    push!(results, "j2_3d" => hashof(K, r))
    update_states!(fem)   # also threaded
    push!(results, "j2_3d_committed" => hash([(s.previous.σ, s.previous.k) for cid in fem.owned_nonlinear_cells for s in fem.states[cid]]))
    K2, r2 = stiffness_matrix(fem, 1.5 .* u; dt=1.0)
    push!(results, "j2_3d_step2" => hashof(K2, r2))
    revert_states!(fem)   # also threaded
    push!(results, "j2_3d_reverted" => hash([(s.current.σ, s.current.k) for cid in fem.owned_nonlinear_cells for s in fem.states[cid]]))
end

# 3. Custom _assemble_element! path (element-structured tangent)
let (grid, dh, ch) = block3d(4)
    zhao = VEVP_Zhao2021_AT(1.0, 100.0, 2, 0.1, 0.1, 1.0, 0.5, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0)
    fem = create_assembler(zhao, dh, ch; quadrature_order=2)
    u = [1.0e-6 * sin(3.1 * i) for i in 1:ndofs(dh)]
    K, r = stiffness_matrix(fem, u; dt=0.1)
    push!(results, "zhao_at_3d" => hashof(K, r))
end

# 4. Plane wrapper (local Newton per quadrature point)
let (grid, dh, ch) = plate2d(N2D)
    fem = create_assembler(PlaneStress(NeoHooke(100.0, 0.3)), dh, ch; quadrature_order=2)
    u = wobble(ndofs(dh))
    K, r = stiffness_matrix(fem, u; dt=0.0)
    push!(results, "planestress_neo_2d" => hashof(K, r))
end

# 5. Mixed SubDofHandlers + linear/nonlinear material dict
let (grid, dh, ch) = mixed2d(N2D)
    fem = create_assembler(Dict("lo" => PlaneStrain(NeoHooke(100.0, 0.3)), "hi" => Hooke2D(200.0, 0.3)),
                           dh, ch; quadrature_order=[3, 2])
    u = wobble(ndofs(dh))
    K, r = stiffness_matrix(fem, u; dt=0.0)
    push!(results, "mixed_sdh_2d" => hashof(K, r))
    push!(results, "mixed_sdh_2d_stress" => hash(compute_stresses(fem, u)))
end

println("NTHREADS ", Threads.nthreads())
for (name, h) in results
    @printf("HASH %s %016x\n", name, h)
end
