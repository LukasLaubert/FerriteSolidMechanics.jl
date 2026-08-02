# Worker for test_mpi.jl; spawned under mpiexec, not run by the test suite
# Serializes results to ARGS[1] for the parent to compare across rank counts

using MPI
MPI.Init()

using Ferrite
using FerriteSolidMechanics
using Tensors
using LinearAlgebra
using Serialization

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)
const OUTFILE = ARGS[1]

# Enough cells that `mod(cellid-1, nranks) == rank` actually splits work
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

wobble(n) = [1.0e-3 * sin(3.1 * i) for i in 1:n]

fail(msg) = (RANK == 0 && println("FAIL ", msg); MPI.Finalize(); exit(1))

"""
Every rank must hold the same bytes after the Allreduce -- exact, unlike
agreement across different rank counts.
"""
function assert_ranks_agree(label, K, r)
    NRANKS == 1 && return
    ref_K = MPI.bcast(copy(K.nzval), 0, COMM)
    ref_r = MPI.bcast(copy(r), 0, COMM)
    agree = (K.nzval == ref_K) && (r == ref_r)
    MPI.Allreduce(agree, &, COMM) || fail("ranks disagree after Allreduce: $label")
end

# Fails on rank 0 only, so the recoverable-failure flag exchange has to carry the
# result to ranks that succeeded locally
struct RankZeroFailure <: LocalAssemblyFailure end
struct FailOnRankZero <: AbstractMaterial end
FerriteSolidMechanics.kinematics(::FailOnRankZero) = SmallStrain()
function FerriteSolidMechanics.material_response(::FailOnRankZero, ε::SymmetricTensor{2,3}, state, dt, cache=nothing)
    RANK == 0 && throw(RankZeroFailure())
    return zero(ε), zero(SymmetricTensor{4,3}), state
end

results = Dict{String,Any}()

# Nonlinear, stateless: exercises the tangent/residual Allreduce
let (grid, dh, ch) = block3d(6)
    fem = create_assembler(NeoHooke(100.0, 0.3), dh, ch; quadrature_order=2)
    u = wobble(ndofs(dh))
    K, r = stiffness_matrix(fem, u; dt=0.0)
    assert_ranks_agree("neo_hooke_3d", K, r)
    results["neo_hooke_3d"] = (copy(K.nzval), copy(r))
    # Flattened to plain Float64 so the parent deserializes without Tensors
    results["neo_hooke_3d_stress"] = collect(vec(reinterpret(Float64, compute_stresses(fem, u))))
end

# Linear material: exercises the separate K_linear preassembly Allreduce
let (grid, dh, ch) = block3d(6)
    fem = create_assembler(Hooke(200.0, 0.3), dh, ch; quadrature_order=2)
    u = wobble(ndofs(dh))
    K, r = stiffness_matrix(fem, u; dt=0.0)
    assert_ranks_agree("hooke_3d_linear", K, r)
    results["hooke_3d_linear"] = (copy(K.nzval), copy(r))
end

# History material: per-rank state ownership + commit across a step
let (grid, dh, ch) = block3d(6)
    fem = create_assembler(J2Plasticity(100.0, 0.3, 1.0e-3, 10.0), dh, ch; quadrature_order=2)
    u = wobble(ndofs(dh))
    K, r = stiffness_matrix(fem, u; dt=1.0)
    assert_ranks_agree("j2_3d", K, r)
    results["j2_3d"] = (copy(K.nzval), copy(r))
    update_states!(fem)
    K2, r2 = stiffness_matrix(fem, 1.5 .* u; dt=1.0)
    assert_ranks_agree("j2_3d_step2", K2, r2)
    results["j2_3d_step2"] = (copy(K2.nzval), copy(r2))
end

# External loads are replicated, not reduced: every rank walks every facet, node
# and cell, so rank counts must agree bitwise rather than to a tolerance
let (grid, dh, ch) = block3d(6)
    fem = create_assembler(NeoHooke(100.0, 0.3), dh, ch; quadrature_order=2)
    addnodeset!(grid, "load_nodes", x -> x[1] ≈ 1.0)
    lh = LoadHandler(fem)
    add!(lh, Traction("right", (x, t) -> Vec(0.3 * t, 0.1 * x[2], 0.0)))
    add!(lh, Pressure("top", (x, t) -> 0.5 * t))
    add!(lh, NodalForce("load_nodes", (x, t) -> Vec(0.0, -0.2 * t, 0.0); distribute=true))
    add!(lh, BodyForce(x -> Vec(0.0, 0.0, -0.05)))
    close!(lh)
    results["external_forces"] = copy(external_forces!(lh, 1.5))
end

# try_stiffness_matrix must report converged collectively on a healthy solve
let (grid, dh, ch) = block3d(4)
    fem = create_assembler(NeoHooke(100.0, 0.3), dh, ch; quadrature_order=2)
    u = wobble(ndofs(dh))
    res = try_stiffness_matrix(fem, u; dt=0.0)
    res.converged || fail("try_stiffness_matrix reported failure on a healthy solve")
    all_ok = MPI.Allreduce(res.converged, &, COMM)
    all_ok || fail("try_stiffness_matrix converged flag not collective")
    results["try_stiffness_ok"] = (copy(res.K.nzval), copy(res.r))
end

# A recoverable failure on one rank must be reported by every rank, and the ranks
# that succeeded locally must receive RemoteAssemblyFailure instead of hanging
let (grid, dh, ch) = block3d(4)
    fem = create_assembler(FailOnRankZero(), dh, ch; quadrature_order=1)
    res = try_stiffness_matrix(fem, wobble(ndofs(dh)); dt=0.0)

    res.converged && fail("rank $RANK reported converged despite a recoverable failure")
    MPI.Allreduce(res.converged, |, COMM) && fail("converged flag is not collective")

    expected_remote = RANK != 0
    if (res.error isa RemoteAssemblyFailure) != expected_remote
        fail("rank $RANK got $(typeof(res.error)), expected remote=$expected_remote")
    end
    RANK == 0 && !(res.error isa RankZeroFailure) && fail("rank 0 lost its original exception")

    results["recoverable_failure"] = (Int(res.converged), Int(res.error isa RemoteAssemblyFailure))
end

if RANK == 0
    results["__nranks__"] = NRANKS
    serialize(OUTFILE, results)
end

MPI.Finalize()
