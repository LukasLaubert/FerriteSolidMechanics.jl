# Worker for test_mumps_solve.jl; spawned under mpiexec, not run by the suite.
# Solves a small replicated 3D system with `distributed_solve` and checks the
# residual on every rank; serializes the update for the parent to compare across
# rank counts. `import MUMPS` activates the FerriteSolidMechanicsMUMPSExt extension.

using MPI
MPI.Init()

import MUMPS
using Ferrite
using FerriteSolidMechanics
using LinearAlgebra
using SparseArrays
using Serialization

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)
const OUTFILE = ARGS[1]

fail(msg) = (RANK == 0 && println("FAIL ", msg); MPI.Finalize(); exit(1))

# Left face clamped and right face pushed, so K is nonsingular after constraints.
grid = generate_grid(Hexahedron, (5, 5, 5))
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]))
add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.01], [1]))
close!(ch)

fem = create_assembler(NeoHooke(100.0, 0.3), dh, ch; quadrature_order=2)
u = zeros(ndofs(dh))
apply!(u, ch)
K, r = stiffness_matrix(fem, u; dt=0.0)
apply_zero!(K, r, ch)

du = distributed_solve(K, r, COMM)

# K and r are replicated, so every rank can check that du solves the system.
resid = norm(K * du - r) / max(norm(r), 1.0)
maxresid = MPI.Allreduce(resid, max, COMM)
maxresid < 1.0e-8 || fail("distributed_solve residual too large: $maxresid")

# A singular K must throw rather than return garbage. Which of the two guards fires
# (status codes or backward error) is not asserted, only that one of them does.
Ksing = copy(K)
Ksing[:, 1] .= 0.0
Ksing[1, :] .= 0.0
dropzeros!(Ksing)
singular_threw = try
    distributed_solve(Ksing, r, COMM)
    false
catch err
    err isa DistributedSolveError || fail("singular K threw $(typeof(err)), not DistributedSolveError")
    true
end
singular_threw || fail("singular K returned a du instead of throwing")

# Every rank must agree, or a survivor would hang the next collective.
MPI.Allreduce(singular_threw, &, COMM) || fail("ranks disagreed on the singular-K failure")

# verify_rtol=0 opts out of the backward-error check on a healthy solve.
du_noverify = distributed_solve(K, r, COMM; verify_rtol=0)
norm(du_noverify - du) == 0.0 || fail("verify_rtol=0 changed the solution")

# Starving the workarray reproduces the cluster's INFO(1) = -9. MUMPS may cope at this
# size, so the assertion is the retry contract: the default schedule must solve regardless.
starved_threw = try
    distributed_solve(K, r, COMM; workspace_pct=1)
    false
catch err
    err isa DistributedSolveError || fail("starved workspace threw $(typeof(err))")
    true
end
RANK == 0 && println("@@info,name=workspace_starved,threw=$starved_threw")
norm(distributed_solve(K, r, COMM) - du) == 0.0 || fail("workspace retry did not recover")

# A throwing solve must free its instance, or the GC finalizes it after MPI_Finalize and
# MUMPS aborts the process. No @test sees that, so the worker's exit code is the assertion.
for _ in 1:3
    try
        distributed_solve(Ksing, r, COMM)
    catch err
        err isa DistributedSolveError || fail("singular K threw $(typeof(err))")
    end
end
norm(distributed_solve(K, r, COMM) - du) == 0.0 || fail("a failed solve disturbed the next one")

if RANK == 0
    serialize(OUTFILE, Dict("nranks" => NRANKS, "du" => du, "maxresid" => maxresid))
end
MPI.Finalize()
