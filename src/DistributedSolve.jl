"""
    distributed_solve(K, residual, comm; verify_rtol=1e-8, workspace_pct=nothing) -> du

Solve `K * du = residual` with the distributed sparse direct solver [MUMPS](https://mumps-solver.org/), returning the update `du` on every rank.
It is a drop-in replacement for `du = K \\ residual` in an MPI Newton loop:

```julia
u .-= distributed_solve(K, residual, MPI.COMM_WORLD)
```

The method comes from a package extension.
Add MUMPS.jl to the environment and load it with `import MUMPS`, which is what defines the method.
Calling `distributed_solve` beforehand raises a `MethodError`.

Every rank passes its own copy of the assembled `K` and `residual`.
MUMPS factorizes across `comm` and broadcasts the solution back, so all ranks return the same `du`.
`residual` is left unchanged.

Before returning, `distributed_solve` checks `du` against `K * du ≈ residual` to a relative tolerance of `verify_rtol` and throws a [`DistributedSolveError`](@ref) when that check fails.
The check exists because MUMPS reported success on a cluster run whose solution was wrong.
It costs one sparse matrix-vector product per call (one per Newton iteration; on each rank's own copy and without synchronizing), which is negligible next to the factorization.
A working solve satisfies it to roughly machine precision.
Set `verify_rtol=0` to skip it.

MUMPS sizes its internal workspace from an estimate, and fails the factorization when that estimate turns out too small (`INFO(1) = -9`), the most frequent failure on the problems measured here.
`distributed_solve` retries that case automatically with progressively more workspace, so that a solve claims extra memory and completes instead of ending the run.
`workspace_pct` sets that percentage (MUMPS `ICNTL(14)`) explicitly and turns the retries off, which caps the memory a single solve may claim.

Use this in 3D above between roughly 4k..20k dofs.
With `K \\ residual` every rank builds the whole factorization, so time and memory both grow with the rank count, while MUMPS splits one factorization across the ranks instead.
Measured on the 72-core nodes of [NHR@FAU Fritz](https://doc.nhr.fau.de/clusters/fritz/) against the best replicated configuration (one rank × 72 cores, BLAS 36), on 3D hexahedral NeoHooke problems, where both methods produce the same solution to within 1e-9 relative:

| 3D grid | replicated `K \\ residual` (BLAS 36; 1 rank × 72) | `distributed_solve` (BLAS 1; 18 ranks × 4, 36 × 2 at 273k) |
|---|---|---|
| 2.2k dofs | 1.4 s | 1.2 s |
| 47k dofs | 37.7 s, 7.2 GB | **11.8 s, 1.5 GB** |
| 108k dofs | 145 s, 20.7 GB | **25.0 s, 2.1 GB** |
| 273k dofs | 634 s, 62.5 GB | **62.3 s, 3.4 GB** |

The two ways of solving cost the same per solve at about 3k dofs, and the 2.2k row covers eight repeats each whose ranges overlap, so the replicated solve is the more straightforward choice in the vicinity of the crossover.
In 2D, `K \\ residual` was faster at every size measured (291k: 52 s vs 62 s; 4.0M with `Hooke`: 30 s vs 39 s), both at BLAS 1, so use it there with fewer ranks than the 18 the distributed layout runs, since each rank's whole factorization copy must fit in node RAM.
At 16M 2D dofs one replicated rank completed using 95 GB of memory while 18 ranks running `distributed_solve` were killed on a 256 GB node: each of them holds its own 4.3 GB copy of `K` on top of its share of the factorization, so at that scale fewer ranks save more memory than splitting the factorization does.
[`recommended_solve_settings`](@ref) applies these rules to a given grid, keeping 18 ranks × 4 threads in 3D rather than the 36 × 2 layout the 273k row used.

Launch it the opposite way from the threaded `K \\ residual`, which runs one rank with all 72 cores as threads: `distributed_solve` takes its parallelism from MPI processes instead, so run many ranks per node (18 to 36 on 72 cores) with a single BLAS thread each.
At 18 ranks, 4 BLAS threads per rank measured 6.3× slower per solve than 1 thread.
[`recommended_blas_threads`](@ref) applies to the replicated solve.
The factorization communicates heavily between ranks, so these measurements all used the cluster's own MPI module rather than the build MPI.jl installs by default.
Use one node while the problem fits its RAM.
At 273k dofs the run took 102 s on 1 node against 123 s on 2 and 76 s on 4, all at 18 ranks per node, while 36 ranks on the single node took 62 s and beat every one of them.
At 985k dofs it took 377 s on 2 nodes, 708 s on 4 and 821 s on 8, the factorization itself accounting for 122 s, 233 s and 270 s per solve, so adding nodes bought no solve time at either size.
Element assembly does scale with the ranks, from 0.168 s to 0.064 s per Newton iteration at 273k dofs over 18 to 72 ranks, but that saving is negligible next to the factorization.
Furthremore, more nodes help only once the memory requires them.
The MPI tutorial covers layout, mesh-size, and memory scaling.
"""
function distributed_solve end

"""
    DistributedSolveError(phase, infog1, [relative_residual]) <: LocalAssemblyFailure

[`distributed_solve`](@ref) did not return a usable `du`.

`phase` is `:factorize` or `:solve` when MUMPS reported the failure itself, and `infog1` is then its `INFOG(1)`/`INFO(1)` error code (`-10` singular, `-13` allocation failure, `-9` workarray too small after every workspace retry has been exhausted).
`phase` is `:verify` when MUMPS reported success but `K * du` did not reproduce `residual`, and `relative_residual` is then how far apart the two were, relative to `residual`, while `infog1` is `0`.

`DistributedSolveError` subtypes [`LocalAssemblyFailure`](@ref), so a driver that shrinks `dt` and retries the step can catch it.
"""
struct DistributedSolveError <: LocalAssemblyFailure
    phase::Symbol
    infog1::Int
    relative_residual::Float64
end

DistributedSolveError(phase::Symbol, infog1::Integer) =
    DistributedSolveError(phase, Int(infog1), NaN)

function Base.showerror(io::IO, err::DistributedSolveError)
    if err.phase === :verify
        print(io, "DistributedSolveError: MUMPS reported success but the solution does not ",
              "satisfy K * du = residual (relative residual ", err.relative_residual, ")")
    else
        print(io, "DistributedSolveError: MUMPS ", err.phase, " failed with error code ",
              err.infog1)
    end
end