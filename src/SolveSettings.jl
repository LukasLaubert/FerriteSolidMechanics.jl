# Launch settings for the linear solve: BLAS threads, replicated-solve memory,
# and the per-node rank/thread layout. Measured on NHR@FAU Fritz; see docs/src/performance.md.

"""
    recommended_blas_threads(dh::DofHandler; ncores=Sys.CPU_THREADS, ranks_per_node=nothing) -> Int

Return the number of BLAS threads for the `K \\ r` solve the driver runs each Newton iteration.
Set it in the driver before solving: `LinearAlgebra.BLAS.set_num_threads(recommended_blas_threads(dh))`.

`recommended_blas_threads` returns `ncores` for a 3D grid whose rank has the node to itself, and `1` in 2D or when several ranks share a node.
`ranks_per_node` defaults to the number of ranks on this rank's node, read from MPI.
Julia starts BLAS at `Sys.CPU_THREADS ÷ 2`, so the returned count is higher in the 3D single-rank case and lower in the other two.

BLAS threading pays off in 3D when the rank owns the node: at 108k dofs the solve took 312 s at 1 BLAS thread against 145 s at 36.
In 2D it measured slower at every thread count.
Ranks sharing a node each start their own BLAS threads, which cost [`distributed_solve`](@ref) 6.3× per solve at 18 ranks.

[`recommended_solve_settings`](@ref) returns all launch settings at once.
[BLAS threads](@ref) states the measurements behind this rule, including the mesh size below which BLAS threading stops paying.
"""
function recommended_blas_threads(dh::DofHandler; ncores::Integer=Sys.CPU_THREADS,
                                  ranks_per_node::Union{Integer,Nothing}=nothing)
    Ferrite.getspatialdim(dh.grid) == 3 || return 1
    rpn = ranks_per_node === nothing ? mpi_ranks_per_node() : Int(ranks_per_node)
    return rpn > 1 ? 1 : Int(ncores)
end

# Ranks sharing this rank's node, via the shared-memory split.
function mpi_ranks_per_node()
    mpi_is_active() || return 1
    node = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, 0)
    n = MPI.Comm_size(node)
    MPI.free(node)
    return n
end

"""
    estimated_replicated_memory_gb(ndofs, dim) -> Float64

Estimate the memory one MPI rank needs, in GB, when the driver solves `K \\ r` directly.

Solving `K \\ r` builds a decomposition of `K` that is much bigger than `K` itself, and every rank builds its own copy.
That copy is what fills up a node: running 18 ranks needs 18 times this number.
[`distributed_solve`](@ref) splits one copy across the ranks instead, so it needs far less memory per rank.

Memory grows faster than the problem: `ndofs^1.43` in 3D and `ndofs^1.09` in 2D.
Doubling a 3D mesh's dofs therefore costs about 2.7× the memory, so a size that fits comfortably can stop fitting after one refinement.

The fitted curve is scaled to `NeoHooke` over three load steps, on linear hexahedra in 3D and linear quadrilaterals in 2D, and the predicted peak memory stays within 15% of every point measured on the 72-core nodes of [NHR@FAU Fritz](https://doc.nhr.fau.de/clusters/fritz/).
Higher-order elements were not measured; at equal dof count they change the sparsity of `K`, so remeasure before relying on the estimate for another element order.
An expensive material and a longer run both raise the peak above it, together by 2.6×, which is the headroom [`recommended_solve_settings`](@ref) budgets in its `fits` field.
Either the expensive material or the longer run reaches that budget on its own, and `VEVP_Zhao2021_AT` exceeded it at 9× the estimate, so take the memory of a long run, or of a model with many Newton iterations per load step, from its own measurement.

[Memory](@ref) documents the range of grid sizes the fit was validated over, the measured series behind the 2.6×, and the size above which each rank's own copy of `K` rather than the factorization sets the limit.
"""
function estimated_replicated_memory_gb(ndofs::Integer, dim::Integer)
    n = float(ndofs)
    return dim == 3 ? 1.60 + 1.0103e-6 * n^1.433 : 1.70 + 3.2597e-6 * n^1.086
end

# Headroom over the NeoHooke fit, from VEVP_MOAMMM at 47k dofs over 12 load steps: 17.1 GB
# needed against 6.6 GB estimated, an expensive material (1.52x at equal load steps) on a run
# four times longer (1.58x). VEVP_Zhao2021_AT measured 9x and stays above this budget.
const _MATERIAL_MEMORY_SPREAD = 2.6

# Fraction of a node's memory a job should plan to occupy, leaving the OS its share.
const _NODE_MEMORY_USABLE = 0.85

"""
    recommended_solve_settings(dh::DofHandler; cores_per_node=Sys.CPU_THREADS,
                               memory_per_node_gb=nothing, mumps_available=false,
                               assembly_bound=false) -> NamedTuple

Return the launch settings for a run on one node: which linear solver to use, how many MPI ranks to start, how many Julia threads to give each, and how many BLAS threads.

There are two ways to solve `K \\ r` each Newton iteration, and the choice is worth up to 5× here:

- `K \\ r` (`:replicated_lu`) – every rank solves the whole system by itself, so each pays the full memory of one solve
- [`distributed_solve`](@ref) (`:distributed_solve`) – the ranks split one solve between them, so each pays a share of the memory

`distributed_solve` needs `import MUMPS`; pass `mumps_available=true` once you have it.

`recommended_solve_settings` returns `(; solver, ranks_per_node, threads_per_rank, blas_threads, memory_gb, fits, note)`.
`memory_gb` is what one rank needs for `K \\ r`, from [`estimated_replicated_memory_gb`](@ref).
`fits` is `true` when all ranks together stay within 85% of `memory_per_node_gb`, budgeting 2.6× `memory_gb` per rank.
That budget is deliberately pessimistic and reports `false` before a run would actually fail; [Memory](@ref) explains the headroom behind the 2.6×.
`VEVP_Zhao2021_AT` measured 9× the estimate and exceeds even this budget, so size a model with many Newton iterations per load step from its own measurement.
`fits` is `missing` if `memory_per_node_gb` is not specified or if `distributed_solve` has been selected: In this case, the factorization is split across the ranks, so the replicated-copies budget checked by `fits` does not apply.
In 2D above a few million dofs also check that all ranks' copies of `K` fit in the node's memory, no matter the solver: every rank holds its own full copy, and at 16M dofs 18 such copies alone exceed the memory of a 256 GB node.

Pass `assembly_bound=true` for a 2D run that spends more time building element matrices than solving, which is typical of rate-dependent models that run their own local Newton solve per quadrature point.
The flag moves the 2D layout toward more ranks.
Raise the rank count further yourself if assembly dominates and the memory allows it; [Rank and thread layout](@ref) shows where the highest rank count stops fitting as the mesh grows.

```julia
s = recommended_solve_settings(dh; cores_per_node=72, memory_per_node_gb=256,
                               mumps_available=true)
LinearAlgebra.BLAS.set_num_threads(s.blas_threads)
# launch with s.ranks_per_node ranks and s.threads_per_rank threads each
```

The rules below were measured on the 72-core nodes of [NHR@FAU Fritz](https://doc.nhr.fau.de/clusters/fritz/):

| Case | Setting |
|---|---|
| 3D, MUMPS available, ≳4k dofs | `distributed_solve`, `cores ÷ 4` ranks × 4 threads, BLAS 1 |
| 3D, no MUMPS | 1 rank × `cores` threads, BLAS `cores` |
| 2D | `K \\ r`, `cores ÷ 8` ranks × 8 threads, BLAS 1 |

`recommended_solve_settings` holds the default 2D rank count at `cores ÷ 8` and, with `mumps_available=true` and `memory_per_node_gb` given, returns `distributed_solve` once that many copies would exceed 85% of the node's memory.
The rank and thread counts come from one mesh per case, so treat them as starting points and better time your own problem before committing to a large run.
Higher-order elements were not measured; at equal dof count they change the sparsity of `K`, so remeasure the thresholds and the memory behind `fits` before relying on them for another element order.
[Choosing the linear solver](@ref) states the timings behind these rules, and [Reproducing these on your cluster](@ref reproduce) shows how to check them on your own cluster.
"""
function recommended_solve_settings(dh::DofHandler; cores_per_node::Integer=Sys.CPU_THREADS,
                                    memory_per_node_gb::Union{Real,Nothing}=nothing,
                                    mumps_available::Bool=false, assembly_bound::Bool=false)
    dim = Ferrite.getspatialdim(dh.grid)
    n = ndofs(dh)
    cores = max(Int(cores_per_node), 1)
    mem_per_rank = estimated_replicated_memory_gb(n, dim)

    # A replicated factor per rank is the memory wall; 3D crosses over to MUMPS at ~3k dofs.
    replicated_3d_ranks = 1
    budget(ranks) = _MATERIAL_MEMORY_SPREAD * ranks * mem_per_rank
    # `assembly_bound` doubles the 2D rank count and so the memory, so the budget has to be
    # tested at the layout actually returned; testing another one lets the solver choice and
    # `fits` disagree, recommending a replicated solve that was already known to overflow.
    replicated_2d_ranks = max(assembly_bound ? cld(cores, 4) : cld(cores, 8), 1)
    use_mumps = mumps_available && (dim == 3 ? n >= 4_000 :
                                    memory_per_node_gb !== nothing &&
                                    budget(replicated_2d_ranks) > _NODE_MEMORY_USABLE * memory_per_node_gb)

    if use_mumps
        ranks = max(cld(cores, 4), 1)
        threads = max(cores ÷ ranks, 1)
        blas = 1
        note = dim == 3 ? "3D: distributed_solve at every size measured above ~4k dofs." :
               "2D above the node's memory budget: distributed_solve measured 1.3x slower at 291k dofs and 1.2x at 516k, but its solve fits."
    elseif dim == 3
        ranks, threads, blas = replicated_3d_ranks, cores, cores
        note = mumps_available ? "3D below the ~4k-dof crossover: the replicated solve is as fast and simpler." :
               "3D without MUMPS: one rank keeps a single factor and gives BLAS the whole node."
    else
        ranks = replicated_2d_ranks
        threads = max(cores ÷ ranks, 1)
        blas = 1
        note = "2D: the replicated solve wins at every size measured; BLAS threading does not pay in 2D."
    end

    fits = memory_per_node_gb === nothing || use_mumps ? missing :
           budget(ranks) <= _NODE_MEMORY_USABLE * memory_per_node_gb
    fits === false && (note *= " Budget $(round(budget(ranks); digits=1)) GB per node " *
                               "exceeds the safe share; load MUMPS, or use fewer ranks or more nodes.")
    return (; solver=use_mumps ? :distributed_solve : :replicated_lu, ranks_per_node=ranks,
            threads_per_rank=threads, blas_threads=blas, memory_gb=round(mem_per_rank; digits=2),
            fits, note)
end
