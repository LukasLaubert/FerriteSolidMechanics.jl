# Performance and parallel execution

This page covers the settings that decide how fast a simulation runs: which linear solver to use, how many MPI ranks and Julia threads to launch, and how many BLAS threads to use for solving.

## Recommendations

[`recommended_solve_settings`](@ref) returns the recommended solver configuration for a given grid and node:

```julia
s = recommended_solve_settings(dh; cores_per_node=72, memory_per_node_gb=256, mumps_available=true)
LinearAlgebra.BLAS.set_num_threads(s.blas_threads)
```

We recommend to verify critical thresholds on your hardware using the procedure in [Reproducing these on your cluster](@ref reproduce).
The following summarized recommendations are based on the measurements in the bottom part of this page.
The numbers below are for linear (Q1) elements; see [How measurements were taken](@ref) for scope.

- **Solver**: Use `distributed_solve` for 3D problems (faster at every size measured, up to 10.2× at 273k DOFs). Without MUMPS, `K \ residual` on a single rank remains viable in 3D. Use `K \ residual` for 2D problems, where it led at every size measured up to 4.0M DOFs; above ~500k its lead is 1.14× to 1.31× and available memory determines the choice. See [Choosing the linear solver](@ref).
- **Rank and thread layout**: For `distributed_solve`, use 18–36 MPI ranks with 2–4 Julia threads per rank. For `K \ residual` in 3D, use a single rank with all cores as Julia threads. In 2D with `K \ residual`, use 4–36 ranks for cheap materials, where all layouts are within 8% of each other, and 18–36 ranks for expensive materials, where the spread reaches 1.5×. See [Rank and thread layout](@ref).
- **BLAS threads**: Set to 1 in all configurations except single-rank 3D `K \ residual`, where BLAS threading pays (measured at 36 threads on a 72-core node). Multi-threaded BLAS under `distributed_solve` caused a 6.3× slowdown. In 2D, BLAS threading degrades performance at every thread count tested. See [BLAS threads](@ref).
- **Memory**: Under `K \ residual`, multiply [`estimated_replicated_memory_gb`](@ref) by the ranks per node and compare against node RAM, or read the `fits` field of [`recommended_solve_settings`](@ref), which does exactly that. Both are scaled to `NeoHooke` over three load steps; hence `fits` budgets 2.6× for a more expensive material on a longer run. An expensive model (`VEVP_Zhao2021_AT` measured 9×) or a long run (`NeoHooke` measured 2.5× over 48 load steps) reaches or exceeds that budget on its own, so measure yours before committing to a large job. See [Memory](@ref).
- **Large meshes**: In 2D, 16M DOFs solve on one node in 154 s at 95 GB, with the rank count lowered so the replicated copies fit. In 3D, the factorization outgrows the memory of 8 nodes somewhere above 1M DOFs. See [Large meshes](@ref).
- **Multi-node scaling**: Stay on a single node while memory permits; the best single-node layout beat every multi-node layout measured, at 273k and at 985k DOFs in 3D. Expand only when factorization memory exceeds single-node capacity. See [Scaling across multiple nodes](@ref).
- **Linear elastic models**: `Hooke` and `Hooke2D` are purely solve-bound (element matrices are assembled once). Use a single rank with BLAS multithreading. See [Assembly cost across material models](@ref).

The sections below present the measurements behind these recommendations.

## How measurements were taken

All benchmarks were measured on `FerriteSolidMechanics.jl v0.1.0` on the [NHR@FAU Fritz cluster](https://doc.nhr.fau.de/clusters/fritz/) (72 cores and 256 GB RAM per node, Open MPI 5.0.8, Julia 1.12, `Ferrite.jl v1.5.0`):
`NeoHooke` was used for 3D solver benchmarks, while `VEVP_Zhao2021_AD` was used for 2D benchmarks and assembly-cost comparisons.
Every grid uses linear (Q1) elements at `quadrature_order=2`: hexahedra in 3D, quadrilaterals in 2D.
Higher-order elements were not measured; at equal DOF count they change the sparsity of `K`, so timings and memory fits need recalibration (see [Reproducing these on your cluster](@ref reproduce)).
Runs cover three load steps unless stated otherwise.

Each figure is the median of its repeats, each submitted as a separate job and timed after a full warm-up Newton solve.
The solver tables below list each row's repeats in the repeats column, K \ residual first and distributed_solve second (split by a `/`); the remaining tables cover two or three repeats, and the 16M and 985k meshes ran once.
Most runs above 20 s varied by up to 8% across repeats; the small-grid measurements in the tables below varied by 18–45%, and more when one repeat was an outlier.
Hence, read those as indicative.
The multi-node layouts in [Scaling across multiple nodes](@ref) are the exception among the long runs; their spread is reported within the table.

Which solver is faster follows from solver architecture, and that ordering held across every problem size and material model measured, so the general rules likely apply to other clusters as well.
However, the exact problem size (DOF or element count) where one solver overtakes the other – and the exact memory required per rank – depend on your network hardware, BLAS build, and element type.
See [Reproducing these on your cluster](@ref reproduce) to measure it for your setup.

## Choosing the linear solver

Every rank assembles its own cells, then the tangent is reduced so that all ranks hold the full `K`.
On a single rank, whether or not MPI is initialized, that process assembles all cells and the reduction is skipped.
The two solvers differ in how the linear solve is performed:

- **`K \ residual`**: every rank solves the whole system independently, allocating memory for a complete matrix factorization.
- **[`distributed_solve`](@ref)** (MUMPS): the ranks split one solve between them, so each stores only a fraction of the factorization. Requires `import MUMPS`, and communicates between ranks during factorization.

In 3D, `distributed_solve` was faster at every size measured when comparing each solver in its optimal combination of MPI ranks and threads:

| 3D grid | `K \ residual` (1 rank × 72 threads, BLAS 36) | `distributed_solve` (18 ranks × 4 threads, BLAS 1) | `distributed_solve` faster by | repeats |
|---|---|---|---|---|
| 2.2k DOFs | 1.4 s | 1.2 s | 1.11× | 8 / 8 |
| 15k DOFs | 9.8 s | 5.8 s | 1.7× | 8 / 8 |
| 47k DOFs | 37.7 s | 11.8 s | 3.2× | 4 / 13 |
| 108k DOFs | 145 s | 25.0 s | 5.8× | 5 / 4 |
| 273k DOFs | 634 s | 62.3 s | 10.2× | 2 / 5 |

These are whole-run wall times, covering element assembly and every Newton solve.
Per solve, the 3D performance gain is larger (3.2× at 47k DOFs and 3.5× at 73k DOFs).
At 2.2k DOFs, `K \ residual` is faster per solve (0.107 s vs 0.131 s), but the multi-rank MPI layout completes element assembly faster than the single-rank threaded layout, making `distributed_solve` faster overall.
That 2.2k row covers eight repeats per solver whose individual runs overlap, 1.22 to 1.53 s replicated against 1.08 to 1.60 s distributed, so treat the ordering there as weak.
The 273k-DOF case ran fastest at 36 ranks × 2 threads, ahead of 18 ranks × 4 threads, which measured 81 to 105 s over three repeats.

Per solve, the two solvers cross at approximately **3k DOFs** in 3D.
Below that threshold, execution times are nearly identical, making `K \ residual` preferable due to its simpler setup without external dependencies.
In 2D, `K \ residual` was faster at every size measured:

| 2D grid | `K \ residual` | `distributed_solve` | `K \ residual` faster by | repeats |
|---|---|---|---|---|
| 8k DOFs | **2.0 s** | 4.6 s | 2.3× | 9 / 9 |
| 291k DOFs | **51.6 s** | 61.6 s | 1.19× | 9 / 5 |
| 516k DOFs | **96.0 s** | 112 s | 1.16× | 7 / 5 |
| 1.0M DOFs (`Hooke`, one load step) | **7.4 s** | 8.8 s | 1.19× | 2 / 2 |
| 2.0M DOFs (`Hooke`) | **52.6 s** | 59.7 s | 1.14× | 5 / 5 |
| 4.0M DOFs (`Hooke`, one load step) | **30.2 s** | 39.5 s | 1.31× | 2 / 2 |

The gap narrows with problem size, from 2.3× at 8k DOFs to 1.14–1.19× between 516k and 2.0M, and it stays in favour of `K \ residual` throughout.
The `Hooke` rows use the linear elastic model, whose element matrices are assembled once, so those runs spend almost all their time on the linear solve.
They also run at the rank count that keeps the total memory of the replicated copies within the node's RAM: 9 ranks at 1.0M DOFs and 4 at both 2.0M and 4.0M.
The 1.0M and 4.0M rows ran one load step and the 2.0M row the default three, so these rows are not a scaling series: 4.0M only looks faster than 2.0M because it ran a third of the steps.
The warm-up matters most at 4.0M: `K \ residual` reuses its symbolic factorization across Newton iterations and ran 30.2 s with the warm-up against 50.8 s without it, while `distributed_solve` re-analyses on every call and measured 39.3 s with the warm-up against 39.6 s without.
Above roughly 500k DOFs the difference is lower than a third either way, so memory consumption becomes the primary selection criterion.

**Rule of thumb**: use `distributed_solve` for 3D problems above ~4k..20k DOFs, and `K \ residual` for 2D problems, which led at every size measured up to 4.0M DOFs and was the only solver to complete at 16M (see [Large meshes](@ref)).

## Memory

Available RAM is a strict execution limit; jobs exceeding node memory are terminated by the operating system.

When using `K \ residual`, each rank constructs its own complete factorization of `K`, which requires significantly more memory than `K` itself.
Accumulated across ranks, these duplicate factorizations are the primary driver of node memory usage.

[`estimated_replicated_memory_gb`](@ref) predicts the peak memory required for a single factorization copy.
Factorization memory grows superlinearly.
Fitted to these measurements, it scales as $\mathcal{O}(\text{dofs}^{1.43})$ in 3D and $\mathcal{O}(\text{dofs}^{1.09})$ in 2D.
A sparse direct factorization on a regular grid is expected to scale as $\mathcal{O}(\text{dofs}^{4/3})$ in 3D and $\mathcal{O}(\text{dofs} \log \text{dofs})$ in 2D, so the exponents should transfer to other clusters even where the prefactors do not.
Doubling the DOFs of a 3D mesh increases memory requirements by approximately 2.7×, meaning a mesh refinement can easily exceed available node RAM:

| 3D DOFs | 47k | 108k | 273k | 556k |
|---|---|---|---|---|
| `K \ residual`, per rank | 7.2 GB | 20.7 GB | 62.5 GB | 179 GB |
| `distributed_solve`, per rank | 1.5 GB | 2.1 GB | 3.4 GB | – |

The predicted peak memory stays within 15% of every measured point, over 2.2k to 556k DOFs in 3D and 8k to 2.0M in 2D.
In 2D, the exponent $\mathcal{O}(\text{dofs}^{1.09})$ still holds an order of magnitude beyond that range, as [Large meshes](@ref) shows.

Two corrections translate the per-rank estimate into a per-node memory budget.

The estimate covers one factorization copy, and under `K \ residual` every rank holds its own.
`J2Plasticity` at 47k DOFs measured ~10 GB per rank, so 36 ranks would require ~360 GB and therefore not survive on a 256 GB node.

[`estimated_replicated_memory_gb`](@ref) is further scaled to `NeoHooke` over three load steps, and two factors raise the peak memory of a rank above it:
- The material, at the same number of load steps: measurements showed 0.7× (`Hooke`) to 1.5× (`VEVP_MOAMMM`) of `NeoHooke`, excluding `VEVP_Zhao2021_AT` at 9× (below). The models differ in history state per quadrature point and in Newton iterations per load step.
- The number of load steps: `NeoHooke` at 47k DOFs measured 7.1 GB over three load steps, 9.3 GB over six (1.3×), 11.0 GB over twelve (1.55×), 13.4 and 16.3 GB in two repeats at 24 steps, and 14.0 and 18.7 GB at 48. Peak memory keeps growing with the run, and repeats of the same run differ by up to 1.3×, because the recorded peak depends on when the garbage collector happens to run.

Together the two factors reach 2.6× in `VEVP_MOAMMM` over twelve load steps, which measured 17.1 GB against the 6.6 GB estimated, and [`recommended_solve_settings`](@ref) allows exactly this 2.6× in its `fits` field.
Either factor reaches that budget on its own: `NeoHooke` measured 2.5× its estimate over 48 load steps, and `VEVP_Zhao2021_AT` 9× over three, at 59.8 GB against the 6.6 GB estimated.
Based on the tests conducted, the two also stop compounding once the run is longer: at six and twelve load steps `VEVP_MOAMMM` measured 1.5× above `NeoHooke`, but at 24 load steps only 15.2 GB against 14.9 GB.
The budget is deliberately pessimistic: one estimate per rank marked configurations as fitting that were then killed for running out of memory.
Because of the variations found, measure the memory of a long run, or of a model with many Newton iterations per load step, directly.

For 2D systems, memory limits become relevant before 1M DOFs.
At 804k DOFs on 18 ranks, `K \ residual` exceeded 256 GB node RAM, whereas `distributed_solve` completed using 3.3 GB per rank.

**Above ~500k DOFs in 2D, total memory capacity determines solver selection.**
At that scale `K \ residual` measured at most 1.31× faster, so choose based on which solver fits in node RAM.

There are two ways to keep the memory within node RAM, and which one is better changes with problem size:
- **`K \ residual`**: every rank allocates its own full factorization copy (e.g., 18 ranks store 18 copies on the node), so lowering the rank count lowers the node's total memory in proportion.
- **`distributed_solve`**: splits one factorization across the ranks, so each rank's memory stays nearly constant as ranks are added, though the node's total memory still grows because each rank also keeps its own copy of `K` and its own MUMPS workspace: at 985k DOFs in 3D it measured 8.6 GB per rank on 36 ranks (2 nodes), 8.1 GB on 72 (4 nodes) and 7.9 GB on 144 (8 nodes), which is 310 GB, 585 GB and 1138 GB across the nodes.

That per-rank copy of `K` sets the ceiling in very large problems, where it is no longer negligible next to the factorization: at 16M 2D DOFs it is 4.3 GB per rank, so replicating it across 18 ranks already exceeds a node's RAM, while a single replicated rank completes the same problem (see [Large meshes](@ref)).
Below a few million DOFs the factorization dominates, so `distributed_solve` is the memory-efficient choice; above that, reducing the rank count saves more memory.

The 2D recommendation is therefore to use `K \ residual` while all rank copies fit in node RAM, lowering the rank count as the mesh grows, and to switch to `distributed_solve` when the factorization no longer fits.
To verify memory fit, evaluate [`estimated_replicated_memory_gb`](@ref) for your grid, multiply by the ranks per node, and check against total node RAM.
[`recommended_solve_settings`](@ref) performs this check automatically via the `fits` field, for the `K \ residual` path, and at the fixed `cores ÷ 8` rank count only – it does not try the other layouts measured above.

## Rank and thread layout

Given a fixed core count, MPI ranks parallelize element assembly significantly better than Julia threads.
Total element assembly time over a 2D run at 291k DOFs across a 72-core node illustrates this trend (a layout like `18 × 4` reads as 18 MPI ranks with 4 Julia threads each) (assembly is solver-independent, so these hold for both solvers):

| Layout | 1 × 72 | 4 × 18 | 9 × 8 | 18 × 4 | 36 × 2 |
|---|---|---|---|---|---|
| Assembly, `NeoHooke` | 22.1 s | 2.09 s | 0.65 s | 0.44 s | 0.45 s |
| Assembly, `VEVP_Zhao2021_AD` | 213 s | 43.8 s | 22.6 s | 12.0 s | 7.3 s |

Each MPI rank manages its own heap and garbage collection independently, whereas Julia threads within a process share a single heap and GC.

Conversely, `K \ residual` scales worse with more ranks because every rank repeats the full linear solve.
In 3D, where the linear solve dominates total runtime, this effect is critical (`J2Plasticity` at 47k DOFs, BLAS 1 throughout):

| Layout | 1 × 72 | 4 × 18 | 9 × 8 | 18 × 4 | 36 × 2 |
|---|---|---|---|---|---|
| `K \ residual` | 166 s | **148 s** | 167 s | 197 s | out of memory |
| `distributed_solve` | 88.9 s | 43.9 s | 36.9 s | 31.7 s | **27.1 s** |

Because the two solvers exhibit opposite scaling behavior, optimal rank-to-thread allocation depends on whether assembly or solving dominates:

| Situation | Layout on a 72-core node |
|---|---|
| `distributed_solve` | **18–36 ranks × 2–4 threads**, BLAS 1 |
| `K \ residual`, 3D | **1 rank × 72 threads**, BLAS 36–72 |
| `K \ residual`, 2D, cheap material | **4–36 ranks** × 2–18 threads, BLAS 1 |
| `K \ residual`, 2D, expensive material | **18–36 ranks × 2–4 threads**, BLAS 1 |

The 2D wall times at 291k DOFs across the same 72-core layouts (`K \ residual`) were:

| Layout | 1 × 72 | 4 × 18 | 9 × 8 | 18 × 4 | 36 × 2 |
|---|---|---|---|---|---|
| `NeoHooke` | 34.4 | 17.9 | 17.5 | 16.8 | **16.6** |
| `VEVP_Zhao2021_AD` | 251 | 82.3 | 65.2 | 51.6 | **42.8** |

Solve time varies by at most 1.4× across this sweep while assembly varies by 29×, so the assembly share drives the layout selection.
For a cheap material the four multi-rank layouts lie within 8% of each other, so any of them seems like a reasonable choice.

Each added rank also holds its own factorization copy, and that limit lies well inside the range the table above recommends.
The table below repeats the `VEVP_Zhao2021_AD` sweep on two further meshes, where memory becomes the limit:

| 2D DOFs | 9 × 8 | 18 × 4 | 36 × 2 |
|---|---|---|---|
| 130k | 30.1 s | 25.1 s | **21.3 s** |
| 516k | 127 s | **95.6 s** | out of memory |

At 130k DOFs the highest rank count is 1.18× ahead of 18 ranks × 4 threads, and at 516k the same layout is killed on a 256 GB node while 18 ranks run at 8.2 GB each.
Check memory before choosing the highest rank count: the maximum number of ranks that fit in node memory decreases as the mesh grows.

At small problem sizes, prefer having ranks over threads: at 1.5k DOFs, 4 ranks took 0.42 s compared to 1.20 s for 1 rank, whereas adding a second thread per rank raised the four-rank time to 0.76 s due to thread overhead and load imbalance.

## BLAS threads

[`recommended_blas_threads`](@ref) provides the recommended BLAS thread configuration:

```julia
LinearAlgebra.BLAS.set_num_threads(recommended_blas_threads(dh))
```

It returns `ncores` for 3D grids when running with a single rank per node, and `1` otherwise (detecting ranks per node via MPI).
The table below applies this rule to the three primary cases:

| Case | Returned BLAS threads | Evidence |
|---|---|---|
| 2D, any layout | 1 | 2D solve at 291k DOFs: 2.61 s (1 thread), 3.31 s (9), 3.38 s (18), 3.49 s (72) |
| 3D, one rank per node | `ncores` | 312 s (1 BLAS thread) → 145 s (36 BLAS threads) at 108k DOFs, 2.1× speedup |
| 3D, multiple ranks per node | 1 | no gain for `K \ residual` at 18 ranks, 15k DOFs (1.21 s vs 1.20 s per solve), and a large loss for `distributed_solve` (below) |

Some OpenBLAS builds carry a fixed thread limit and abort with `precompiled NUM_THREADS exceeded`, which happened here on `distributed_solve` runs launched at 72 BLAS threads, while `K \ residual` at 72 completed.
Cap BLAS threads at half the node core count if you encounter that error.

BLAS threading only pays on a sufficiently large 3D mesh: at 15k DOFs on one rank, the solve took 1.61 s at 1 BLAS thread against 1.46 s at 36 and 1.52 s at 72, whereas the table above shows a 2.1× speedup at 108k DOFs, so the gain appears somewhere between these two mesh sizes on the architecture tested.

In 2D, no thread count measured faster than 1 thread: solve time rises from 2.61 s at 1 thread to 3.49 s at 72 threads.
Julia defaults BLAS threading to `Sys.CPU_THREADS ÷ 2`, so 2D runs should set BLAS threads to 1 explicitly (e.g. via `LinearAlgebra.BLAS.set_num_threads(recommended_blas_threads(dh))`).

!!! warning "`distributed_solve` requires single-threaded BLAS"
    In a 3D benchmark at 108k DOFs with 18 ranks, increasing BLAS threads from 1 to 4 made `distributed_solve` **6.3× slower** (3.8 s vs 23.6 s per solve).

    `recommended_blas_threads` returns 1 whenever ranks share a node, which covers every `distributed_solve` layout.

## Assembly cost across material models

Assembly cost per Newton iteration spans a factor of 17 across the material models tested.
The table below lists that per-iteration cost alongside the number of Newton iterations per load step, which together determine the total assembly time over a run, for a unit cube under simple shear over three load steps (47k DOFs, 18 ranks × 4 threads):

| Material | `Hooke` | `NeoHooke` | `ArrudaBoyce` | `J2Plasticity` | `Ogden` | `VEVP_Zhao2021_AD` | `VEVP_MOAMMM` |
|---|---|---|---|---|---|---|---|
| Assembly per iteration | preassembled | 0.031 s | 0.030 s | 0.027 s | 0.39 s | 0.34 s | 0.46 s |
| Newton iterations per load step | 2 | 3 | 3 | 9 | 4 | 9 | 5 |

The applied shear is set per material, just below the load at which that model stops converging, so the inelastic models are in flow.
The iteration counts therefore describe each model under its own load case rather than a comparison at equal load, and the `VEVP_MOAMMM` column comes from a twelve-step run.

The `VEVP_Zhao2021_AT` model, not listed in the table, is the most expensive of those tested: on the same grid and layout it measured 5.9 s of assembly per iteration, 17× the `VEVP_Zhao2021_AD` value, and 670 s over three load steps.
Due to convergence issues with the AT model, the solve measurement was undertaken using a shear rate ten times lower than the AD model's.
Despite the lower shear rate, it still needs 28 Newton iterations per load step against 9 for `VEVP_Zhao2021_AD`; see [VEVP Zhao 2021](@ref "VEVP Zhao 2021") for the implementation differences.

At an equal core count, one rank with many threads assembles more slowly than many ranks with few threads, by a factor of about 10 in both of these cases:
`VEVP_Zhao2021_AD` costs 3.3 s per iteration at 1 rank × 72 threads against 0.34 s at 18 ranks × 4 threads, and `NeoHooke` 0.31 s against 0.031 s.
Threads in one process also share a single heap and garbage collector, which an allocation-heavy material pays for additionally: `VEVP_Zhao2021_AD` spent 70 s in garbage collection at 1 rank × 72 threads against 5.5 s per rank at 18 ranks × 4 threads.
[`recommended_solve_settings`](@ref) covers this with `assembly_bound`, which adds ranks in 2D.
In 3D, however, the solve still dominates, so the layout stays at one rank.

Linear elastic models (`Hooke` and `Hooke2D`) set `is_linear(material) = true`, allowing element stiffness matrices to be assembled once during initialization and reused, so the Newton loop performs no element assembly at all.
Because linear models are purely solve-bound, single-rank execution with BLAS multithreading is optimal (1 rank × 72 threads with BLAS multithreading took 17.4 s vs 35.7 s across 18 ranks at 47k DOFs in 3D).

## Large meshes

The measurements above cover 2.2k to 556k DOFs in 3D and 8k to 2.0M in 2D.
Runs beyond that range, to 16M DOFs in 2D and 985k in 3D, were conducted to check whether the recommendations scale along:

| Mesh | Solve | Layout | Wall time | Peak memory |
|---|---|---|---|---|
| 2D, 1.0M DOFs | `K \ residual` | 9 × 8, BLAS 1 | 7.4 s | 6.4 GB per rank |
| 2D, 4.0M DOFs | `K \ residual` | 4 × 18, BLAS 1 | 30.2 s | 21.9 GB per rank |
| 2D, 16.0M DOFs | `K \ residual` | 1 × 72, BLAS 36 | 154 s | 95 GB |
| 3D, 985k DOFs | `distributed_solve` | 36 × 4, 2 nodes | 377 s | 8.6 GB per rank |

Every row covers one load step, using `Hooke` in 2D and `NeoHooke` in 3D.
Wall time is essentially the linear solve: `Hooke` is preassembled, and assembly took 1.4 s of 377 s in the 3D row.
The rank count decreases along the 2D rows to keep the replicated copies within node RAM, so mesh size and rank count both change between them and their wall times do not measure scaling with mesh size alone.
Peak memory, by contrast, does extrapolate: the three 2D points imply an exponent of 1.083, against the 1.086 that [`estimated_replicated_memory_gb`](@ref) fits over meshes an order of magnitude smaller.

Four configurations did not complete, in every case for lack of memory:
- 2D at 16.0M DOFs with `distributed_solve`, at 18 ranks on one node and at 36 ranks across two. Each rank holds its own copy of `K`, and at that size 18 copies already exceed node RAM (see [Memory](@ref)).
- 3D at 4.0M and 16.0M DOFs with `distributed_solve` on 8 nodes. The fitted memory curve estimates their factorizations at 2.9 TB and 21 TB. Both runs reached 1998 GB and 2006 GB of the 2048 GB available, and the scheduler terminated them within five minutes.

In 3D the factorization therefore exceeds the memory of eight nodes somewhere above 1M DOFs, while a 2D mesh of the same DOF-size still fits on a single node.
Both solvers form an explicit factorization, so no rank or thread layout within `distributed_solve` overcomes this limit.
Only adding more nodes provides more memory, at the solve-time cost shown in [Scaling across multiple nodes](@ref).

## Scaling across multiple nodes

Use a single node while memory constraints allow.
For problem sizes fitting on one node, adding nodes costs wall time through inter-node network communication during `distributed_solve` factorization.
Each row is the median of three repeats, with the spread given as the ratio of the slowest repeat to the fastest:

| 3D, 273k DOFs, `distributed_solve` | Wall time | Spread |
|---|---|---|
| 18 ranks × 4 threads, 1 node | 102 s | 1.30× |
| 36 ranks × 2 threads, 1 node | **62 s** | 1.01× |
| 36 ranks × 4 threads, 2 nodes | 123 s | 1.15× |
| 72 ranks × 4 threads, 4 nodes | 76 s | 1.06× |

The best single-node layout outperformed every multi-node layout measured.
Furthermore, no individual run of any other layout was faster than the slowest 36 ranks × 2 threads run, at 62.3 s.

The same ordering was observed on a mesh four times larger: at 985k DOFs the run took 377 s on 2 nodes, 708 s on 4, and 821 s on 8, with the MUMPS factorization accounting for 122 s, 233 s, and 270 s per solve, respectively.

Only the linear solve grows with the node count.
Element assembly at 273k DOFs decreased from 0.168 s to 0.064 s per Newton iteration between 18 and 72 ranks, and the tangent reduction increased from 2.1 s to 4.7 s when a second node was added.
Multi-node execution is therefore recommended when the factorization exceeds single-node RAM, rather than as a means of shortening a run that already fits on one node.

## [Reproducing these on your cluster](@id reproduce)

The solver crossover can be located by timing the linear solve of a single Newton step, without running the full simulation.
The snippet below times that solve alone, so compare whole runs as well where assembly takes a large share of the run time.

Insert the benchmark snippet below into a script that initializes `dh`, `ch`, `assembler`, and `u` (such as `examples/mpi_four_point_bending.jl`).
The `distributed_solve` path requires MUMPS.jl (Unix-only); omit `import MUMPS`, the `distributed_solve` warm-up call, and the `t_dist` measurement to benchmark the `K \ residual` path alone.

```julia
using LinearAlgebra, MPI, Ferrite, FerriteSolidMechanics
import MUMPS                                  # distributed solver (Unix-only)

K, r = stiffness_matrix(assembler, u; dt=dt)  # executed by all ranks
apply_zero!(K, r, ch)

# Warm up JIT compilation before timing
K \ r
distributed_solve(K, r, MPI.COMM_WORLD)

# Timing
t_rep  = minimum(@elapsed(K \ r) for _ in 1:3)
MPI.Barrier(MPI.COMM_WORLD)
t_dist = minimum(@elapsed(distributed_solve(K, r, MPI.COMM_WORLD)) for _ in 1:3)

# Output measurements
MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println(
    ndofs(dh), " dofs, ", MPI.Comm_size(MPI.COMM_WORLD), " ranks: ",
    "replicated ", round(t_rep; digits=2), " s; distributed ", round(t_dist; digits=2), " s")
```

Execute the benchmark script in two distinct launches to evaluate each solver in its optimal layout:

```sh
# Replicated path: single rank, BLAS threads enabled -> record t_rep
mpiexec -n 1 julia --project=. --threads=72 bench.jl

# Distributed path: multiple ranks, single-threaded BLAS -> record t_dist
OPENBLAS_NUM_THREADS=1 mpiexec -n 18 julia --project=. --threads=4 bench.jl
```

Calling `BLAS.set_num_threads(recommended_blas_threads(dh))` within the script automatically sets appropriate BLAS thread counts for both execution modes (assuming our cluster benchmarks also hold on your hardware).
We suggest repeating this measurement across three mesh sizes spanning an order of magnitude; the performance threshold is the mesh size where `t_rep` and `t_dist` intersect.
Repeat each measurement in a separate job submission rather than several times within one, so that node-to-node differences show up in the spread.
`distributed_solve` also pays one extra factorization whenever MUMPS retries with more workspace.
In a 2D case with four retries over sixteen solves, that extra factorization cost 15% more time per solve.

To calibrate the memory model, compare `estimated_replicated_memory_gb(ndofs(dh), dim)` against the peak memory usage reported by your job scheduler (e.g., `MaxRSS` in SLURM's `sacct` output) for a single-rank run.
Multiply the single-rank memory usage by the desired ranks per node to determine total node memory demands.
Repeat this check after changing the material model, since a more expensive one raised the memory by up to 1.5× in our benchmarks (`VEVP_Zhao2021_AT` reaches 9×, see [Memory](@ref)).
Repeat it as well after lengthening the run: over the series in [Memory](@ref), `NeoHooke` at 47k DOFs rose from 7.1 GB over three load steps to 11.0 GB over twelve (1.55×) and to 14.0–18.7 GB over 48.

The [MPI four-point bending](@ref) tutorial is the 2D problem measured on this page, and it includes the full runnable script.
