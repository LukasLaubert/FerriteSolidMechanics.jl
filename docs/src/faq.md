# FAQ

This page summarizes characteristics of the package in a Q&A format, specifically regarding the dimensionality wrappers, the `alpha_value` extension point, and the assembled output.
If you have questions that are not addressed here, check the [Concepts](concepts.md) page or file an issue.

## General

### Does the assembler run a Newton loop for me?

No. The assembler returns `(K, r)` from `(try_)stiffness_matrix`; the user writes the convergence test and the `u .-= K \ r` update.
The tutorials demonstrate the typical Newton loop.
The [Concepts](concepts.md) page details the exact scope of the assembler.

### What does `dt` actually do?

The `dt` keyword is the load step time increment, using the unit convention of your material parameters.
The assembler (invoked via `(try_)stiffness_matrix`) uses it to advance trial history.
Assembly semantics are model-dependent:

- For **rate-independent** materials (see the [Stable models](models/index.md#Stable-models) table), `dt` has no effect on the tangent.
  Recommendation: Pass the actual load step size anyway; if you ever swap a model for a rate-dependent one, your code does not need to change.
- For **rate-dependent** materials, `dt` influences the local update (e.g. governing relaxation, creep, or viscoplastic flow).
  Setting `dt = 0.0` freezes the rate-dependent state evolution; generally, this does not yield a rate-independent material response.

### Does `compute_stresses` advance my material state?

No for the bundled models. Use `(try_)stiffness_matrix` (or `compute_forces`) in the Newton loop to write trial state, call `update_states!` once after a converged load step, then call `compute_stresses` whenever you need stress
output.
`compute_stresses` reports the stress corresponding to the current assembler state (whether trial or committed) without advancing internal variables.
The postprocessing hooks (`material_stress`, or the lower-level `_compute_stress_qp`) receive `dt`, but leave it at the default `dt = 0.0`: they must not advance state, and are not a substitute for `(try_)stiffness_matrix` or `update_states!`.
However, passing different values for `dt` can be used as an advanced extension point.

### When do I call `update_states!`?

Once per load step, **after** Newton has converged, **before** moving on to the next step.
Do not call it inside the Newton loop.
For stateless material models the call is a no-op.

### How do I reject a load step?

The bundled models do not reject steps themselves.
If you implement line search or adaptive time stepping in your outer loop and want to roll the material state back, call `revert_states!(assembler)`, which iterates over all quadrature point states and calls `revert_state!`.

The `PlaneStress` wrapper additionally remembers `F33_previous`, and the underlying `revert_state!` method on `PlaneStressStateWrapper` restores it.

### How do I reduce `dt` after local material nonconvergence?

Use `try_stiffness_matrix(assembler, u; dt)` in the Newton loop when the outer solver should treat local constitutive failures as a rejected step.
It returns `(converged=false, K=nothing, r=nothing, error=err)` for failures that subtype `LocalAssemblyFailure`, such as `VEPD_Detrez2010ConvergenceError`, `VEVP_MOAMMMConvergenceError`, and `PlaneStressConvergenceError`.
Other exceptions are rethrown.

On `converged=false`, restore the material states with `revert_states!(assembler)`, restore `u` to the last accepted displacement, reduce `dt` using `dt = reject_step!(controller, dt)` or an equivalent policy, update the boundary conditions for the smaller trial step, rebuild any time-dependent external loads, and try again.
Do not call `update_states!` until the retried step has converged.
Use `TimeStepController` or an equivalent `dt_min` guard and rejection budget so a permanently failing step cannot retry forever.
`TimeStepController()` defaults to a practically unbounded rejection budget, so pass a finite `max_rejections` when exhaustion should be enforced.
See [Adaptive time stepping and recoverable local failures](@ref) for a full solver-loop skeleton.

In MPI runs, `try_stiffness_matrix` synchronizes failure states across all ranks to prevent deadlocks.
Unrecoverable errors are thrown collectively.
If a recoverable failure occurs on any rank, all ranks safely abort assembly and return `converged=false`.
If the original failure happened on another rank, `result.error` is a `RemoteAssemblyFailure` marker on the ranks that did not fail locally.
All ranks must then reject the step collectively.

## Plane strain and plane stress

### What is the difference between `Hooke2D` and `PlaneStrain(Hooke)`?

`Hooke2D` is a direct 2D analytical form – it takes `(E, ν)` and builds the appropriate 2D stiffness tensor (plane strain by default; `plane_stress=true` for the analytical plane stress matrix).
It has no internal state and uses `is_linear == true`.

`PlaneStrain(Hooke)` wraps the 3D `Hooke` and reduces it to plane strain by embedding the 2D gradient into 3D (`F̄₃₃ = 1`), calling `compute_PK1_3D`, and extracting the in-plane part.
The wrapper always reports `is_linear == false` because the generic element path is unaware that the wrapped model
is in fact linear.

For a linear problem, hence prefer using `Hooke2D`.
It avoids the AD overhead of the `PlaneStrain` wrapper and the assembler preassembles its stiffness into the static `K_linear`.

### Why is `PlaneStress` always treated as nonlinear?

Because it has to solve for the out-of-plane stretch `F̄₃₃` at every quadrature point such that the 3D PK1 satisfies `P̄₃₃ = 0`.
The local Newton iteration runs inside the wrapper's `material_response` at every quadrature point, with configurable controls (`PlaneStress(model; maxiter=20, tol=1e-10)`).
The wrapper therefore reports `is_linear == false` even when the wrapped 3D material is linearly elastic.
Consequently, `K_linear` is never filled for cells using `PlaneStress`.

If the local Newton solve does not converge, or if the local derivative needed for the Newton update or tangent condensation is too small, assembly raises a `PlaneStressConvergenceError` instead of committing a non-converged plane stress state.
Use `try_stiffness_matrix` when an outer load step controller should receive a failure flag and reduce `dt` instead of handling an exception.

### `PlaneStress` is slow. What can I do?

The practical options to speed up the computations are:

1. **Loosen the local Newton tolerance.** You can configure the local Newton solver via `PlaneStress(model; tol=1e-6)`.
   A looser tolerance requires fewer iterations per quadrature point, which speeds up the evaluation at the cost of satisfying the out-of-plane zero-stress condition less accurately.
2. **Implement a direct 2D analytical formulation.** If the plane stress equations for your material can be derived analytically, write a custom 2D material model (e.g. `My2DMaterial <: AbstractMaterial`) that directly evaluates the 2D stress and tangent.
   This bypasses the numerical condensation and is compatible with the assembler.

### My stress output looks like it dropped/increased a factor. What happened?

Two possible causes:

1. **`alpha_value` is not 1.** If you supplied a non-`nothing` `ah` to `create_assembler` and the alpha values at some QPs are 0 or very small, the stress at those QPs is silently scaled.
   Inspect `alpha_value(av, qp)` for the QPs in question.
2. **`thickness` mismatch.** The element residual and stiffness are multiplied by `thickness` (passed as a keyword argument to `create_assembler`, default is `1.0`).
   If you pass a value other than `1.0`, your reaction forces (which are derived from the residual) are scaled accordingly.
   `thickness` acts as a geometric multiplier during the 2D volume integration to compute forces, while Cauchy stress is a physical property independent of plate thickness.
   Therefore, `compute_stresses` always returns the true physical stress, while the integrated nodal reaction forces scale with `thickness`.

## The `alpha_value` extension point

### What is `alpha_value`?

`alpha_value` returns a scalar for one quadrature point, which the generic element routine multiplies into the integration weight:

```julia
α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)
```

The package defines a single fallback method `alpha_value(::Nothing, ::Int) = 1.0`.
The assembler creates an `alphavalues` object for every `CellValues` when (and only when) the user passes a non-`nothing` `ah` to `create_assembler`.
The provided extension hook is

```julia
FerriteSolidMechanics.create_alpha_values(ah::MyAlphaSource, cv)
```

The fallback also looks for an `AlphaValues(ah, cv)` constructor in `ah`'s defining module.
The returned object is reinitialized with `Ferrite.reinit!(av, cellid)` before each cell is assembled or postprocessed.

### What is it *for*?

The intended use is **quadrature point stiffness scaling**, such as coupling to external scalar fields or reducing the material stiffness by a fixed scalar factor at each quadrature point.
Element routines multiply `α` into the residual and element stiffness tangent, and `compute_stresses` applies the same factor to the postprocessed stress.

## Multithreading

### How do threads come into play?

FerriteSolidMechanics automatically runs element assembly in parallel across all available Julia threads.
You do not need to change any code to enable this. Simply start your script with multiple threads (e.g., `julia -t 4` or by setting the `JULIA_NUM_THREADS=4` environment variable).

When you call `(try_)stiffness_matrix`, the package distributes the nonlinear cells across threads using a dynamic task scheduler (`OhMyThreads.@tasks`).
The element matrices are computed in parallel and then safely scattered into the global sparse stiffness matrix `K`.
Similarly, the state commit phase (`update_states!`) is also parallelized across threads.

The generic assembler handles all thread-safety internally by pre-allocating dedicated workspaces and cell buffers for each thread during `create_assembler`.

## Hooking up MPI

### Do I have to do anything to enable MPI?

No.
The assembler automatically detects whether MPI is active.
If your script calls `MPI.Init()`, the per-rank results are summed across `MPI.COMM_WORLD` after every `(try_)stiffness_matrix` and `compute_stresses` call.
If `MPI.Init()` has not been called, the MPI reductions are no-ops and the package runs perfectly sequentially without throwing errors.
If your MPI run fails during initialization or reports one rank per process, see [MPI troubleshooting](@ref).

### Does MPI parallelize the linear solve?

Not by default, but [`distributed_solve`](@ref) does.
FerriteSolidMechanics partitions the element-local assembly work, including the constitutive updates at quadrature points.
After assembly, the full sparse matrix and residual are available on every rank.
By default, every rank then performs the same replicated solving step (e.g., `K \ residual`) in the outer Newton loop, so MPI accelerates the constitutive assembly while each rank builds the whole factorization.
For this replicated path, `LinearAlgebra.BLAS.set_num_threads(recommended_blas_threads(dh))` speeds up the solve in 3D problems.

[`distributed_solve(K, residual, MPI.COMM_WORLD)`](@ref distributed_solve) replaces `K \ residual` and splits a single MUMPS factorization across the ranks.
The method comes from a package extension: add MUMPS.jl to the environment and load it with `import MUMPS`, otherwise the call raises a `MethodError`.
Measured on 3D hexahedral `NeoHooke` at 108k DOFs, `distributed_solve` on 18 ranks × 4 threads took 25 s and 2.1 GB per rank against 145 s and 20.7 GB for the replicated solve on 1 rank × 72 threads.
The two cost the same per solve at about 3k DOFs in 3D, and in 2D the replicated solve is faster up to roughly 1M DOFs.
[`recommended_solve_settings`](@ref) applies these rules to a given grid; [Performance and parallel execution](performance.md) and the [MPI four-point bending](tutorials/mpi_four_point_bending.md) tutorial cover the measurements and the rank, thread, and BLAS layout for each solver.

### How are cells partitioned across ranks?

Via `mod(cellid - 1, nranks) == rank`.
This is a static, round-robin partition by cell id.
There is no load balancing: if some cells are much more expensive than others, the computational load per rank may be imbalanced.
The static partition guarantees that a specific cell is always processed by the exact same rank across all assembly calls throughout the entire simulation, preserving its local history state.

## Output

### What is the output of the `compute_stresses` function?

The `compute_stresses` function returns the assembler's reusable stress cache matrix of size `(max_nquadpoints, ncells)`.
The value `max_nquadpoints` is the maximum number of quadrature points across the grid, and `ncells` is the total number of cells.
The second dimension of the returned cache matrix corresponds to the Ferrite cell ID.
The elements of this matrix are of type `Tensor{2,2,Float64,4}` in 2D and `Tensor{2,3,Float64,9}` in 3D.
Because it's a reusable cache, subsequent `compute_stresses` calls overwrite it; use `copy(stresses)` if you need to retain the result.

### What are the dimensions of the global stiffness matrix K?

The `(try_)stiffness_matrix` function returns a `SparseMatrixCSC{Float64,Int}` representing the global stiffness matrix `K`.
`K` has dimensions of `ndofs(dh) × ndofs(dh)`.
Its sparsity pattern is dictated by the `ConstraintHandler`.
Constrained degrees of freedom (DOFs) are allocated, while their corresponding rows and columns should be zeroed out by `apply_zero!` (see e.g. the [Plate with hole](tutorials/plate_with_hole.md) tutorial).

## Performance

### Should I use a `Vector{Int}` or `Int` for `quadrature_order`?

A scalar `Int` is broadcast to every sub-dof-handler.
A `Vector{Int}` assigns the specified quadrature orders directly to the sub-dof-handlers in declaration order.
For example, in mixed-element grids (e.g., quads + triangles), you can use the vector form to set a different quadrature order for each element type.

### Why is my first Newton step slow?

This is almost entirely due to Julia's just-in-time (JIT) compilation.
When you run `(try_)stiffness_matrix` for the first time, Julia compiles the element routine.
This compilation cost is paid only once; subsequent Newton steps will be significantly faster.
To benchmark your assembly, warm up with a `(try_)stiffness_matrix(assembler, u0)` call on a zero `u0` before timing.

### Does the assembler store simulation state?

Yes.
The assembler holds the material's history state (e.g., plastic strain) for every cell.
During Newton iterations, `(try_)stiffness_matrix` updates a temporary trial state, and `update_states!` commits the converged state for the next load step.
If you need to start a completely fresh simulation, you must build a new assembler to reset this history.

### Are `Plots` and `WriteVTK` required?

No.
They are optional example-output dependencies.
Core assembly, material models, tests, and stress postprocessing do not use `Plots.jl` or `WriteVTK.jl` directly.
Install `Plots.jl` before calling `plot_results`, and install `WriteVTK.jl` before running tutorials that export data, such as the [DMA tutorial](tutorials/cantilever_dma.md).