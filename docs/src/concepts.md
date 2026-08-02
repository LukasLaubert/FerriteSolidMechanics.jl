# Concepts

This page explains the structure of the assembler, the behavior of public functions, and the interface design.
We recommend reading this page before implementing a custom material.

## The `GenericMaterialAssembler`

The core architecture is implemented in `src/GenericMaterialAssembler.jl`.
The function `create_assembler` returns a `GenericMaterialAssembler` struct, which stores and preallocates core buffers to minimize allocations during the Newton solve:

| Field (selected) | Purpose |
| --- | --- |
| `K_linear` | Preassembled global linear stiffness matrix (only nonzero if any material has `is_linear == true`) |
| `K_tangent` | Preallocated buffer for the global tangent stiffness matrix (refilled every call to `stiffness_matrix`) |
| `r` | Preallocated buffer for the global internal residual vector |
| `materials`, `cell_to_mat_idx` | The list of materials and the mapping `cellid → material index` |
| `linear_cells`, `nonlinear_cells` | Sorted partitions of cell ids by `is_linear(material)` |
| `owned_nonlinear_cells` | The subset of `nonlinear_cells` assigned to this MPI rank (`mod(cellid-1, nranks) == rank`) |
| `states` | One quadrature point state vector per owned nonlinear cell (entries for linear or non-owned cells are unassigned) |
| `dh`, `ch`, `ah` | The closed Ferrite `DofHandler`, `ConstraintHandler`, and optional custom alpha-value source |
| `quadrature_orders` | Per-sub-dof-handler quadrature order |
| `thickness` | Scalar multiplier applied to the local element stiffness `ke` and residual `re` in the element loop |
| `_sdh_*` | Per-sub-dof-handler metadata: cell-to-sdh index, `CellValues`, `nquadpoints`, `ndofs_per_cell`, owned-cell indices |
| `_cell_ke`, `_cell_re` | Per-owned-cell caches for the local element stiffness matrices and residual vectors, used by the two-phase assembly |
| `_stresses_cache`, `_stresses_mpi_buffer` | Reusable output buffers for `compute_stresses` (callers should `copy` the returned stress field before the next evaluation) |

### Task-local workspaces

The sparse matrices, per-cell state, stress cache, task-owned workspaces, and per-cell result buffers are created in `create_assembler`.
The parallel loops hand each spawned task a reference to one of these preallocated workspaces for element-local scratch storage.
Individual materials can allocate inside their `material_response` (or custom `_assemble_element!`) methods; [`allocate_material_cache`](@ref) provides a per-element scratch object to reuse buffers across quadrature points.
This is particularly relevant when using automatic differentiation or temporary tensor objects.

### Multi-material cellset coverage

When `create_assembler` is called with a dictionary of materials (`Dict{String, <:AbstractMaterial}`), the string keys must correspond to named cellsets in the underlying grid.
These cellsets must collectively cover all active cells of the `DofHandler` exactly once.
Inactive grid cells within those cellsets are simply ignored.
However, active cells that are missing entirely, or assigned to multiple material cellsets, will trigger an error during assembler construction.

For example:
```julia
# The grid has cellsets defining the two domains
addcellset!(grid, "steel", Set(1:100))
addcellset!(grid, "rubber", Set(101:200))

# The dictionary keys match the cellset names exactly
materials = Dict(
    "steel"  => PlaneStrain(J2Plasticity(200e3, 0.3, 250.0, 1e3)),
    "rubber" => PlaneStrain(MooneyRivlin(1e3, 0.2e3, 10e3))
)

assembler = create_assembler(materials, dh, ch)
```

## Linear and nonlinear cells

Each material reports `is_linear(material)`.
Linear materials have a displacement-independent tangent.
Their cell IDs are collected in the assembler's `linear_cells` vector, and the element stiffness is precomputed once during `create_assembler`.
On every subsequent `stiffness_matrix` call, the linear part is added to the tangent without recomputation.

```text
K_tangent = K_nonlinear(u) + K_linear
r         = r_nonlinear(u) + K_linear * u
```

Nonlinear materials include any model with internal state, history dependence, or displacement-dependent stiffness.
The cells assigned to nonlinear materials are stored in `nonlinear_cells`, while the corresponding element tangent and residual are recomputed during every Newton iteration.
The quadrature point state is owned and updated by the rank hosting the cell.

Every `stiffness_matrix` call on a nonlinear cell runs `_assemble_element!`.
Inside the element routine, the material model is evaluated at every quadrature point.
If this involves a return mapping, local Newton solve, or AD tangent, this cost is paid on each global Newton iteration.
The values written during this call are trial values; they are committed only later through `update_states!`.

Materials normally do not implement `_assemble_element!` themselves: they implement the quadrature point hook `material_response(material, ε_or_F, state, dt, cache)` and declare their strain measure via `kinematics(material)` (`SmallStrain()` or `FiniteStrain()`).
A generic element routine then owns the quadrature loop, the integration weights (including alpha scaling), the weak-form contractions into `ke` (element stiffness) and `re` (element residual), and the trial-state bookkeeping (`set_trial!`).
`_assemble_element!` remains the stable element-level extension hook for formulations whose element structure is material-specific – the bundled `VEVP_Zhao2021_AT` assembles its hand-derived tangent from nodal shape-gradient blocks and is the canonical example.
Material-specific kinematics, state updates, wrapper condensation, and local solves stay in the material's `material_response`, while all repeated finite element algebra is shared.

## The element loop in two phases

The element assembly orchestrated by `stiffness_matrix` (and internally `_assemble_stiffness_local!`) consists of **two phases** to parallelize the element computations while keeping the global matrix assembly sequential:

1. **Phase 1** (`_stiffness_sdh_batch!`): each task computes (elementwise) `ke` and `re` into a task-local workspace, then copies them into `_cell_ke[idx]` / `_cell_re[idx]`.
2. **Phase 2**: a single sequential loop scatters every `_cell_ke[idx]`, `_cell_re[idx]` into the global matrix and vector via `assemble!`.

Phase 1 is parallelized across threads using `OhMyThreads.@tasks`, and the element workload is distributed across MPI ranks.
See the [Threading and MPI](#Threading-and-MPI) section below for implementation details.

## State management

Material models with internal history implement a trial/commit pattern on their state object.

```text
            [Start Newton Iteration] ◄──────────────┐
                        │                           │
                        │ (try_)stiffness_matrix    │
                        │  ↳ material_response      │
                        ▼                           │
             [Trial / Current State]                │
                 ¦             ▲                    │
   update_state! ¦             ¦ revert_state!      │
(step converged) ¦     OR      ¦ (step failed) ─────┤ (retry step with smaller dt)
                 ▼             ¦                    │
          [Committed / Previous State] ─────────────┘ (next load step)
```

- `create_state(mat)` returns a new quadrature point state.
  For history-dependent materials it initializes trial and committed history to the same values.
- During a Newton step, assembly (called by `(try_)stiffness_matrix`) writes trial values into the current fields at every quadrature point – the generic element routine stores the trial returned by `material_response` via `set_trial!`; custom `_assemble_element!` methods write it directly.
  The committed state is never modified by assembly, so it is always safe to revert from if a step fails.
- If the step converges, the outer loop calls `update_states!(assembler)`, which dispatches to `update_state!` on every quadrature point state of every owned nonlinear cell, and with that commits the trial state.
  The assembler inspects the return value: if the material model's `update_state!` returns `nothing` (the standard for `mutable struct` states), it assumes the state was updated in-place.
  If it returns a new `AbstractMaterialState` (required for immutable `struct` states), the assembler overwrites the old state in the state vector with the new one.
- If the step must be rejected, `revert_states!(assembler)` restores the last committed history for all owned states.
  This wipes out all values left in the trial state from the failed iterations, ensuring the next attempt (e.g. with a smaller `dt`) starts from the previously committed state.
- `compute_stresses` is postprocessing.
  Stateful materials read from their current/trial state fields to calculate the final stress, without mutating or advancing the history.

The shape of the trial and committed history is up to the material.
Some models use `current`/`previous` structs (e.g. `J2PlasticityState`, `VEVP_Zhao2021_ATState`, `VEVP_Zhao2021_ADState`); others use named fields such as `current_Fp`/`previous_Fp` (`VEPD_Detrez2010`) or `current_Fvp`/`previous_Fvp` (`VEVP_MOAMMM`).
`update_state!`/`revert_state!` implementations typically delegate to `copy_state!(dest, src)` for the actual copy (which evaluates to a value assignment for immutable types or a deep copy for arrays).
This bundled `FerriteSolidMechanics.copy_state!` is a default fallback that handles scalars, immutable types, and arrays recursively.

## Threading and MPI

- **Threading.** `OhMyThreads.@tasks` with a `:greedy` scheduler is used to parallelize the stiffness and stress element loops.
  `OhMyThreads.@local` binds each spawned task to a unique preallocated `AssemblyWorkspace`, so task migration in Julia 1.10/1.11 cannot make two tasks share element buffers, and the Newton loop does not reallocate those buffers.
  The number of spawned assembly tasks is capped to the number of preallocated workspaces; if that invariant is ever violated, an error is thrown instead of reusing a workspace.
  `update_states!` uses `Threads.@threads` because each cell state is owned by exactly one loop iteration.
- **MPI.** Cells are partitioned by `mod(cellid-1, nranks) == rank` for linear, nonlinear, and stress-output work.
  After local assembly, the per-rank `K_tangent.nzval`, `K_linear.nzval`, residual vector, and stress cache are summed across ranks.
  `stiffness_matrix` and `compute_stresses` are therefore collective calls that every rank must reach, including on failure: a local failure on one rank is made collective before it propagates, so peer ranks throw instead of blocking in the reduction.
  All `mpi_allreduce!` calls are no-ops when `nranks == 1`.
  The current MPI path replicates the global sparse matrices, residual vector, and stress cache on every rank; MPI distributes the cell-level constitutive work but not the global linear algebra storage or the sparse linear solve.

Threading scales automatically with the Julia thread count.
MPI support is enabled automatically whenever `MPI.Init()` has been called in the user's script.

## Dimensionality wrappers

The generic `PlaneStrain` and `PlaneStress` wrappers embed the 2D deformation gradient into 3D, call the 3D constitutive update, and extract the in-plane parts.
Wrappable 3D materials expose `compute_PK1_3D(material, F, dt, state)` and `update_state_from_3D!(state, material, F, dt)`.
The second hook writes the current/trial state for the converged embedded 3D deformation; the final commit still happens through `update_states!`.
See the [Wrappers](wrappers.md) page for the exact embedding, static-condensation, and local-Newton math.
Some materials provide custom wrapper routes:
`Ogden` follows the same public wrapper API but uses an exact-tangent wrapper assembly, while `VEVP_Zhao2021_AT` supports `PlaneStrain` through an analytical tangent path but not `PlaneStress` (this would require derivation and implementation of another matching 2D tangent).

## Material constructor conventions

Every exported material constructor follows these conventions:

- **Material parameters** are passed as **positional arguments**.
- **Behavior options** – integration modes, time-unit scalings, tangent types – are passed as **keyword arguments** with sensible defaults.
  Changing a keyword argument changes *how* the model evaluates its response, not *what* material it represents.

```julia
# VEPD_Detrez2010: material parameters positional, integration modes are kwargs
mat1 = VEPD_Detrez2010(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ;
                 plastic_update=:end_step, maxwell_update=:closed_form_cv)

# VEVP_Zhao2021_AD / VEVP_Zhao2021_AT: material parameters positional, dt_scale is a kwarg
mat2 = VEVP_Zhao2021_AD(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV;
                 dt_scale=1.0)

# MooneyRivlin: material parameters positional, tangent evaluation type is a kwarg
mat3 = MooneyRivlin(C10, C01, κ; tangent=:AD)

# PlaneStress wrapper: mat1 is the base model, local-Newton controls are kwargs
mat4 = PlaneStress(mat1; tol=1e-10, maxiter=20)
```

Material model authors are encouraged to follow these conventions as outlined in the [Developer guide](developer_guide.md).

## The `dt` keyword

The public assembly functions (`stiffness_matrix`, `try_stiffness_matrix`, `compute_forces`) and the postprocessing function (`compute_stresses`) all accept `dt` as a keyword argument. However, only the assembly functions are allowed to advance material history.

- **Rate-independent** materials ignore `dt` (see the [Stable models](models/index.md#Stable-models) table).
- **Viscoelastic / viscoplastic** materials use `dt` to evolve internal variables (viscous stretches, plastic flow, cumulative strain, etc.).
  Passing `dt=0.0`, however, does not lead to obtaining a rate-independent model behavior; it only freezes the (viscous) state evolution.

`stiffness_matrix` and `compute_forces` pass `dt` into the element update and write trial state.
`compute_stresses` is a stress-output pass: for the bundled stateful materials it evaluates stresses using the state currently held in the assembler (whether that is an uncommitted trial state during debugging, or the committed state during standard postprocessing) without taking another constitutive time step.
Typical postprocessing calls `compute_stresses(assembler, u)` and uses the default `dt = 0.0`.
Postprocessing accepts `dt` so that custom rate-dependent materials have access to the time increment if their stress formula requires it, but their stress routines must still remain non-evolving and not commit state.

!!! info "Time units (`dt_scale` kwarg)"
    Some models take a `dt_scale` keyword that multiplies `dt` before the constitutive update runs.
    Set it to a value that matches the time unit used during parameter calibration (`1.0` if `dt` already uses that unit, `1e-9` when the driver passes nanoseconds but the calibration used seconds, etc.):

    ```julia
    mat = VEVP_Zhao2021_AD(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV;
                     dt_scale = 1e-9)
    ```

    Check the individual material model pages to see if this keyword is allowed, e.g., the [Zhao model page](@ref "VEVP Zhao 2021").

## Adaptive time stepping and recoverable local failures

Some nonlinear local updates can fail before the global Newton loop recovers.
Examples include the plastic return in `VEPD_Detrez2010`, the Perzyna solve in `VEVP_MOAMMM`, and the local plane stress condensation in `PlaneStress`.
These exceptions are thrown as subtypes of [`LocalAssemblyFailure`](@ref).
Calling [`try_stiffness_matrix`](@ref) instead of [`stiffness_matrix`](@ref) turns those failures into a result object that an outer time step controller can inspect (see [Adaptive time stepping](tutorials/adaptive_time_stepping.md)):

```julia
result = try_stiffness_matrix(assembler, u_trial; dt=dt_step)
if !result.converged
    # Reject this load/time step and retry with a smaller dt_step.
    @info "Reducing time step after local assembly failure" error=result.error
end
```

`try_stiffness_matrix` catches only recoverable `LocalAssemblyFailure` exceptions.
Other errors are still rethrown to expose bugs.
In multithreaded assembly, `try_stiffness_matrix` can safely recover from thread errors, provided they are caused exclusively by `LocalAssemblyFailure` exceptions.
On failure, the assembler's matrix and residual buffers hold partial results, but they are re-zeroed by the next assembly call and need no manual reset.
Inspect `result.error` to see what went wrong, and call [`revert_states!`](@ref) before retrying the step to roll back the trial material history written by the failed attempt.
In MPI runs, failure states are synchronized across all ranks to prevent deadlocks.
If one rank fails, every rank receives `converged = false`; consequently, ranks without the original local exception also show a [`RemoteAssemblyFailure`](@ref) in `result.error`.

A typical adaptive skeleton looks as follows:

```julia
t = 0.0
u = zeros(ndofs(dh))
controller = TimeStepController(; dt_min, dt_max, max_rejections=20)
dt = 0.25

while t < t_end
    accepted = false
    u_start = copy(u)
    t_start = t

    while !accepted
        t_trial = min(t_start + dt, t_end)
        dt_step = t_trial - t_start
        update!(ch, t_trial)
        u .= u_start
        apply!(u, ch)
        # Construct specific time-dependent external load variables for t_trial here.

        newton_converged = false
        step_rejected = false
        for _ in 1:max_newton
            result = try_stiffness_matrix(assembler, u; dt=dt_step)
            if !result.converged
                revert_states!(assembler)
                u .= u_start
                dt = reject_step!(controller, dt_step)
                step_rejected = true
                break
            end

            K, r = result.K, result.r
            apply_zero!(K, r, ch)
            if norm(r) < newton_tol
                newton_converged = true
                break
            end
            u .-= K \ r
        end

        if newton_converged
            update_states!(assembler)
            t = t_trial
            accepted = true
            dt = accept_step!(controller, dt_step)
        elseif !step_rejected
            revert_states!(assembler)
            u .= u_start
            dt = reject_step!(controller, dt_step)
        end
    end
end
```

Call [`update_states!`](@ref) only after an accepted, converged outer step.
If an assembly attempt or Newton step is rejected, restore the quadrature states with `revert_states!(assembler)` before retrying from the last accepted displacement and time.
A failed local update may already have written trial state before throwing; rollback is therefore mandatory even though no global step was accepted.
Keep `t_start` and `u_start` fixed while retrying a rejected step; only `dt`, `t_trial`, boundary conditions, and time-dependent loads should change.
Pass `dt_step`, the actually attempted increment after final-time clipping, to `accept_step!` and `reject_step!`.
`TimeStepController` supplies the shrink/grow policy, optional `dt_min`/`dt_max` bounds, and a maximum consecutive rejection count.
Use `dt_min` together with `max_rejections` when a permanently failing step must not retry forever; `dt_min = NaN` disables only the lower-bound guard.
When the controller cannot shrink further, `reject_step!` throws [`TimeStepControllerExhausted`](@ref).
In MPI, every rank must participate in the same sequence of `try_stiffness_matrix` calls and reject the step collectively when `converged == false`.

## What the package does *not* do on its own

- It does **not** set up `DofHandler` or `ConstraintHandler` – that is the user's job using Ferrite.jl syntax.
- It does **not** run a Newton loop – `stiffness_matrix` or `try_stiffness_matrix` takes one `(u, dt)` and returns one `(K, r)`, i.e. the tangent and the internal force; the user tests the convergence.
- It does **not** integrate in time – the user owns the time loop, the time step size, and any accepted/rejected-step policy.
- It does **not** apply Dirichlet boundary conditions – the user calls Ferrite's `apply!(u, ch)` and `apply_zero!(K, r, ch)` around the solve.
- It does **not** form the residual – the user subtracts the external force from the internal one: `residual .= f_int .- f_ext`.

For complete loops, see the fixed-step Newton solve in [2D plate with a hole](tutorials/plate_with_hole.md) and the adaptive time stepping driver in [Adaptive time stepping](tutorials/adaptive_time_stepping.md).
For assembling `f_ext`, see [External loads](loads.md).