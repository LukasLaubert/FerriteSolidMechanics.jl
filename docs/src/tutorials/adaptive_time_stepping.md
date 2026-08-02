# Adaptive time stepping

A plane strain quarter plate with a hole, solved with an adaptive outer loop that rejects and retries load steps when a local material update fails.
The runnable script is `examples/adaptive_time_stepping.jl` and can be executed from the repository root with:

```sh
julia --project=. examples/adaptive_time_stepping.jl
```

## Problem setup

The geometry and boundary conditions follow the [2D plate with a hole](plate_with_hole.md) tutorial.
The right edge is displaced outward with the load factor `t`, while the bottom and left edges enforce symmetry.

This tutorial focuses on time step-controlling:
Instead of prescribing a fixed step size for the whole run, it attempts a step, checks whether both the local material updates and the global Newton loop converges, and, if not succeeding, retries with a smaller `dt`.

## Grid, interpolation, constraints

The mesh, interpolation, displacement field, and boundary conditions are the same as in the [2D plate with a hole](plate_with_hole.md) tutorial.
This tutorial reuses that setup and changes only the material model and the outer load step loop.
For the full source, see [Plain program](#Plain-program).

## Material

We apply `VEPD_Detrez2010` through a plane strain wrapper:

```julia
detrez_parameters = (
    167.0, 0.27, 3.24, 28.6, 20.8,
    0.59, 41.7, 4.5, 3.58,
    [101.0, 21.0], [10.0, 100.0],
)
material = PlaneStrain(VEPD_Detrez2010(detrez_parameters...))
```

`VEPD_Detrez2010`'s local plastic return can raise `VEPD_Detrez2010ConvergenceError`, which subtypes [`LocalAssemblyFailure`](@ref).
The adaptive time stepping strategy uses this to recover from local failures.

## Assembler

The assembler is created in the same way as in the fixed-step tutorials:

```julia
assembler = create_assembler(material, dh, ch; quadrature_order=2)
```

## Rolling back state

Assembly is stateful: While `stiffness_matrix` and `try_stiffness_matrix` evaluate the constitutive response at quadrature points, they store the resulting trial material states in the assembler (at the new `dt` provided).
If a trial step is rejected, those tentative states must be discarded before the same step is retried.
To this end, call FerriteSolidMechanics' [`revert_states!`](@ref) function to restore the assembler's material states to their last committed values.
This call works with both serial and MPI assemblers by default:

```julia
revert_states!(assembler)
```

## Adaptive time stepping and Newton loop

The outer loop advances from `t = 0` to `t_end`.
Each trial step starts from the last accepted displacement `u_start` and time `t_start`:

```julia
t = 0.0                                           # last accepted time
dt = 0.25                                         # trial step size
dt_min = 1e-4                                     # smallest allowed trial step
dt_max = 0.25                                     # largest step after accepted steps
dt_shrink = 0.5                                   # factor after rejected steps
dt_grow = 2.0                                     # factor after accepted steps
controller = TimeStepController(; dt_min, dt_max, shrink=dt_shrink, grow=dt_grow, max_rejections=20) # dt shrink/grow policy after accepted/rejected steps

while t < t_end - 1e-12                           # continue until the final time is accepted
    accepted = false
    u_start = copy(u)                             # displacement to restore on retry
    t_start = t                                   # time to restore on retry

    while !accepted                               # retry until accepted
        t_trial = min(t_start + dt, t_end)        # trial end time, clipped at t_end
        dt_step = t_trial - t_start               # material time increment
        update!(ch, t_trial)                      # Dirichlet data at t_trial
        u .= u_start                              # restore displacement at t_start
        apply!(u, ch)                             # impose Dirichlet values at t_trial

        newton_converged = false
        step_rejected = false

        for iter in 1:max_newton                  # Newton iterations for this trial step
            result = try_stiffness_matrix(assembler, u; dt=dt_step) # returns converged=false on failure
            if !result.converged                  # recover local material if failed
                revert_states!(assembler)         # discard trial material updates
                u .= u_start                      # restore displacement at t_start
                dt = reject_step!(controller, dt_step) # shrink the attempted step for the retry
                step_rejected = true
                break
            end

            K, r = result.K, result.r             # convergence successful: get tangent/residual
            apply_zero!(K, r, ch)                 # apply zero-displacement constraints
            if norm(r) <= newton_tol              # converged global Newton solve
                newton_converged = true
                break
            end
            u .-= K \ r                           # Newton correction
        end

        if newton_converged
            update_states!(assembler)             # commit trial material states
            t = t_trial                           # accept the trial end time
            accepted = true
            dt = accept_step!(controller, dt_step) # next trial step size
        elseif !step_rejected
            revert_states!(assembler)             # discard failed Newton trial states
            u .= u_start                          # restore displacement at t_start
            dt = reject_step!(controller, dt_step) # shrink the attempted step for the retry
        end
    end
end
```

The user-owned outer loop decides whether a step is accepted, retried, or rejected.
[`TimeStepController`](@ref) supplies a basic scalar grow/shrink policy and rejection budget using [`accept_step!`](@ref) and [`reject_step!`](@ref).

Two failure modes lead to the same retry path:
- A local material failure makes [`try_stiffness_matrix`](@ref) return `converged = false`.
- A global Newton failure occurs when the Newton loop reaches `max_newton` without satisfying `newton_tol`.

In both cases, the loop restores the material state, restores `u`, reduces the attempted `dt_step` through [`reject_step!`](@ref), and retries from the same `t_start`.
Only an accepted step calls [`update_states!(assembler)`](@ref update_states!).
The [`TimeStepController`](@ref) uses both the lower bound `dt_min` and `max_rejections` as exhaustion guards.
Setting the default value of `dt_min = NaN` or `dt_max = NaN` disables the lower- or upper-bound guard, respectively.
The `max_rejections` default is unbounded as well.

For why this loop calls [`try_stiffness_matrix`](@ref) instead of [`stiffness_matrix`](@ref), see [Adaptive time stepping and recoverable local failures](@ref).

## Postprocessing

After the final accepted step, the example evaluates the stresses:

```julia
stresses = compute_stresses(assembler, u)
```

The named tuple returned by `run_adaptive_time_stepping` also stores the accepted times, accepted step sizes, final residuals, iteration counts, rejected-step count, and external counters for how often `dt` actually decreased or increased.
The script further prints a short summary:

```text
Solved adaptive time stepping quarter plate
  cells: 48
  dofs: 130
  accepted steps: 8
  rejected/retried steps: 7
  dt decreases: 7
  dt increases: 5
  peak displacement: 0.12
  dt range over accepted steps: [0.015625, 0.25]
```

The accepted `dt` values show which step sizes were eventually accepted.
The `dt_decreases` and `dt_increases` counters are tracked by the tutorial's code, not by [`TimeStepController`](@ref).
`dt_increases` counts only accepted steps where [`accept_step!`](@ref) returns a larger `dt`; a step already capped at `dt_max` is not counted as an increase.
Similarly, `dt_decreases` counts only successful calls to [`reject_step!`](@ref); if the next step would fall below `dt_min`, `reject_step!` throws before returning.

## Swapping the material model

The adaptive loop can drive any material model that implements the `AbstractMaterial` interface.
To swap the material, change only the material constructor; the assembler, rollback, and adaptive loop stay the same:

```julia
material = PlaneStrain(VEPD_Detrez2010(detrez_parameters...))
material = PlaneStrain(J2Plasticity(70e3, 0.3, 250.0, 1e3))
material = PlaneStress(VEVP_Zhao2021_AD(5.0, 20.0, 6, 0.0, 0.0, 283.0, 18.51, 0.1421, 0.6152, 3.0, 3.0, 0.0, 0.0, 283.0))
```

Materials with robust local updates may never reject a step.
The same driver still works; upon reaching `dt_max`, it simply behaves like a fixed-step loop unless a local update or global Newton iteration fails.

## Plain program

The full runnable source is rendered from `examples/adaptive_time_stepping.jl`:

```@eval
using Markdown
import FerriteSolidMechanics
path = joinpath(pkgdir(FerriteSolidMechanics), "examples", "adaptive_time_stepping.jl")
Markdown.parse("```julia\n" * read(path, String) * "\n```")
```
