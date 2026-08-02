# API reference

This page provides the public API for `FerriteSolidMechanics`, grouped by functionality.
The material model constructors and the dimensionality wrappers are covered on the [Material model API](api_models.md) page.
For examples and core concepts, see the [Tutorials](tutorials/plate_with_hole.md) and [Concepts](concepts.md) pages.

You likely only need four functions to run a basic simulation: `create_assembler`, `stiffness_matrix`, `update_states!`, and `compute_stresses` (or `try_stiffness_matrix` alongside `TimeStepController` for adaptive time stepping).

---

## Assembler

The `GenericMaterialAssembler` is the central object that connects all phases of the simulation.
You construct it using `create_assembler`, evaluate it during the solver loop via `stiffness_matrix` (or `try_stiffness_matrix` or `compute_forces`), and extract results for postprocessing through `compute_stresses`.
Once a step has converged, you must call `update_states!` to commit the material history (if it is history-dependent).
The `compute_stresses` function acts as a non-evolving postprocessing call for the bundled stateful materials.
In normal use, call `compute_stresses(assembler, u)` without a `dt` keyword, as this optional argument is only forwarded for custom stress-output methods and must never be used to advance or commit history.

```@docs
create_assembler
GenericMaterialAssembler
stiffness_matrix
try_stiffness_matrix
compute_forces
compute_stresses
update_states!
revert_states!
```

Because the direct sparse solve (`K \ r`) lives in the driver's Newton loop rather than in this package, you manage its performance yourself.
As general guidance, three helpers provide settings measured on the [NHR@FAU Fritz cluster](https://doc.nhr.fau.de/clusters/fritz/):

[`recommended_solve_settings`](@ref) returns solver, ranks, threads, BLAS threads and estimated memory for a given grid and node, and whether that memory fits.
[`recommended_blas_threads`](@ref) returns the BLAS thread count alone: `ncores` in 3D when the rank has the node to itself, and 1 in 2D or when ranks share a node. The 3D gain is mesh-gated, reaching 2.1× at 108k DOFs while a 15k-DOF grid solved faster on a single thread.
[`estimated_replicated_memory_gb`](@ref) predicts the per-rank memory peak of the replicated solve, where every rank holds its own copy of the LU factorization.

The [Performance and parallel execution](@ref) page provides the measurements behind them.

```@docs
recommended_solve_settings
recommended_blas_threads
estimated_replicated_memory_gb
```

For large 3D systems, where the replicated direct solve dominates time and its factors may not fit on a single node, [`distributed_solve`](@ref) is a drop-in replacement for `K \ r` that solves the system with the distributed MUMPS solver across MPI ranks.
It is provided by a package extension and is only available once you load MUMPS.jl (`import MUMPS`); see the [MPI four-point bending](@ref) tutorial for when it pays off and how to launch it.

A failed factorization or solve throws [`DistributedSolveError`](@ref).

```@docs
distributed_solve
DistributedSolveError
```

## External loads

`LoadHandler` assembles the external force vector.
It is built and closed like Ferrite's `ConstraintHandler`, using `add!` and `close!`, and evaluated once per load step with `external_forces!`.
Prescribed displacements are not part of it; use Ferrite's `Dirichlet` for those.

Each load type is named after the units of the value its function returns: [`Traction`](@ref) and [`Pressure`](@ref) are force per unit area in 3D and per unit length in 2D, [`NodalForce`](@ref) is a force, and [`BodyForce`](@ref) is force per unit volume in 3D and per unit area in 2D.
Load functions accept the call forms `f(x)`, `f(x, t)` and, on facet loads, `f(x, t, n)`, so spatial and temporal load profiles are written directly in the load function.

The [External loads](loads.md) page covers the units, the placement of `external_forces!` in the Newton loop, the MPI semantics, and how to change the loading during a simulation.

```@docs
LoadHandler
external_forces!
Traction
Pressure
NodalForce
BodyForce
AbstractLoad
```

## Adaptive time stepping

You can use `try_stiffness_matrix` to safely capture recoverable local material failures into a result object, which pairs well with the `TimeStepController` that provides a scalar step size adaptation policy.
However, because the package leaves the overall solver logic up to you, your custom Newton loop must still handle state rollback, boundary conditions, external loads, and convergence criteria.

```@docs
TimeStepController
TimeStepControllerExhausted
accept_step!
reject_step!
reset_controller!
```

## Material interface

A custom material typically implements the quadrature point-level constitutive interface by providing [`material_response`](@ref) and [`kinematics`](@ref), while history-dependent models must additionally supply a state type with the lifecycle hooks listed below.
Once these are defined, the generic element routine automatically handles the quadrature loop, integration weights, and trial-state bookkeeping.
For concrete implementation details, see the [Developer guide](developer_guide.md), or refer to the [Concepts](concepts.md) page for an explanation of trial and commit semantics.

Small-strain materials may additionally declare [`tangent_symmetry`](@ref).
The default [`Unsymmetric`](@ref) integrates every entry of the element stiffness; returning [`MajorSymmetric`](@ref) integrates only its lower triangle and mirrors the rest, which is roughly 1.4× faster but valid only when `D[i,j,k,l] == D[k,l,i,j]`.

```@docs
AbstractMaterial
AbstractMaterialState
LocalAssemblyFailure
NoState
material_response
material_stress
kinematics
AbstractKinematics
SmallStrain
FiniteStrain
tangent_symmetry
AbstractTangentSymmetry
MajorSymmetric
Unsymmetric
set_trial!
allocate_material_cache
is_linear
create_state
update_state!
revert_state!
copy_state!
```

### Hyperelastic quick path

If you are implementing a stateless hyperelastic model, you can subtype [`AbstractHyperelastic`](@ref) and implement only the strain energy density `Ψ(C, material)`.
Because the stress, tangent, element assembly, stress output, and 2D wrapper hooks are then derived automatically, you only need to override them with analytic expressions if you require better performance.

```@docs
AbstractHyperelastic
FerriteSolidMechanics.Ψ
```

### MaterialModelsBase.jl bridge

If you want to use material models implementing the `MaterialModelsBase.jl` interface, they can plug in through a wrapper as described in [Using MaterialModelsBase.jl models](models/index.md#Using-MaterialModelsBase.jl-models).
The necessary interface methods are provided by a package extension that is automatically activated when you call `import MaterialModelsBase`.

```@docs
FromMaterialModelsBase
MMBState
MaterialModelsBaseConvergenceError
```

### Element-level extension points

These underscore-prefixed methods are exported, stable extension points for materials with model-specific element structures (such as hand-assembled nodal tangent blocks, mixed formulations, or nonlocal coupling).
Because materials implementing `material_response` receive generic fallbacks for both functions, you normally do not need to define them.
History models with expensive tangents are an exception and should provide a cheap, tangent-free stress path for postprocessing: override the higher-level [`material_stress`](@ref) (which receives the strain measure directly), and reach for `_compute_stress_qp` only when you need control over the quadrature point extraction itself.

```@docs
_assemble_element!
_compute_stress_qp
```

## Kinematics

The `deformation_gradient` function is the main kinematics helper intended for direct use in element routines.
In contrast, wrapper hooks such as `compute_PK1_3D` and `update_state_from_3D!` are documented separately below.

```@docs
deformation_gradient
```

## Recoverable assembly failures

```@docs
RemoteAssemblyFailure
PlaneStressConvergenceError
VEPD_Detrez2010ConvergenceError
VEVP_MOAMMMConvergenceError
```

## Material models and dimensionality wrappers

The constructor docstrings of the eleven bundled material models, the `PlaneStrain` and `PlaneStress` wrappers, and the two wrapper extension points `compute_PK1_3D` and `update_state_from_3D!` are covered on the [Material model API](api_models.md) page.

## Alpha Scaling

These hooks define optional quadrature point scaling for stiffness, residual, and stress output.

```@docs
alpha_value
create_alpha_values
```