# FerriteSolidMechanics.jl

A solid mechanics assembly framework and material model library for Ferrite.jl.

FerriteSolidMechanics.jl provides finite element assembly infrastructure for solid mechanics in Ferrite.jl alongside a library of ready-to-use constitutive models and a quadrature point interface for implementing new ones.
The package handles element-level assembly, external load integration, state bookkeeping, thread and MPI parallelism, and provides MPI-distributed solving through MUMPS.
Dimensionality wrappers adapt 3D models to 2D plane strain and plane stress analyses; an adaptive time step controller allows step size adjustment after failed material updates.

!!! note
    Contributions are welcome! If you implement a new material model using FerriteSolidMechanics.jl, consider opening a [pull request](https://github.com/LukasLaubert/FerriteSolidMechanics.jl) to add it to the package.

In a nutshell, FerriteSolidMechanics.jl works as follows:

```julia
using Ferrite
using FerriteSolidMechanics

# After setting up a grid and initializing Ferrite's DofHandler `dh`,
# ConstraintHandler `ch`, LoadHandler `lh`, and displacement vector `u`:
material = J2Plasticity(100.0, 0.3, 1.0, 10.0)  # assign material model
assembler = create_assembler(material, dh, ch)  # create material assembler

K, r = stiffness_matrix(assembler, u; dt=0.5)   # get stiffness tangent and internal residual
# ... solve the global Newton step and update u ...
update_states!(assembler)                       # commit material state
stresses = compute_stresses(assembler, u)       # postprocessing
```

## Why this package

Running solid mechanics simulations in Ferrite.jl requires cell iteration, per-sub-dof-handler cell values and quadrature rules, and – depending on the problem – state management, external load integration, and thread/MPI parallelism. FerriteSolidMechanics.jl provides this infrastructure generically for small- and finite-strain models, alongside an extendable catalog of models built on it. The package includes:

- a **quadrature point constitutive interface** (`material_response`, `kinematics`, plus the state lifecycle `create_state`, `update_state!`, `revert_state!`): a material author writes only the constitutive update – parameters, state, stress, and tangent – comparable to an Abaqus `UMAT`. A generic element routine owns the quadrature loop, integration weights, weak form, and trial-state bookkeeping; stateless hyperelastic models can even be defined by a single strain energy function via `AbstractHyperelastic`. For formulations whose element structure is material-specific, the element-level hooks (`_assemble_element!`, `_compute_stress_qp`) remain stable extension points,
- **dimensionality wrappers** ([`PlaneStrain`](@ref), [`PlaneStress`](@ref)) that use compatible 3D materials in 2D analyses: `PlaneStrain` uses 2D-to-3D embedding, while `PlaneStress` solves the out-of-plane stretch with a local Newton iteration and condenses the tangent,
- a **generic material assembler** that handles 2D / 3D, mixed-element grids, sub-dof-handlers, linear preassembly, thread parallelism (`OhMyThreads.@tasks`), and MPI-parallel constitutive evaluation and element assembly,
- **external load assembly** ([`LoadHandler`](@ref)) for Neumann boundary conditions: [`BodyForce`](@ref), [`Traction`](@ref), [`Pressure`](@ref) and [`NodalForce`](@ref) are collected with `add!` and integrated into the external force vector by [`external_forces!`](@ref), using the assembler's quadrature order and thickness so that external and internal forces match,
- **adaptive time stepping** ([`TimeStepController`](@ref)) that retries a failed material update with a smaller step size, using the recoverable failures reported by [`try_stiffness_matrix`](@ref),
- a **distributed linear solve** ([`distributed_solve`](@ref)) that splits one MUMPS factorization across MPI ranks, instead of every rank factorizing the whole system,
- **bundled material models** ranging from linear elasticity to finite-strain hyperelastic and viscoelastic-viscoplastic laws (including `MooneyRivlin`, `Ogden`, `ArrudaBoyce`, `VEPD_Detrez2010`, `VEVP_Zhao2021_AD`, `VEVP_MOAMMM`),
- a **MaterialModelsBase.jl bridge** ([`FromMaterialModelsBase`](@ref)) that runs a material written for [MaterialModelsBase.jl](https://github.com/KnutAM/MaterialModelsBase.jl) or [MechanicalMaterialModels.jl](https://github.com/KnutAM/MechanicalMaterialModels.jl) through the same assembly path as a bundled model.

To add a new material, implement `material_response` and `kinematics` for it.
The [Developer guide](developer_guide.md) walks through the full interface using the bundled `J2Plasticity` model, and through the one-function hyperelastic path that needs only a strain energy density `Ψ(C)`.

## Where to start

| If you want to… | Read |
| --- | --- |
| Get started by running a simple 2D problem | [2D plate with a hole](tutorials/plate_with_hole.md) |
| Run a 3D viscoelastic problem and plot the results | [DMA cantilever beam](tutorials/cantilever_dma.md) |
| Retry failed material updates with smaller steps | [Adaptive time stepping](tutorials/adaptive_time_stepping.md) |
| Run large-scale problems on clusters using MPI | [MPI four-point bending](tutorials/mpi_four_point_bending.md) |
| Pick a constitutive model for your problem | [Material models](models/index.md) |
| Reduce a 3D model to plane stress / plane strain | [Wrappers](wrappers.md) |
| Apply tractions, pressures, body forces, or nodal forces | [External loads](loads.md) |
| Understand the assembler, time stepping, and material states | [Concepts](concepts.md) |
| Choose a linear solver, rank and thread layout, and BLAS threads | [Performance and parallel execution](performance.md) |
| Add your own material model | [Developer guide](developer_guide.md) |
| Look up the signature of a function | [General API](@ref) |
| Diagnose a behaviour you don't understand | [FAQ](faq.md) |
| See relations to further Ferrite-based packages | [Related packages and acknowledgements](@ref) |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/LukasLaubert/FerriteSolidMechanics.jl")
```

For local development:

```julia
using Pkg
Pkg.develop(path=".")
Pkg.test("FerriteSolidMechanics")
```

## Status

FerriteSolidMechanics.jl requires Julia `1.10` or newer; CI tests Julia `1.10` and the latest stable Julia `1.x`.
The implementation is tested with Julia `1.11.6` and Ferrite `1.4.1`.
It is licensed under the MIT license (see `LICENSE` in the repository root); material model provenance and acknowledgements are documented on the material pages.

## Related packages and acknowledgements

FerriteSolidMechanics.jl sits alongside several Julia packages for constitutive modelling and finite element assembly, and follows some conventions established by them.

- [MaterialModels.jl](https://github.com/kimauth/MaterialModels.jl) (K. Auth and contributors) is a material model library without an assembly layer. Its `material_response` convention returning `(stress, tangent, new_state)`, the `AbstractMaterial`/`AbstractMaterialState` pair, and `PlaneStrain`/`PlaneStress` as dimensional wrappers are used here under the same names.
- [MaterialModelsBase.jl](https://github.com/KnutAM/MaterialModelsBase.jl) (K. A. Meyer) is an implementation-independent interface package developed from MaterialModels.jl, with [MechanicalMaterialModels.jl](https://github.com/KnutAM/MechanicalMaterialModels.jl) as the model library built on it. The constitutive interface in FerriteSolidMechanics.jl follows the MaterialModelsBase conventions, and models written for MaterialModelsBase can be used through the [`FromMaterialModelsBase`](@ref) wrapper.
- [FerriteAssembly.jl](https://github.com/KnutAM/FerriteAssembly.jl) (K. A. Meyer) covers similar ground to the assembler here: a generic Ferrite assembly path over pluggable materials, with an element-level hook, `update_states!` for committing converged state, and per-task scratch buffers. Its `LoadHandler`, a container populated with `add!` and applied at a given time, is followed here under the same name. FerriteAssembly also assembles general facet contributions to the tangent and threads load assembly, neither of which FerriteSolidMechanics.jl currently does; assembly of the tangent and internal force in FerriteSolidMechanics.jl is instead MPI-parallel, with the external force vector replicated on every rank.
- [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)'s hyperelasticity and von Mises plasticity tutorials are the source of the `Ψ(C)`/`constitutive_driver` convention and of the element integration loops.
- [FerriteDistributed.jl](https://github.com/Ferrite-FEM/FerriteDistributed.jl) runs MPI by partitioning the grid and the DofHandler. FerriteSolidMechanics.jl replicates the global system on every rank and reduces it instead, which leaves the Ferrite `DofHandler` unchanged at the cost of per-rank memory; see [Performance and parallel execution](performance.md).
- [CAPRICCIO - Tool to run concurrent Finite Element-Molecular Dynamics Simulations (Version 3.0.0)](https://doi.org/10.5281/zenodo.18326736) (J. Roksvaag, Apache-2.0) couples finite elements to molecular dynamics and is the origin of the assembler interface used here. The entry points `create_assembler`, `stiffness_matrix`, `compute_forces` and `compute_stresses`, the `_assemble_element!` element hook, the `alpha_value` quadrature point scaling hook with its `AlphaValues` companion, and the replicated MPI assembly keep their names and design from it.

Scientific references and provenance for the bundled constitutive models are on the individual [material pages](models/index.md).