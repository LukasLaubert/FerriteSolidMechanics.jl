# FerriteSolidMechanics.jl

[![CI](https://github.com/LukasLaubert/FerriteSolidMechanics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/LukasLaubert/FerriteSolidMechanics.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/LukasLaubert/FerriteSolidMechanics.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/LukasLaubert/FerriteSolidMechanics.jl/actions/workflows/docs.yml)

*A solid mechanics assembly framework and material model library for [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl).*

FerriteSolidMechanics.jl provides finite element assembly infrastructure for solid mechanics in Ferrite.jl alongside a library of ready-to-use constitutive models.
One generic material model assembler takes care of (sub-)DOF handling, sparse assembly, threading, MPI reduction, and trial/commit state bookkeeping; a separate `LoadHandler` integrates the external loads.
Use a bundled model or implement your own constitutive law and reuse the same assembly path.

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/LukasLaubert/FerriteSolidMechanics.jl")
```

## Minimal example

```julia
using Ferrite
using FerriteSolidMechanics

grid = generate_grid(Quadrilateral, (2, 2))     # Ferrite grid
dh = DofHandler(grid)                           # Ferrite DofHandler
add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2) # 2D displacement field, linear interpolation
close!(dh)

ch = ConstraintHandler(dh)                      # Ferrite ConstraintHandler
# ... add boundary conditions here ...
close!(ch)

material = PlaneStress(ArrudaBoyce(100.0, 1000.0, 7.0)) # assign a material model
assembler = create_assembler(material, dh, ch)          # build the assembler

u = zeros(ndofs(dh))                                    # initial displacement

K, r = stiffness_matrix(assembler, u; dt=1.0) # assemble tangent and residual (trial state)
# ... solve the global Newton step, update u, and repeat until converged ...

update_states!(assembler)                     # commit the converged material state
stresses = compute_stresses(assembler, u)     # compute stresses (postprocessing)
```

[`examples/plate_with_hole_planestress.jl`](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/tutorials/plate_with_hole/) provides a complete Newton solve on a 2D plate with a hole.

## Features

- **[Material model catalogue](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/models/)**: linear elastic, hyperelastic, plastic, and viscoelastic-viscoplastic models. Among those are `Ogden`, `ArrudaBoyce`, `J2Plasticity`, and complex literature-specific polymer models such as `VEPD_Detrez2010`, which adds damage.
- **Material model interface**: implement the quadrature point hook `material_response` (UMAT-like), or, for a hyperelastic model, only the strain energy density `Ψ(C)`.
- **Element-level override**: optionally implement `_assemble_element!` (UEL-like), e.g., when the element structure itself is material-specific (hand-assembled tangent blocks, mixed or nonlocal formulations).
- **Generic material assembler**: 2D and 3D, mixed-element grids, linear preassembly, thread parallelism, and MPI-parallel assembly.
- **External loads**: `LoadHandler` collects `BodyForce`, `Traction`, `Pressure`, and `NodalForce` entries and assembles the external force vector.
- **2D wrappers**: `PlaneStrain` and `PlaneStress` embed a 3D material model into a 2D analysis.
- **Distributed linear solve**: `distributed_solve` splits one factorization across MPI ranks using MUMPS, instead of every rank factorizing the whole system.
- **Adaptive time stepping**: `TimeStepController` retries a failed material update with a smaller step size, using the recoverable failures reported by `try_stiffness_matrix`.
- **MaterialModelsBase.jl bridge**: wrap any [MaterialModelsBase.jl](https://github.com/KnutAM/MaterialModelsBase.jl) / [MechanicalMaterialModels.jl](https://github.com/KnutAM/MechanicalMaterialModels.jl) model with `FromMaterialModelsBase` and use it like a bundled model.

## Documentation

- [Material models](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/models/): catalogue of bundled models.
- [Tutorials](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/tutorials/plate_with_hole/): 2D plate with a hole, DMA cantilever beam, adaptive time stepping, and MPI four-point bending.
- [Concepts](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/concepts/): the assembler, the element loop, and trial/commit state management.
- [External loads](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/loads/): `LoadHandler`, the four load types, and their placement in the Newton loop.
- [Performance and parallel execution](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/performance/): choosing the linear solver, the rank and thread layout, BLAS threads, and memory per rank.
- [In-plane wrappers](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/wrappers/): `PlaneStrain` and `PlaneStress` usage.
- [Developer guide](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/developer_guide/): how to implement your own material model.
- [FAQ](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/faq/): possible uncertainties and troubleshooting.
- [API reference](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/api/): all exported functions and types.

## Related packages and acknowledgements

FerriteSolidMechanics.jl follows conventions established by earlier Julia packages.
The `material_response` interface, the `AbstractMaterial` / `AbstractMaterialState` pair, and the `PlaneStrain` / `PlaneStress` wrapper names originate from [MaterialModels.jl](https://github.com/kimauth/MaterialModels.jl) (K. Auth and contributors), from which [MaterialModelsBase.jl](https://github.com/KnutAM/MaterialModelsBase.jl) (K. A. Meyer) was developed as an implementation-independent interface package; materials written for MaterialModelsBase are used here through the `FromMaterialModelsBase` wrapper.
[FerriteAssembly.jl](https://github.com/KnutAM/FerriteAssembly.jl) (K. A. Meyer) covers similar ground to the assembler and the `LoadHandler` in this package, and [FerriteDistributed.jl](https://github.com/Ferrite-FEM/FerriteDistributed.jl) is a partitioned alternative to the replicated MPI assembly used here.
The assembler entry points, the `_assemble_element!` and `alpha_value` hooks, and that replicated MPI assembly originate from [CAPRICCIO](https://doi.org/10.5281/zenodo.18326736) (J. Roksvaag), a concurrent FE–MD coupling tool.
The `Ψ(C)` / `constitutive_driver` convention and the element integration loops originate from [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)'s tutorials; per-model provenance is on the material documentation pages.

The [documentation](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/#Related-packages-and-acknowledgements) compares these packages in more detail.

## License

FerriteSolidMechanics.jl is distributed under the MIT license; see `LICENSE`.
See the individual material documentation pages for scientific references, implementation provenance, and acknowledgements.