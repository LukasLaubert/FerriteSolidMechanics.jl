# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Unified the material model docstrings: every model documents its `Parameters` and `Keyword arguments` before its `References`. The `compute_PK1_3D` and `update_state_from_3D!` extension points moved from the material model API page to the general API page, next to the other material interface hooks.

### Fixed

- Documentation: the wrapper extension points are required of finite-strain materials only, not of every wrapped material; the `AbstractHyperelastic` wrapper fallback derives the stress from the strain energy `Ψ` rather than from `material_response`; the `Ogden` page no longer documents a `tangent` keyword that the constructor does not accept.

## [0.1.0] - 2026-08-09

### Added

- Material catalogue: `Hooke` and `Hooke2D` (linear elastic), `NeoHooke`, `ArrudaBoyce`, `MooneyRivlin` and `Ogden` (hyperelastic), `J2Plasticity`, `VEPD_Detrez2010` (viscoelastic-plastic with damage), `VEVP_Zhao2021_AD` and `VEVP_Zhao2021_AT` (automatic differentiation and analytic tangent), and `VEVP_MOAMMM` (viscoelastic-viscoplastic with tension-compression asymmetric isotropic and kinematic hardening).
- `GenericMaterialAssembler`, built by `create_assembler` from a single material model or a dict mapping cellset names to material models, for 2D and 3D grids with mixed element types. `stiffness_matrix` evaluates elements across Julia threads and reduces the global tangent and internal force across the ranks of `MPI.COMM_WORLD`. Cells whose material reports `is_linear` are assembled once and reused. Further entry points are `compute_forces`, `compute_stresses`, `update_states!` and `revert_states!`, which separate trial writes during assembly from the commit after a converged Newton step.
- Quadrature point constitutive interface: `material_response` returns stress, tangent and new state for a strain measure selected by `kinematics` (`SmallStrain` or `FiniteStrain`). History is managed by `create_state`, `update_state!`, `revert_state!` and `copy_state!`, with `NoState` for stateless models. A subtype of `AbstractHyperelastic` needs only the strain energy density `Ψ(C, material)`. `_assemble_element!` overrides the interface at element level, and `_compute_stress_qp` at quadrature point level for postprocessing.
- `PlaneStrain` and `PlaneStress` wrappers, which run a 3D material in a 2D analysis. `PlaneStress` solves for the out-of-plane stretch at every quadrature point.
- `TimeStepController` with `accept_step!` and `reject_step!` for adaptive step sizes. `try_stiffness_matrix` reports a `LocalAssemblyFailure` raised inside a local material update as a failed step instead of throwing, and synchronizes that failure across MPI ranks.
- `distributed_solve`, which splits one factorization of the global system across MPI ranks. Provided by the MUMPS package extension and available on Unix.
- `LoadHandler` with the load types `BodyForce`, `Traction`, `Pressure` and `NodalForce`. `external_forces!` assembles the external force vector for a given load factor or time.
- `FromMaterialModelsBase`, which runs a material written for MaterialModelsBase.jl or MechanicalMaterialModels.jl through the same assembly path as a bundled model. Provided by the MaterialModelsBase package extension.

[Unreleased]: https://github.com/LukasLaubert/FerriteSolidMechanics.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LukasLaubert/FerriteSolidMechanics.jl/releases/tag/v0.1.0