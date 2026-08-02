# Material models

This page helps you choose a material model based on your requirements.
Each linked model page (see [Stable models](#Stable-models)) provides the equations, constructor, state behavior, implementation notes, provenance, acknowledgements, and references.

## Which model should I use?

The table below is a non-exhaustive overview of frequent problem classes and suitable material formulations; applicability should still be checked for each case.

| Problem class | Reach for |
| --- | --- |
| Linear elasticity | [`Hooke`](@ref)/[`Hooke2D`](@ref) |
| Small-strain plasticity | [`J2Plasticity`](@ref) |
| Finite-strain rubber / elastomer | [`NeoHooke`](@ref), [`MooneyRivlin`](@ref), [`Ogden`](@ref), [`ArrudaBoyce`](@ref) |
| Semi-crystalline / crosslinked polymer with damage | [`VEPD_Detrez2010`](@ref) |
| Thermoplastic / glassy polymer | [`VEVP_MOAMMM`](@ref), [`VEVP_Zhao2021_AD`](@ref) |
| 2D plane strain/stress for any model | [`PlaneStrain`](@ref), [`PlaneStress`](@ref) wrappers |

See the [Stable models](#Stable-models) table below for the full list and related variants.

## Stable models

The `dt` column indicates whether the material is rate-dependent and actively uses the time step size `dt` for material state updates.

| Constructor | Page | Use when you need | `dt` | 2D support |
| --- | --- | --- | --- | --- |
| `Hooke` | [Linear Elasticity](@ref "Linear Elasticity") | 3D isotropic linear elasticity | No | `PlaneStrain` / `PlaneStress`; prefer `Hooke2D` |
| `Hooke2D` | [Linear Elasticity](@ref "Linear Elasticity") | Direct 2D linear elasticity | No | Direct |
| `NeoHooke` | [Neo–Hookean](@ref "Neo–Hookean") | Compressible finite-strain hyperelasticity | No | `PlaneStrain` / `PlaneStress` |
| `ArrudaBoyce` | [Arruda–Boyce](@ref "Arruda–Boyce") | Limited-chain finite-strain hyperelasticity | No | `PlaneStrain` / `PlaneStress` |
| `MooneyRivlin` | [Mooney–Rivlin](@ref "Mooney–Rivlin") | Two-invariant finite-strain hyperelasticity | No | `PlaneStrain` / `PlaneStress` |
| `Ogden` | [Ogden](@ref "Ogden") | Principal stretch finite-strain hyperelasticity with arbitrary term count | No | Specialized `PlaneStrain` / `PlaneStress` |
| `J2Plasticity` | [J2 Plasticity](@ref "J2 Plasticity") | Small-strain von Mises plasticity | No | `PlaneStrain` / `PlaneStress` |
| `VEPD_Detrez2010` | [VEPD Detrez 2010](@ref "VEPD Detrez 2010") | Finite-strain viscoelastic-plastic-damage response | Yes | `PlaneStrain` / `PlaneStress` |
| `VEVP_Zhao2021_AD` | [VEVP Zhao 2021](@ref "VEVP Zhao 2021") | Finite-strain viscoelastic-viscoplastic glassy-polymer response | Yes | `PlaneStrain` / `PlaneStress` |
| `VEVP_Zhao2021_AT` | [VEVP Zhao 2021](@ref "VEVP Zhao 2021") | Related Zhao formulation with trial-state stress and analytical tangent | Yes | 3D and `PlaneStrain` only |
| `VEVP_MOAMMM` | [VEVP MOAMMM](@ref "VEVP MOAMMM") | Finite-strain viscoelastic-viscoplastic response with Drucker-Prager plasticity | Yes | `PlaneStrain` / `PlaneStress` |

## Using MaterialModelsBase.jl models

FerriteSolidMechanics.jl focuses on established material models provided as a whole, but also supports modularly composed materials from the following packages by Knut Andreas Meyer (which also offer extended tooling for them, such as parameter differentiation for calibration and stress-state-driven point simulations):
[MaterialModelsBase.jl](https://github.com/KnutAM/MaterialModelsBase.jl) specifies an interface for constitutive models, and [MechanicalMaterialModels.jl](https://github.com/KnutAM/MechanicalMaterialModels.jl) provides an existing library of material models (elasto-(visco)plasticity and viscoelasticity).
Any material implementing this interface can be used directly through the [`FromMaterialModelsBase`](@ref) wrapper.

The wrapper is a thin bridge: FerriteSolidMechanics' own constitutive interface deliberately follows similar conventions.
Trial/commit state management, threaded/MPI assembly, stress output, and `try_stiffness_matrix`-based adaptive stepping (local convergence failures surface as [`MaterialModelsBaseConvergenceError`](@ref)) therefore all work as for bundled material models.

The bridge targets MaterialModelsBase.jl **0.4**; a different version may require an update.
The extension activates when MaterialModelsBase is loaded:

```julia
using Pkg
Pkg.add(url="https://github.com/KnutAM/Newton.jl") # dependency of MechanicalMaterialModels
Pkg.add(url="https://github.com/KnutAM/MaterialModelsBase.jl")
Pkg.add(url="https://github.com/KnutAM/MechanicalMaterialModels.jl")
```

```julia
using FerriteSolidMechanics
import MaterialModelsBase   # activates the extension; prefer `import` over `using` (both packages export `material_response`)
using MechanicalMaterialModels: Plastic, LinearElastic, Voce, ArmstrongFrederick, NortonOverstress

mmmMaterial = Plastic(;
    elastic    = LinearElastic(E=210.0e3, ν=0.3),
    yield      = 100.0,
    isotropic  = Voce(Hiso=50.0e3, κ∞=100.0),
    kinematic  = ArmstrongFrederick(Hkin=200.0e3, β∞=200.0),
    overstress = NortonOverstress(tstar=1.0, nexp=2.0)
)

mat = FromMaterialModelsBase(mmmMaterial)  # kinematics=SmallStrain() is the default
fem = create_assembler(mat, dh, ch)        # then solve exactly as with bundled models
```

For finite-strain MaterialModelsBase materials pass `FromMaterialModelsBase(mmmMaterial; kinematics=FiniteStrain())`, which uses the deformation gradient `F` and expects the material to return `(P, ∂P∂F, new_state)` (first Piola–Kirchhoff stress, its tangent, and updated state).
Note that FerriteSolidMechanics' [`PlaneStrain`](@ref)/[`PlaneStress`](@ref) *material wrappers* are unrelated to MaterialModelsBase's stress-state types of the same names.
Whether a material model used through `FromMaterialModelsBase` can be combined with these 2D wrappers is currently untested.
[Hooking into the wrappers](../wrappers.md#Hooking-into-the-wrappers) states the requirements for a wrappable material model.

## Related pages

- [Material model API](../api_models.md) lists the constructor docstrings of all bundled models, the `PlaneStrain` / `PlaneStress` wrappers, and the wrapper extension points.
- [Wrappers](../wrappers.md) explains how 3D material models are embedded into 2D plane strain and plane stress analyses.
- [Experimental models](../experimental.md) lists comparison variants under `FerriteSolidMechanics.Experimental`.
- [Developer guide](../developer_guide.md) explains the material interface (`material_response` + `kinematics`, trial/commit state handling, and the optional element-level assembly hook) needed to support these features in custom materials.