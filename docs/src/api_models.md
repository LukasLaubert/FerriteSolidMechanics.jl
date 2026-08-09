# Material model API

This page documents the constructors of the bundled material models together with the `PlaneStrain` and `PlaneStress` wrappers that embed a 3D model into a 2D analysis.
The assembler, the load handler, the material interface, and the remaining public functions are on the [General API](api.md) page.

---

## Material models

Each material model is a concrete `AbstractMaterial` subtype whose constructor signature is documented on its dedicated page within the [Material models](models/index.md) section.
The grouped autodocs below provide the associated parameter docstrings.

```@docs
Hooke
Hooke2D
NeoHooke
ArrudaBoyce
MooneyRivlin
Ogden
J2Plasticity
VEPD_Detrez2010
VEVP_Zhao2021_AD
VEVP_Zhao2021_AT
VEVP_MOAMMM
```

## Dimensionality wrappers

The `PlaneStrain` and `PlaneStress` wrappers turn a 3D material model into a 2D one, with the mathematical details explained on the [Wrappers](wrappers.md) page.
A bundled model requires no further preparation before it is wrapped.
A custom 3D model becomes wrappable through the `compute_PK1_3D` and `update_state_from_3D!` extension points, which are documented with the rest of the material interface on the [General API](api.md) page.

```@docs
PlaneStrain
PlaneStress
```