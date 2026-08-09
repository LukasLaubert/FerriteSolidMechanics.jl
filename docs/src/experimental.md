# Experimental models

!!! note
    The variants under `FerriteSolidMechanics.Experimental` are **not exported** by the top-level package.
    They are alternative implementations kept for reference and comparison, are not part of the stable API, and may be renamed or removed in future versions.

The exported models are listed on the [Material models](models/index.md) page.
Provenance and permissions for these variants follow the corresponding stable model pages: [VEPD Detrez 2010](models/vepd_detrez2010.md), [VEVP Zhao 2021](models/vevp_zhao2021.md), and [VEVP MOAMMM](models/vevp_moammm.md).

!!! warning "Rollback behavior"
    Some VEPD experimental variants do not define model-specific `revert_state!` methods.
    Treat rejected-step rollback behavior as variant-specific unless documented below.

## Importing variants

The variants live in a separate submodule.
To use one, import the symbol explicitly:

```julia
using FerriteSolidMechanics
using FerriteSolidMechanics.Experimental: VEPD_Detrez2010_Optimized, VEPD_Detrez2010_Implicit, VEPD_Detrez2010_ExactVisco, VEPD_Detrez2010_ClosedCVEndStep, VEVP_Zhao2021_AD_Simplified, VEVP_Zhao2021_AT_Matlab, VEVP_MOAMMM_VarsN
```

## `VEVP_MOAMMM_VarsN`

`src/models/experimental/VEVP_MOAMMM_VarsN.jl`.
Archived previous implementation of `VEVP_MOAMMM` with a flat `Vector{Float64}` state buffer (`vars`/`vars_n`) rather than the named typed fields of the exported version.
It follows the same constitutive equations and Drucker–Prager/Perzyna local solver, stores history in the original Fortran `STATEV`-style layout, and uses the same `Tensors.gradient` AD tangent.

## `VEVP_Zhao2021_AD_Simplified`

`src/models/experimental/VEVP_Zhao2021_AD_Simplified.jl`.
Comparison variant of `VEVP_Zhao2021_AD` with the same history fields (`Cik`, `muVk`, `strain_maxk`).
It uses a simpler stress path: the stress returned from the local solve is assembled from trial/previous-state branch stresses, while the updated history fields are still stored for the next step.
It supports the same optional `dt_scale` keyword as `VEVP_Zhao2021_AD` and applies it to `dt` inside the constitutive update.

## `VEVP_Zhao2021_AT_Matlab`

`src/models/experimental/VEVP_Zhao2021_AT_Matlab.jl`.
Archived previous AT implementation with `Matrix{Float64}` storage, following the CAPRICCIO/Zhao Matlab routines more directly than the exported `VEVP_Zhao2021_AT`.
It is kept so the translation is not lost; use `VEVP_Zhao2021_AT` for new work.
It supports the same optional `dt_scale` keyword as `VEVP_Zhao2021_AT`.

## `VEPD_Detrez2010_Optimized`

`src/models/experimental/VEPD_Detrez2010_Optimized.jl`.
Experimental variant that closely follows `VEPD_Detrez2010` under separate type and state names.
It uses the same state quantities (`Fp`, `p`, `Cv`) and the same broad update structure: plastic return loop, linear `(1 + Delta p*N)` plastic map, exponential Maxwell-branch update, damage scaling, and AD tangent assembly.

This variant is not a full drop-in replacement for `VEPD_Detrez2010`: it lacks the positional constructor and a model-specific `revert_state!` method.

## `VEPD_Detrez2010_Implicit`

`src/models/experimental/VEPD_Detrez2010_Implicit.jl`.
Variant of `VEPD_Detrez2010` that keeps the same plastic-return structure and linear `(1 + Delta p*N)` plastic flow map, but uses an implicit/rational Maxwell-branch update: `Cv_next = (Cv_n + (2dt/tau) C) / (1 + 2dt/tau)`.
It also omits the damage cap used by `VEPD_Detrez2010` and does not define a model-specific `revert_state!`.

## `VEPD_Detrez2010_ExactVisco`

`src/models/experimental/VEPD_Detrez2010_ExactVisco.jl`.
Comparison variant of `VEPD_Detrez2010` around the Maxwell-branch evolution.
Both this variant and `VEPD_Detrez2010` use an exponential Maxwell update, so this is not a higher-accuracy viscoelastic option.
It also differs mechanically from `VEPD_Detrez2010`: the Arruda–Boyce network stress is omitted, the network parameters `n_ab` and `mu_ab` are stored but unused, and damage is not clamped to `1.0`.
This variant also does not implement a model-specific `revert_state!`.

## `VEPD_Detrez2010_ClosedCVEndStep`

`src/models/experimental/VEPD_Detrez2010_ClosedCVEndStep.jl`.
Archived copy of the former `VEPD_Detrez2010` default implementation: end-step plastic return mapping with closed-form $\boldsymbol{C}^{\mathrm{v}}$ Maxwell updates.
It is available as `FerriteSolidMechanics.Experimental.VEPD_Detrez2010_ClosedCVEndStep`.

This variant is kept as a performance/reference baseline for the current `VEPD_Detrez2010` default settings `plastic_update=:end_step, maxwell_update=:closed_form_cv`.
In the local scratch benchmark it was about 2--4% faster than the current default implementation, with zero differences in the compared final $\sigma_{11}$, accumulated plastic strain, and $\boldsymbol{C}^{\mathrm{v}}$ branch states.
