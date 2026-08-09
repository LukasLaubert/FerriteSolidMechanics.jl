# Ogden

[`Ogden`](@ref) is a stateless finite-strain hyperelastic material model (`src/models/Ogden.jl`).
It implements Ogden's isotropic hyperelastic model [1, 2, 3] with a quadratic volumetric penalty.
The model accepts an arbitrary number of terms through the parameter vectors `μ` and `α`, while scalar arguments are treated as a one-term shorthand.

### Mathematical formulation

The model uses the deformation gradient $\boldsymbol{F}$, the right Cauchy-Green tensor $\boldsymbol{C}=\boldsymbol{F}^{\mathrm{T}}\boldsymbol{F}$, and the volume ratio $J=\det\boldsymbol{F}=\sqrt{\det\boldsymbol{C}}$.
Further, let $\lambda_i$ denote the principal stretches, with $\lambda_i^2$ being the eigenvalues of $\boldsymbol{C}$.
The implemented free energy density is defined as

```math
\Psi(\boldsymbol{C})
:=
\sum_{a=1}^{N_\mathrm{terms}}
\frac{\mu_a}{\alpha_a}
\left[
J^{-\alpha_a/3}
\sum_{i=1}^{3}\lambda_i^{\alpha_a}
-3
\right]
+
\frac{\kappa}{2}\left[J-1\right]^2 \text{.}
```

The first part is isochoric since $J^{-\alpha_a/3}\lambda_i^{\alpha_a}=\bar{\lambda}_i^{\alpha_a}$ with $\bar{\lambda}_i=J^{-1/3}\lambda_i$.
The second part is a quadratic volumetric penalty governed by the bulk modulus $\kappa$.
The implementation follows the $\mu_a/\alpha_a$ Ogden convention, so the infinitesimal shear modulus is

```math
\mu_0
=
\frac{1}{2}
\sum_{a=1}^{N_\mathrm{terms}}\mu_a\alpha_a \text{.}
```

The one-term case with $\alpha_1=2$ reduces to [`MooneyRivlin`](@ref) with $C_{10}=\mu_1/2$, $C_{01}=0$, and the same $\kappa$.

#### Stress evaluation

For implementation, we define $p_a=\alpha_a/2$, so that $\lambda_i^{\alpha_a}=\left(\lambda_i^2\right)^{p_a}$ turns each stretch power into a power of an eigenvalue of $\boldsymbol{C}$; the stretch sum becomes $\sum_i\lambda_i^{\alpha_a}=\operatorname{tr}\left(\boldsymbol{C}^{p_a}\right)$, evaluated without an explicit eigendecomposition into principal directions.
Using spectral powers of $\boldsymbol{C}$, the isochoric contribution of term $a$ to the second Piola-Kirchhoff stress (PK2) reads

```math
\boldsymbol{S}^{\mathrm{iso}}_a
=
\mu_a J^{-\alpha_a/3}
\left[
\boldsymbol{C}^{p_a-1}
-
\frac{\operatorname{tr}\left(\boldsymbol{C}^{p_a}\right)}{3}
\boldsymbol{C}^{-1}
\right] \text{,}
```

while the volumetric contribution is

```math
\boldsymbol{S}_\mathrm{vol}
=
\kappa J\left[J-1\right]\boldsymbol{C}^{-1} \text{,}
```

The second Piola-Kirchhoff stress is

```math
\boldsymbol{S}
=
\boldsymbol{S}_\mathrm{vol}
+
\sum_{a=1}^{N_\mathrm{terms}}\boldsymbol{S}^{\mathrm{iso}}_a \text{,}
```

while the first Piola-Kirchhoff stress (PK1) used by the assembler is

```math
\boldsymbol{P}=\boldsymbol{F}\boldsymbol{S}
```

and the Cauchy stress is

```math
\boldsymbol{\sigma}
=
\frac{1}{J}\boldsymbol{P}\boldsymbol{F}^{\mathrm{T}} \text{.}
```

## Implementation details

### State variables
`Ogden` is rate-independent and stateless (`create_state` returns `NoState()`).
The `dt` keyword is ignored.

### Tangent formulation
`Ogden` provides an exact analytical tangent.
The tangent differentiates the spectral powers $\boldsymbol{C}^{p_a}$ through divided differences and uses the repeated-eigenvalue limit when two eigenvalues coincide.
At the undeformed state $\boldsymbol{C}=\boldsymbol{I}$, and more generally when two principal stretches coincide, direct AD through the spectral powers can return non-finite derivatives.
The analytical tangent avoids this by using divided differences with repeated-eigenvalue limits.
No AD tangent option is therefore provided.

### Wrapper paths
`PlaneStrain(Ogden(...))` and `PlaneStress(Ogden(...))` are supported through Ogden-specific assembly methods.
These methods call the same exact 3D PK1 stress and tangent as the direct 3D model, then extract or condense the 2D response, respectively.
The public wrapper API is unchanged; however, this internal route is specific to Ogden because it avoids generic AD through spectral powers.
Since the material has no history, `update_state_from_3D!` is a no-op.

## Constructor

```julia
Ogden(μ, α, κ)
```

`μ` and `α` can be real scalars, tuples, or vectors.
They must contain the same nonzero number of finite terms.
Every `α` entry must be nonzero, `κ` must be finite and positive, and the combined small-strain shear modulus `0.5 * sum(μ .* α)` must be positive.
Individual negative fitted terms are accepted when the combined small-strain shear modulus is positive.
This constructor check does not guarantee ellipticity or stability over all finite-strains.

## 2D usage

```julia
material = PlaneStress(Ogden(μ, α, κ))   # plane stress
material = PlaneStrain(Ogden(μ, α, κ))   # plane strain
```
For wrapper mechanics, see [Wrappers](../wrappers.md).

## Model parameters

| Constructor argument | Symbol | Description |
| --- | --- | --- |
| `μ` | $\mu_a$ | Term coefficient in the classic $\mu_a/\alpha_a$ Ogden convention |
| `α` | $\alpha_a$ | Nonzero principal stretch exponent for each term |
| `κ` | $\kappa$ | Bulk modulus |

## Provenance and acknowledgements

The constitutive equations follow Ogden's isotropic hyperelastic model [1].
Ogden also treated compressible rubberlike solids [2]; the decoupled form implemented here follows Flory [4] and Ogden [5].
The Julia implementation was written for FerriteSolidMechanics.jl; no external code was translated or adapted.

The implementation is maintained by Lukas Laubert.
Sebastian Pfaller is gratefully acknowledged for proofreading this page.

## References

[1] Ogden, R. W. (1972). Large deformation isotropic elasticity – on the correlation of theory and experiment for incompressible rubberlike solids. *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences*, 326(1567), 565-584. [doi:10.1098/rspa.1972.0026](https://doi.org/10.1098/rspa.1972.0026)

[2] Ogden, R. W. (1972). Large deformation isotropic elasticity: on the correlation of theory and experiment for compressible rubberlike solids. *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences*, 328(1575), 567-583. [doi:10.1098/rspa.1972.0096](https://doi.org/10.1098/rspa.1972.0096)

[3] Ogden, R. W. (1984). *Non-Linear Elastic Deformations*. Ellis Horwood. (Dover reprint, 1997).

[4] Flory, P. J. (1961). Thermodynamic relations for high elastic materials. *Transactions of the Faraday Society*, 57, 829-838. [doi:10.1039/TF9615700829](https://doi.org/10.1039/TF9615700829)

[5] Ogden, R. W. (1978). Nearly isochoric elastic deformations: application to rubberlike solids. *Journal of the Mechanics and Physics of Solids*, 26(1), 37-57. [doi:10.1016/0022-5096(78)90012-1](https://doi.org/10.1016/0022-5096(78)90012-1)