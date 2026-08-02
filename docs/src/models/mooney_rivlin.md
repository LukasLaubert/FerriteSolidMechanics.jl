# Mooney–Rivlin

[`MooneyRivlin`](@ref) is a stateless finite-strain hyperelastic material model (`src/models/MooneyRivlin.jl`).
It implements the two-parameter Mooney–Rivlin response [1, 2], expressed here in the isochoric–volumetric split with a quadratic volumetric penalty.

### Mathematical formulation

The model uses the deformation gradient $\boldsymbol{F}$, the right Cauchy-Green tensor $\boldsymbol{C}=\boldsymbol{F}^{\mathrm{T}}\boldsymbol{F}$, and the volume ratio $J=\det\boldsymbol{F}=\sqrt{\det\boldsymbol{C}}$.
The first two invariants of $\boldsymbol{C}$ are

```math
I_1
:=
\operatorname{tr}\boldsymbol{C} \text{,}
\qquad
I_2
:=
\frac{1}{2}\left[\left(\operatorname{tr}\boldsymbol{C}\right)^2 - \boldsymbol{C}:\boldsymbol{C}\right] \text{.}
```

Mooney's original theory [1] was expressed in principal stretches; Rivlin [2] later developed the invariant-based form.
The same invariants can equivalently be evaluated from the left Cauchy-Green tensor $\boldsymbol{b}=\boldsymbol{F}\boldsymbol{F}^{\mathrm{T}}$, since $\boldsymbol{b}$ and $\boldsymbol{C}$ have identical principal invariants (see e.g., Holzapfel [3], Ch. 6).

The implemented isochoric invariants are

```math
\bar{I}_1 = J^{-2/3} I_1 \text{,}
\qquad
\bar{I}_2 = J^{-4/3} I_2 \text{,}
```

while the implemented free energy density is

```math
\Psi(\boldsymbol{C})
:=
C_{10}\left[\bar{I}_1 - 3\right]
+ C_{01}\left[\bar{I}_2 - 3\right]
+ \frac{\kappa}{2}\left[J - 1\right]^2 \text{.}
```

Here, $C_{10}$ and $C_{01}$ control the distortional response and $\kappa$ is the bulk modulus of the volumetric penalty.
The small-strain shear modulus at the reference state is $\mu_0 = 2\left[C_{10}+C_{01}\right]$.

#### Stress evaluation

The second Piola-Kirchhoff stress (PK2) is

```math
\boldsymbol{S}
:=
2\frac{\partial\Psi}{\partial\boldsymbol{C}} \text{.}
```

With $\boldsymbol{I}$ denoting the identity tensor, the implemented closed-form PK2 stress is

```math
\boldsymbol{S}
=
2 C_{10} J^{-2/3}
\left[\boldsymbol{I} - \frac{I_1}{3}\boldsymbol{C}^{-1}\right]
+ 2 C_{01} J^{-4/3}
\left[I_1\boldsymbol{I} - \boldsymbol{C} - \frac{2I_2}{3}\boldsymbol{C}^{-1}\right]
+ \kappa J\left[J-1\right]\boldsymbol{C}^{-1} \text{.}
```

The first Piola-Kirchhoff stress (PK1) used by the assembler is

```math
\boldsymbol{P} = \boldsymbol{F}\boldsymbol{S} \text{,}
```

while the Cauchy stress is

```math
\boldsymbol{\sigma}
=
\frac{1}{J}\boldsymbol{P}\boldsymbol{F}^{\mathrm{T}} \text{.}
```

## Implementation details

### State variables
`MooneyRivlin` is rate-independent and stateless (`create_state` returns `NoState()`).
The `dt` keyword is ignored.

### Tangent formulations
Two exact tangent routes are provided:

1. **`tangent=:AD`**: differentiates the PK1 stress response with `Tensors.gradient`.
2. **`tangent=:AT`**: uses the hardcoded material tangent `∂S/∂C` and maps it to `∂P/∂F`.

The constructor default is `tangent=:AD`; representative `stiffness_matrix` checks favor this route for this model.

### Wrapper paths
The generic 2D wrappers use the model's `compute_PK1_3D` and `update_state_from_3D!` hooks.
Since the material has no history, `update_state_from_3D!` is a no-op.

## Constructor

```julia
MooneyRivlin(C10, C01, κ; tangent=:AD)
```

The optional `tangent` keyword must be `:AD` or `:AT`.
The constructor requires finite `C10` and `C01`, a positive sum `C10 + C01`, and a positive finite `κ`.
Individual negative values of `C10` or `C01` are accepted when their sum is positive.
The constructor check does not guarantee stability over all finite-strains.

## 2D usage

```julia
material = PlaneStress(MooneyRivlin(C10, C01, κ))   # plane stress
material = PlaneStrain(MooneyRivlin(C10, C01, κ))   # plane strain
```
For wrapper mechanics, see [Wrappers](../wrappers.md).

## Model parameters

| Constructor argument | Symbol | Description |
| --- | --- | --- |
| `C10` | $C_{10}$ | Coefficient of the first modified invariant |
| `C01` | $C_{01}$ | Coefficient of the second modified invariant |
| `κ` | $\kappa$ | Bulk modulus |
| `tangent` | - | Selects the exact tangent route, `:AD` by default or optional `:AT` |

## Provenance and acknowledgements

The constitutive equations follow the two-invariant form of Mooney [1] and Rivlin [2].
The decoupled isochoric–volumetric split follows Flory [4].
The Julia implementation was written for FerriteSolidMechanics.jl; no external code was translated or adapted.

The implementation is maintained by Lukas Laubert.
Sebastian Pfaller is gratefully acknowledged for proofreading this page.

## References

[1] Mooney, M. (1940). A theory of large elastic deformation. *Journal of Applied Physics*, 11(9), 582-592. [doi:10.1063/1.1712836](https://doi.org/10.1063/1.1712836)

[2] Rivlin, R. S. (1948). Large elastic deformations of isotropic materials. IV. Further developments of the general theory. *Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences*, 241(835), 379-397. [doi:10.1098/rsta.1948.0024](https://doi.org/10.1098/rsta.1948.0024)

[3] Holzapfel, G. A. (2000). *Nonlinear Solid Mechanics: A Continuum Approach for Engineering*. John Wiley & Sons. (Chapter 6, Hyperelastic Materials).

[4] Flory, P. J. (1961). Thermodynamic relations for high elastic materials. *Transactions of the Faraday Society*, 57, 829-838. [doi:10.1039/TF9615700829](https://doi.org/10.1039/TF9615700829)