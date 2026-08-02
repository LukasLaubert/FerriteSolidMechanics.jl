# Arruda–Boyce

[`ArrudaBoyce`](@ref) is a stateless finite-strain hyperelastic material model (`src/models/ArrudaBoyce.jl`).
It implements Arruda and Boyce's eight-chain network response [1] with an isochoric network term and a quadratic volumetric penalty.

### Mathematical formulation

The model uses the deformation gradient $\boldsymbol{F}$, the right Cauchy-Green tensor $\boldsymbol{C} = \boldsymbol{F}^{\mathrm{T}}\boldsymbol{F}$, and the Jacobian $J = \det\boldsymbol{F} = \sqrt{\det\boldsymbol{C}} \text{.}$
Its isochoric first invariant is defined as

```math
\bar{I}_1 = J^{-2/3}\operatorname{tr}\boldsymbol{C} \text{.}
```

The implemented free energy density is defined as

```math
\Psi(\boldsymbol{C})
=
\mu \sum_{i=1}^{5}\frac{c_i}{N^{i-1}}
\left[\bar{I}_1^i - 3^i\right]
+ \frac{\kappa}{2}\left[J - 1\right]^2 \text{,}
```

while the truncated inverse-Langevin series coefficients are

```math
c_1=\frac{1}{2} \text{,}\quad
c_2=\frac{1}{20} \text{,}\quad
c_3=\frac{11}{1050} \text{,}\quad
c_4=\frac{19}{7000} \text{,}\quad
c_5=\frac{519}{673750} \text{.}
```

Here, $\mu$ is the network shear modulus, $\kappa$ the bulk modulus, and $N$ the limiting chain extensibility parameter.
It sets the stiffness scale of the eight-chain network; the resulting small-strain shear modulus is $\mu_0 = \mu\left(1 + \frac{3}{5N} + \mathcal{O}(N^{-2})\right)$, which approaches $\mu$ as $N\to\infty$.
The subtracted constants $3^i$ make the isochoric part vanish at $\boldsymbol{C}=\boldsymbol{I}$, where $\bar{I}_1 = 3$.
The volumetric part vanishes there as well, so $\Psi(\boldsymbol{I})=0$ holds in the undeformed state.

The second Piola-Kirchhoff stress (PK2) $\boldsymbol{S}$ follows from differentiating the free energy density as

```math
\boldsymbol{S}
=
2\frac{\partial\Psi}{\partial\boldsymbol{C}} \text{.}
```

The first Piola-Kirchhoff stress (PK1) used by the assembler is

```math
\boldsymbol{P} = \boldsymbol{F}\boldsymbol{S} \text{,}
```

while the Cauchy stress follows as

```math
\boldsymbol{\sigma}
=
\frac{1}{J}\boldsymbol{P}\boldsymbol{F}^{\mathrm{T}} \text{.}
```

## Implementation details

`ArrudaBoyce` is rate-independent and stateless (`create_state` returns `NoState()`).
The `dt` keyword is ignored.

By default (`tangent=:AT`), the element assembly uses a hardcoded exact material tangent for `∂S/∂C` and maps it to the tangent `∂P/∂F`.
Using the keyword argument `tangent=:AD`, the same stress formulation is differentiated by `Tensors.gradient` over the PK1 response.
Both tangent modes are exact.
Depending on your case, `:AD` may compute faster than the constructor default `:AT`.
The generic 2D wrappers use the model's `compute_PK1_3D` and `update_state_from_3D!` hooks.
Since the material has no history, `update_state_from_3D!` is a no-op.

## Constructor

```julia
ArrudaBoyce(μ, κ, N; tangent=:AT)
```

All three constructor arguments must be positive.
The optional `tangent` keyword must be `:AT` or `:AD`.

## 2D usage

```julia
material = PlaneStress(ArrudaBoyce(μ, κ, N))   # plane stress
material = PlaneStrain(ArrudaBoyce(μ, κ, N))   # plane strain
```
For wrapper mechanics, see [Wrappers](../wrappers.md).

## Model parameters

| Constructor argument | Symbol | Description |
| --- | --- | --- |
| `μ` | $\mu$ | Network shear modulus |
| `κ` | $\kappa$ | Bulk modulus |
| `N` | $N$ | Limiting chain extensibility parameter |

## Provenance and acknowledgements

The constitutive equations follow Arruda and Boyce's eight-chain model [1].
The decoupled isochoric–volumetric split follows Flory [2].
The Julia implementation was written for FerriteSolidMechanics.jl; no external code was translated or adapted.

The implementation is maintained by Lukas Laubert.
Sebastian Pfaller is gratefully acknowledged for proofreading this page.

## References

[1] Arruda, E. M., & Boyce, M. C. (1993). A three-dimensional constitutive model for the large stretch behavior of rubber elastic materials. *Journal of the Mechanics and Physics of Solids*, 41(2), 389-412. [doi:10.1016/0022-5096(93)90013-6](https://doi.org/10.1016/0022-5096(93)90013-6)

[2] Flory, P. J. (1961). Thermodynamic relations for high elastic materials. *Transactions of the Faraday Society*, 57, 829-838. [doi:10.1039/TF9615700829](https://doi.org/10.1039/TF9615700829)
