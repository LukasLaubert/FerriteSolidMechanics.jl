# Neo–Hookean

The compressible Neo–Hookean hyperelastic material model [`NeoHooke`](@ref) (`src/models/NeoHooke.jl`) is parameterized by Young's modulus `E` and Poisson's ratio `ν`.

### Mathematical formulation

The material's behavior is derived from its free energy density function $\Psi(\boldsymbol{C})$ as

```math
\Psi(\boldsymbol{C}) = \frac{1}{2}\mu \left[I_1 - 3 - 2\ln J\right] + \frac{1}{2}\lambda \left[J - 1\right]^2 \text{,}
```

where $\boldsymbol{F}$ is the deformation gradient, $\boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}$ the right Cauchy–Green tensor, $I_1 = \mathrm{tr}(\boldsymbol{C})$ its first invariant, $J = \det(\boldsymbol{F}) = \sqrt{\det(\boldsymbol{C})}$ the volume ratio, and $\mu, \lambda$ the Lamé parameters.

The parameters $\mu$ (shear modulus) and $\lambda$ (first Lamé parameter) are computed from the input Young's modulus $E$ and Poisson's ratio $\nu$ as

```math
\mu = \frac{E}{2\left[1 + \nu\right]}, \qquad \lambda = \frac{E\nu}{\left[1 + \nu\right]\left[1 - 2\nu\right]} \text{.}
```

## Implementation details

`NeoHooke` is rate-independent and stateless (`create_state` returns `NoState()`).
The tangent is obtained analytically via `Tensors.hessian` over the free energy `Ψ(C)`, yielding both the second Piola-Kirchhoff stress (PK2) `S = 2 ∂Ψ/∂C` and the material derivative `∂S/∂C` in a single call.
The generic 2D wrappers use the model's `compute_PK1_3D` and `update_state_from_3D!` hooks.
Since the material has no history, `update_state_from_3D!` is a no-op.

## Constructor

```julia
NeoHooke(E, ν)
```

## 2D usage

```julia
material = PlaneStress(NeoHooke(E, ν))   # plane stress
material = PlaneStrain(NeoHooke(E, ν))   # plane strain
```
For wrapper mechanics, see [Wrappers](../wrappers.md).

## Model parameters

| Constructor argument | Symbol | Description |
| --- | --- | --- |
| `E` | $E$ | Young's modulus |
| `ν` | $\nu$ | Poisson's ratio |

## Provenance

The constitutive energy, stress/tangent derivation, element-level linearization, and related documentation are adapted from the Ferrite.jl hyperelasticity tutorial [1].
The upstream copyright/license notice is retained in the repository `LICENSE`.

## References

[1] Ferrite.jl documentation. Hyperelasticity tutorial. [https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/hyperelasticity/](https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/hyperelasticity/)