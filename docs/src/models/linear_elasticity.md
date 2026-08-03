# Linear Elasticity

Standard isotropic linear elasticity (`src/models/LinearElasticity.jl`).

Two constructors are provided:

- [`Hooke`](@ref) – 3D isotropic linear elasticity, parameterized by
  `E` and `ν`.
- [`Hooke2D`](@ref) – 2D isotropic linear elasticity, with a
  `plane_stress::Bool` keyword. By default it builds a plane strain
  stiffness; pass `plane_stress=true` for the analytical plane stress
  matrix.

### Mathematical formulation

The 3D `Hooke` model uses the fourth-order isotropic Lamé stiffness $C_{ijkl} = \lambda\delta_{ij}\delta_{kl} + \mu\left[\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}\right]$, where $C_{ijkl}$ maps strain to stress, $\delta_{ij}$ is the Kronecker delta, and the indices $i,j,k,l$ run over the spatial dimensions.
The Lamé parameters are $\lambda = E\nu / \left[\left[1+\nu\right]\left[1-2\nu\right]\right]$ and $\mu = E / \left[2\left[1+\nu\right]\right]$. `Hooke2D` uses the same Lamé form for plane strain and an analytical Voigt matrix for plane stress.

## Implementation details

`Hooke` and `Hooke2D` are stateless, i.e., `create_state` returns `NoState()`.
The stress given by `σ = C : ε`, using the constant material tangent `C`.
The generic small-strain element routine accumulates `(ε(δu_i) : C : ε(δu_j)) * α_dΩ` at each quadrature point.
Here, `ε(δu_i)` is the symmetric strain associated with basis function displacement `δu_i`, and `α_dΩ` is the quadrature weight including the [alpha scaling](../faq.md#The-alpha_value-extension-point) and the independent volume measure.
`Hooke` and `Hooke2D` implement the material-interface method `is_linear` with `true`, so the assembler preassembles `K_linear` once during `create_assembler`; subsequent `stiffness_matrix` calls only handle nonlinear cells and add `K_linear * u` to the residual.

## Constructor

```julia
Hooke(E, ν)                       # 3D isotropic linear elasticity
Hooke2D(E, ν; plane_stress=false) # 2D plane strain
Hooke2D(E, ν; plane_stress=true)  # 2D plane stress
```

## Hooke2D analytical plane stress

When `Hooke2D` is constructed with `plane_stress=true`, the stiffness matrix is built in Voigt form as

```math
C^{\mathrm{ps}} = \frac{E}{1-\nu^2}
\begin{pmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \left[1-\nu\right]/2
\end{pmatrix}
```

and converted to a `SymmetricTensor{4,2}` via `fromvoigt`.
Otherwise the plane strain stiffness is the standard Lamé form as

```math
C^{\mathrm{pe}}_{ijkl} = \lambda\,\delta_{ij}\delta_{kl} + \mu\,
\left[\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}\right]
```

with $\lambda = E\nu / \left[\left[1+\nu\right]\left[1-2\nu\right]\right]$
and $\mu = E / \left[2\left[1+\nu\right]\right]$.

## Hooke2D vs PlaneStrain(Hooke) / PlaneStress(Hooke)

`Hooke2D` is a direct analytical 2D form, while `PlaneStrain(Hooke)` / `PlaneStress(Hooke)` are the wrappers that take a 3D `Hooke` and reduce it to a 2D problem through 2D-to-3D embedding and, for plane stress, local Newton iteration plus static condensation (see [Wrappers](../wrappers.md)).
For a linear elastic material the two approaches are mathematically equivalent: both use the same infinitesimal-strain stiffness, so the in-plane stresses agree to within numerical round-off (`rtol ≈ 10⁻¹⁰`) at any displacement level.
The regression test in `test/test_dimensionality_wrappers.jl` verifies this on a single-element uniaxial load case.

!!! note "Prefer `Hooke2D` for production 2D linear elasticity"
    The generic wrappers always run through the nonlinear wrapper path:
    `PlaneStrain(Hooke(...))` and `PlaneStress(Hooke(...))` are treated as nonlinear by the assembler, even though the wrapped material is linear.
    This way, there's no preassembly into `K_linear`; the `PlaneStrain` / `PlaneStress` element routine is evaluated on every Newton iteration, and `PlaneStress` also solves the local out-of-plane condition.
    For linear 2D analyses, use `Hooke2D` directly unless you specifically want to exercise the generic wrapper contract.

## Provenance

The 3D `Hooke` element routine and related documentation are adapted from the Ferrite.jl linear elasticity tutorial [1].
The upstream copyright/license notice is retained in the repository `LICENSE`.
`Hooke2D` is a package-local model with analytical plane strain and plane stress moduli.

## References

[1] Ferrite.jl documentation. Linear elasticity tutorial. [https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/linear_elasticity/](https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/linear_elasticity/)