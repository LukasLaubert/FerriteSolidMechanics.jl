# J2 Plasticity

[`J2Plasticity`](@ref) implements small-strain von Mises (J2) plasticity with linear isotropic hardening (`src/models/J2Plasticity.jl`).
The stress is integrated by radial return mapping, with a consistent algorithmic tangent [1, 2].

### Mathematical formulation

The model is defined by the yield function $\Phi(\boldsymbol{\sigma}, k)$ as

```math
\Phi(\boldsymbol{\sigma}, k) = \sigma_{\mathrm{eq}} - \left[\sigma_0 + H k\right] \le 0 \text{,}
```

where $\boldsymbol{\sigma}$ is the Cauchy stress, $\sigma_{\mathrm{eq}} = \sqrt{\frac{3}{2} \boldsymbol{s} : \boldsymbol{s}}$ the von Mises equivalent stress, $\boldsymbol{s} = \operatorname{dev}(\boldsymbol{\sigma})$ the deviatoric stress, $\sigma_0$ the initial yield stress, $H$ the linear isotropic hardening modulus, and $k$ the accumulated plastic strain.

The superscript $\mathrm{tr}$ denotes a trial value, and the subscript $n$ denotes the committed start-of-step state.
Here, $\boldsymbol{D}^{\mathrm{e}}$ is the elastic stiffness tensor, $\boldsymbol{\epsilon}$ the total small-strain tensor, $\boldsymbol{\epsilon}^{\mathrm{p}}_n$ the start-of-step plastic strain, $G$ the shear modulus, and $\boldsymbol{D}$ the returned algorithmic tangent.
Using this notation, the return mapping algorithm is evaluated as follows:

1. **Trial elastic state**
   ```math
   \boldsymbol{\sigma}^{\mathrm{tr}} = \boldsymbol{D}^{\mathrm{e}} : \left[\boldsymbol{\epsilon} - \boldsymbol{\epsilon}^{\mathrm{p}}_n\right], \quad \Phi^{\mathrm{tr}} = \sigma^{\mathrm{tr}}_{\mathrm{eq}} - \left[\sigma_0 + H k_n\right]
   ```
   with $\boldsymbol{s}^{\mathrm{tr}}=\operatorname{dev}(\boldsymbol{\sigma}^{\mathrm{tr}})$ and $\sigma^{\mathrm{tr}}_{\mathrm{eq}} =
   \sqrt{\frac{3}{2}\boldsymbol{s}^{\mathrm{tr}}:\boldsymbol{s}^{\mathrm{tr}}}$.
2. **If $\Phi^{\mathrm{tr}} \le 0$ (elastic step)**
   ```math
   \boldsymbol{\sigma} = \boldsymbol{\sigma}^{\mathrm{tr}}, \quad \boldsymbol{\epsilon}^{\mathrm{p}} = \boldsymbol{\epsilon}^{\mathrm{p}}_n, \quad k = k_n, \quad \boldsymbol{D} = \boldsymbol{D}^{\mathrm{e}}
   ```
3. **If $\Phi^{\mathrm{tr}} > 0$ (plastic step)**
   The plastic multiplier increment $\mu$ is defined as
   ```math
   \mu = \frac{\Phi^{\mathrm{tr}}}{H + 3G} \text{;}
   ```
   it scales the trial deviatoric stress by
   ```math
   c_1 = 1 - \frac{3G\mu}{\sigma^{\mathrm{tr}}_{\mathrm{eq}}} \text{,}
   \qquad
   \boldsymbol{s} = c_1 \boldsymbol{s}^{\mathrm{tr}} \text{,}
   \qquad
   \boldsymbol{\sigma} = \boldsymbol{s} + \mathrm{vol}(\boldsymbol{\sigma}^{\mathrm{tr}}) \text{.}
   ```
   Here, $c_1$ is the scalar return mapping scale factor and $\mathrm{vol}(\cdot)$ extracts the spherical stress part.
   The updated yield stress is $\sigma_{\mathrm{eq}} = \sigma_0 + H\left[k_n + \mu\right]$, and the state update follows
   ```math
   \boldsymbol{\epsilon}^{\mathrm{p}} =
   \boldsymbol{\epsilon}^{\mathrm{p}}_n
   + \frac{3}{2}\frac{\mu}{\sigma_{\mathrm{eq}}}\boldsymbol{s} \text{,}
   \qquad
   k = k_n + \mu \text{.}
   ```
   The plastic tangent returned by `compute_stress_tangent` is
   ```math
   \boldsymbol{D}
   =
   \boldsymbol{D}^{\mathrm{e}}
   - 2G\,b\,\boldsymbol{Q}
   - \frac{9G^2}{h\,\sigma_{\mathrm{eq}}^2}\boldsymbol{s}\otimes\boldsymbol{s} \text{,}
   ```
   with
   ```math
   h = H + 3G,\qquad
   b = \frac{3G\mu/\sigma_{\mathrm{eq}}}{1 + 3G\mu/\sigma_{\mathrm{eq}}}
   ```
   and
   ```math
   Q_{ijkl}
   =
   I^{\mathrm{dev}}_{\mathrm{sym},ijkl}
   - \frac{3}{2\sigma_{\mathrm{eq}}^2}s_{ij}s_{kl} \text{.}
   ```

## Implementation details

`J2PlasticityState` holds the plastic strain `ϵᵖ`, Cauchy stress `σ`, and hardening variable `k` in a trial/commit pair.
In the elastic regime, the analytical tangent returns `Dᵉ`; in the plastic regime, it returns the algorithmic tangent from the return mapping update.
Hardening is further linear isotropic: `σʸ = σ₀ + H k`.

The generic 2D wrappers use the model's `compute_PK1_3D` and `update_state_from_3D!` hooks.
`update_state_from_3D!` writes the current trial plastic state from the embedded 3D strain.

## Constructor

```julia
J2Plasticity(E, ν, σ₀, H)
```

## 2D usage

```julia
material = PlaneStress(J2Plasticity(E, ν, σ₀, H))   # plane stress
material = PlaneStrain(J2Plasticity(E, ν, σ₀, H))   # plane strain
```
For wrapper mechanics, see [Wrappers](../wrappers.md).

## Model parameters

| Constructor argument | Symbol | Description |
| --- | --- | --- |
| `E` | $E$ | Young's modulus |
| `ν` | $\nu$ | Poisson's ratio |
| `σ₀` | $\sigma_0$ | Initial yield stress |
| `H` | $H$ | Linear isotropic hardening modulus |

## Provenance

`J2Plasticity`, its constitutive driver implementation, and related documentation are adapted from the Ferrite.jl von Mises plasticity tutorial [3].
The upstream copyright/license notice is retained in the repository `LICENSE`.
The [Developer guide](../developer_guide.md) walks through the implementation as a worked example for new material authors.

## References

[1] de Souza Neto, E. A., Perić, D., & Owen, D. R. J. (2008). *Computational Methods for Plasticity: Theory and Applications*. Wiley.

[2] Simo, J. C., & Hughes, T. J. R. (1998). *Computational Inelasticity*. Springer.

[3] Ferrite.jl documentation. Von Mises plasticity tutorial. [https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/plasticity/](https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/plasticity/)
