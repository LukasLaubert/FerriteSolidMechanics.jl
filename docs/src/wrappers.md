# Plane strain / plane stress wrappers

`PlaneStrain` and `PlaneStress` are wrappers that adapt a **3D** material to a **2D** plane analysis.
Both delegate the constitutive evaluation to the wrapped 3D model and extract the in-plane parts.
The code lives in `src/PlaneStrainStress.jl`.

This page documents:

1. The [embedding of the 2D deformation gradient into 3D](#Embedding-the-2D-deformation-gradient).
2. How [`PlaneStrain` recovers the 2D PK1 stress and consistent tangent](#PlaneStrain:-PK1,-tangent,-state).
3. How [`PlaneStress` solves the out-of-plane equilibrium and statically condenses the tangent](#PlaneStress:-local-Newton-and-condensation).
4. The [state wrapper used by `PlaneStress`](#The-PlaneStressStateWrapper) to track the out-of-plane stretch across Newton iterations and load steps.

For a usage example, see the [2D plate with a hole](tutorials/plate_with_hole.md).
To make a new 3D material wrappable, see [Hooking into the wrappers](#Hooking-into-the-wrappers) below.

## Notation

A 2D problem lives in the `x₁–x₂` plane.
The 3D embedding adds an `x₃`-direction that is *out-of-plane*.
Vectors and tensors follow the conventions of [Tensors.jl](https://github.com/KristofferC/Tensors.jl).

- `F ∈ Tensor{2,2}` – the 2D deformation gradient.
- `C = Fᵀ F` – the 2D right Cauchy–Green tensor.
- `P ∈ Tensor{2,2}` – the 2D first Piola–Kirchhoff stress.
- `S ∈ SymmetricTensor{2,2}` – the 2D second Piola–Kirchhoff stress.
- `σ ∈ SymmetricTensor{2,2}` – the 2D Cauchy stress.

The 3D counterparts carry an overbar: `F̄`, `P̄`, etc.

## Embedding the 2D deformation gradient

Both wrappers start by embedding the 2D `F` into a 3D `F̄` (using `embed_F_2D_to_3D(F, F33)`).
The choice of `F̄₃₃` distinguishes plane strain from plane stress.

### Plane strain with `F̄₃₃ = 1`

The constraint `F̄₃₃ = 1` in

```math
\bar F \;=\;
\begin{pmatrix} F_{11} & F_{12} & 0 \\ F_{21} & F_{22} & 0 \\ 0 & 0 & 1 \end{pmatrix}
```

enforces `ε₃₃ = 0` in the reference configuration.

### Plane stress with unknown `F̄₃₃`

Plane stress enforces `σ̄₃₃ = 0`.
Due to the absence of out-of-plane shear (`F̄₁₃ = F̄₂₃ = F̄₃₁ = F̄₃₂ = 0`) in 2D, this constraint is equivalent to `P̄₃₃ = 0` (since `P̄ = J σ̄ F̄⁻ᵀ` and thus `P̄₃₃ = J σ̄₃₃ (F̄⁻ᵀ)₃₃ = J σ̄₃₃ / F̄₃₃`).
The 3D constitutive update becomes an implicit equation in the scalar `F̄₃₃` by defining a scalar residual function `g` as

```math
g(F̄_{33}) \;:=\; \bar P_{33}(F_{11}, F_{12}, F_{21}, F_{22}, F̄_{33}) \;=\; 0 \text{.}
```

A local Newton iteration solves `g(F̄₃₃) = 0` at each quadrature point per assembly call, using the previously converged value (stored on the [`PlaneStressStateWrapper`](#The-PlaneStressStateWrapper)) as the initial guess.

## `PlaneStrain`: PK1, tangent, state

`PlaneStrain` returns the in-plane PK1 via `P = extract_P_2D(P̄)` and the in-plane consistent tangent `∂P/∂F` via automatic differentiation of the complete embedded 3D response in a single call:

```julia
dP2D_dF2D = Tensors.gradient(F2D_ -> extract_P_2D(Tensor{2,3}(compute_PK1_3D(ps.model, embed_F_2D_to_3D(F2D_, 1.0), dt, state))), F2D)
```

The gradient is taken with respect to the 2D `F`, so the constraint `F̄₃₃ = 1` is naturally propagated through the differentiation.

## `PlaneStress`: local Newton & condensation

`PlaneStress` wraps a Newton solver around the 3D material.
The loop runs **inside** the wrapper's `material_response`, i.e. at every quadrature point of every assembly call.
The local Newton controls are configurable via keyword arguments `tol` and `maxiter` in `PlaneStress(model; tol=1e-10, maxiter=20)`.
The core algorithm for this local solve runs as follows:

```text
F̄₃₃  ←  state.F33_previous      # initial guess from previous load step
g(x) = compute_PK1_3D(model, embed_F_2D_to_3D(F, x), dt, state.inner)[3,3]

repeat up to maxiter:
    residual  ←  g(F̄₃₃)         # constitutive evaluation
    if |residual| < tol: break
    dP̄₃₃/dF̄₃₃  ←  g'(F̄₃₃)       # constitutive evaluation (via AD)
    F̄₃₃  ←  F̄₃₃  −  residual / (dP̄₃₃/dF̄₃₃)
state.F33_current ← F̄₃₃
```

Each Newton correction performs two constitutive evaluations: one primal evaluation for the residual, and one AD evaluation for the derivative.

If the local Newton solve does not converge within `maxiter`, or if `∂P̄₃₃/∂F̄₃₃` drops below the threshold `sqrt(eps(Float64)) ≈ 1.5e-8`, the wrapper throws a `PlaneStressConvergenceError` (a [`LocalAssemblyFailure`](@ref) subtype) containing a `reason` symbol (`:small_newton_derivative` or `:local_newton_nonconvergence`).
Outer load step controllers can catch this directly or call `try_stiffness_matrix`, which returns a named tuple with `converged = false` and the error object.

Once the local iteration converges, the wrapper evaluates the full 3D tangent `∂P̄/∂F̄` and extracts the in-plane PK1 stress.
The consistent 2D tangent is then determined via static condensation (Schur complement) as

```math
\frac{\partial P_{ij}}{\partial F_{kl}}
\;=\;
\frac{\partial \bar P_{ij}}{\partial \bar F_{kl}}
\;-\;
\frac{\partial \bar P_{ij}}{\partial \bar F_{33}}
\frac{\partial \bar P_{33}}{\partial \bar F_{kl}}
\;\Big/\;
\frac{\partial \bar P_{33}}{\partial \bar F_{33}}
\quad \forall\, i,j,k,l \in \{1,2\} \text{,}
```

which is implemented by `condense_tangent_2D` in `src/PlaneStrainStress.jl`.
Note that if the out-of-plane stiffness component in this full 3D tangent drops below the numerical threshold, this condensation step will independently throw a `PlaneStressConvergenceError` with the reason `:singular_condensation`.

## The `PlaneStressStateWrapper`

Both `PlaneStrain` and `PlaneStress` update the wrapped material's trial state with a single call to `update_state_from_3D!(state, model, F̄, dt)`.
That call is the last step of the wrapper's `material_response`, after stress and tangent have been computed.
Final state commits happen canonically via `update_states!`.

`PlaneStrain` does not need a custom state wrapper because its out-of-plane stretch `F̄₃₃ = 1` is fixed. In contrast, `PlaneStress` wraps the 3D material's state to track the variable out-of-plane stretch `F̄₃₃`:

```julia
mutable struct PlaneStressStateWrapper{S} <: AbstractMaterialState
    inner::S                       # the wrapped 3D material state
    F33_current::Float64           # F̄₃₃ at the most recent trial
    F33_previous::Float64          # F̄₃₃ at the last converged step
end
```

The wrapped 3D state (`inner`) is updated only *after* the local Newton loop has successfully converged.
If the outer load step is rejected, `revert_state!` rolls back both `F33` and the wrapped 3D state. `F33_previous` is then used as the initial guess for the next attempt.

## Hooking into the wrappers

Two methods connect a 3D material to the wrapper pipeline, beyond the standard state / element hooks:

```julia
compute_PK1_3D(material, F̄, dt, state) -> P̄
update_state_from_3D!(state, material, F̄, dt)
```

The first returns the 3D first Piola–Kirchhoff stress.
The second writes the wrapped material's current/trial variables at the converged 3D deformation.
Final commit happens through the usual `update_state!` / `update_states!` path.

Small-strain models (and `AbstractHyperelastic` subtypes) do not need to implement either method.
For a small-strain model, a generic fallback derives both from `material_response`: it forms `ε = sym(F̄ - I)`, evaluates the constitutive response, and converts the resulting Cauchy stress to PK1.
For an `AbstractHyperelastic` subtype, the fallback derives `P̄` from the strain energy `Ψ`, and the state update performs no work because the model is stateless.
Every other finite-strain model must implement both methods explicitly, as no universal conversion from deformation gradient to stress exists for them.

Both wrappers differentiate `compute_PK1_3D` with ForwardDiff.
The deformation gradient therefore carries `ForwardDiff.Dual` entries, while the committed history remains `Float64`.
If a model constructs a state object or scratch cache that mixes both types, the call will fail with a `MethodError`.
As this is not a [`LocalAssemblyFailure`](@ref), `try_stiffness_matrix` does not catch it.
A wrappable model must therefore accept both types in its internal state and scratch cache; see the [Developer guide](developer_guide.md#wrapper-hooks) for details.

Check the [Stable models](models/index.md#Stable-models) table for a full list of bundled 3D materials that support the wrapper API.

Most models use the generic wrapper assembly explained above.
`Ogden` is an exception: `PlaneStrain(Ogden(...))` and `PlaneStress(Ogden(...))` dispatch to exact-tangent methods in `src/models/Ogden.jl`, avoiding AD through spectral powers (which are non-finite at repeated principal stretches).

`VEVP_Zhao2021_AT` does not support the generic wrapper AD path: its analytic-tangent implementation is `Float64`/matrix-based.
It provides a specialized `PlaneStrain(VEVP_Zhao2021_AT(...))` assembly path, but `PlaneStress` is not implemented for `VEVP_Zhao2021_AT`.
Use `VEVP_Zhao2021_AD` for plane stress analyses.

If you write a new 3D model, see the [Developer guide](developer_guide.md) – its [wrapper hooks](developer_guide.md#wrapper-hooks) section shows the generic small-strain derivation and explains why a finite-strain model must supply both methods itself.