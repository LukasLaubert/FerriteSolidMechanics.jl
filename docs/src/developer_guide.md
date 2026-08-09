# Developer guide

This page is for users who want to add a new constitutive model to `FerriteSolidMechanics.jl`.
It explains the required interface that every material must implement, then walks through a practical example: the [`J2Plasticity`](@ref) model.

!!! info
    We recommend opening an existing material model from `src/models/` side-by-side with this guide to see how these concepts translate to practical code.

If you are using the package rather than extending it, skip this page and start with the [Concepts](concepts.md) or [Tutorials](tutorials/plate_with_hole.md).

## Naming conventions

!!! tip "Start in experimental – avoid conventions"
    You can skip any instructions on this page to contribute a model in the first place.
    Drop it into `src/models/experimental/` and it is exempt from the naming conventions and the testing and documentation requirements below (see [Experimental models](#Experimental-models)).
    Adopt the conventions only once you decide to promote to the official catalog.

When adding a bundled (non-experimental) material model, please stick to these naming conventions:

**1. Material Model Constructors**
* **Classic / Universal Models**: Use standard PascalCase for well-known models (e.g., `J2Plasticity`, `ArrudaBoyce`, `MooneyRivlin`).
* **Specific Literature Models**: Use the format `[Class]_[Author/Project][Year]_[Variant]`, where `[Class]` refers to the physics acronym (e.g. VEVP for viscoelastic–viscoplastic), `[Author/Project]` identifies the original author or project name, and `[Variant]` is an optional variant specifier.
Adding `[Year]` is recommended to avoid naming collisions but might be left out for unique project names.
Examples:
  * `VEPD_Detrez2010` (Class: VEPD, AuthorYear: Detrez2010)
  * `VEVP_MOAMMM` (Class: VEVP, Project: [MOAMMM](https://www.moammm.eu/))
  * `VEVP_Zhao2021_AT` (Class: VEVP, AuthorYear: Zhao2021, Variant: AT (analytical tangent))

**2. Auxiliary Data Structures**
Any auxiliary structs required by a model (such as state variables or local caches) should be named by directly appending a PascalCase suffix to the exact constructor name, without additional underscores:
* State wrapper: `[ConstructorName]State` (e.g., `VEVP_Zhao2021_ATState`)
* Internal state: `[ConstructorName]StateInternal` (e.g., `VEPD_Detrez2010StateInternal`)
* Caches / Temp data: `[ConstructorName]Cache` or `[ConstructorName]StiffnessDataGP`

**3. Documentation**
The documentation filename is based on the constructor name from rule 1, converted to lowercase `snake_case`.
Avoid the `_[Variant]` suffix if multiple variants share a documentation page, such as `VEVP_Zhao2021_AT/AD.jl` → `vevp_zhao2021.md` ("VEVP Zhao 2021").
The sidebar title on the website (generated via the `make.jl` file) is the model name formatted with spaces instead of underscores, preserving acronym capitalizations and separating the year.
* `J2Plasticity.jl` → `j2_plasticity.md` ("J2 Plasticity")
* `VEPD_Detrez2010.jl` → `vepd_detrez2010.md` ("VEPD Detrez 2010")

## Which level should a new model target?

The table below lists the available implementation levels, ordered from simplest to most general.
Pick the highest row that fits; drop to a lower one only when the physics requires it.

| Model class | Level to implement |
| --- | --- |
| Stateless hyperelasticity | [`AbstractHyperelastic`](@ref) + `Ψ(C)` |
| Local small-strain models | [`material_response`](@ref) with [`SmallStrain`](@ref)() |
| Local finite-strain models | [`material_response`](@ref) with [`FiniteStrain`](@ref)() |
| Element-structured assembly | [custom `_assemble_element!`](#Custom-element-assembly) (see `VEVP_Zhao2021_AT`) |
| Mixed / multi-field formulations | Not supported by the current assembler |
| Nonlocal regularization | [custom `_assemble_element!`](#Custom-element-assembly) (needs cross-point coupling) |
| Rate-form / hypoelastic models | [`material_response`](@ref) carrying `F`, `σ` in state (see `VEPD_Detrez2010`) |
| Cohesive-zone / interface models | Not supported by the current assembler |
| Beams / shells | Not supported by the current assembler |

The `kinematics` interface is deliberately open: new strain measures (and their weak-form contributions) can be added as new [`AbstractKinematics`](@ref) subtypes.

## The required interface

Every material constructor follows this convention: material parameters are positional, integration modes / other options are keyword arguments with defaults.
See [Concepts → Material constructor conventions](concepts.md#Material-constructor-conventions) for the convention and examples.

A material is a `struct` that subtypes `AbstractMaterial`; history objects subtype `AbstractMaterialState`.
There are two levels at which a material can plug into the assembler:

1. **The constitutive level.**
   Implement [`material_response`](@ref), i.e., one function mapping the strain measure and the committed history at a single quadrature point to `(stress, tangent, new_state)`, and declare the strain measure with [`kinematics`](@ref) (`SmallStrain()` or `FiniteStrain()`).
   The assembler's built-in element routine then handles the quadrature loop, integration weights (including alpha scaling), weak form, and trial-state bookkeeping automatically.
   (Comparable to Abaqus' `UMAT`)
2. **The element level.**
   Implement [`_assemble_element!`](@ref) yourself when the *element* structure is material-specific.
   The bundled `VEVP_Zhao2021_AT` provides an example: its hand-derived tangent is assembled from nodal shape-gradient blocks and cannot be expressed as a per-point `∂P/∂F`.
   (Comparable to Abaqus' `UEL`)

Stateless hyperelastic models have a third, even shorter path: subtype [`AbstractHyperelastic`](@ref) and implement only the strain energy density `Ψ(C, material)` (see below).

!!! note "Design acknowledgement"
    The constitutive-level interface follows the conventions established by Knut Andreas Meyer's [MaterialModelsBase.jl](https://github.com/KnutAM/MaterialModelsBase.jl) (single response function returning `(stress, tangent, new_state)`, strain-measure-driven stress/tangent pairs, material cache hook).
    Materials written for that interface here run through the [`FromMaterialModelsBase`](@ref) wrapper; state handling differences (e.g., mutable trial and commit vs immutable return values) are bridged by this wrapper.

### Feature support

Most solver-side behavior is provided automatically once a material implements the required hooks.
Some capabilities, however, depend on the material model or wrapper explicitly supporting them:

| Capability | What must be implemented or supported |
| --- | --- |
| Assembly | `material_response` + `kinematics` (generic element routine), **or** a custom `_assemble_element!` |
| Stress output | Automatic via the generic fallback; override [`material_stress`](@ref) for a cheap tangent-free path when the tangent is expensive (AD models) or the output convention is model-specific |
| Trial state tracking | Automatic for the `current`/`previous` pattern; override [`set_trial!`](@ref) for other state layouts |
| History commit and rollback | `create_state`, `update_state!`, and `revert_state!` for stateful materials |
| Linear preassembly | `is_linear(material) = true` |
| Symmetric-tangent fast path | `tangent_symmetry(material) = MajorSymmetric()`, for small-strain materials whose `D` satisfies `D[i,j,k,l] == D[k,l,i,j]` |
| Rate dependence | The material update must actually use `dt` |
| Generic `PlaneStrain` / `PlaneStress` wrappers | `compute_PK1_3D` and `update_state_from_3D!` for the wrapped 3D material (derived automatically from `material_response` for any `SmallStrain` material model and from the strain energy `Ψ` for `AbstractHyperelastic` subtypes; every other finite-strain model must implement them) |
| Local failure reporting | The material or wrapper must throw a [`LocalAssemblyFailure`](@ref) ([`try_stiffness_matrix`](@ref) then converts this to `converged = false`) |

### Required functions (linear elastic model)

These three methods fully define a linear elastic material model:

```julia
is_linear(::MyMaterial) = true
kinematics(::MyMaterial) = SmallStrain()

function material_response(mat::MyMaterial, ε::SymmetricTensor{2}, state, dt, cache=nothing)
    return mat.C ⊡ ε, mat.C, state
end
```

`is_linear = true` tells the assembler to build the element stiffness matrix exactly once.
Consequently, this setting is only valid for models with a constant tangent.

See `src/models/LinearElasticity.jl`, which defines both `Hooke` and `Hooke2D`, for complete examples.

!!! note
    For linear elastic material models, the package's assembler preassembles the element stiffness `ke` once through the generic element routine and stores it globally in `K_linear`.
    During the solver steps, the residual is computed as `r = K_linear * u` and the stress is evaluated using `material_response`.

### Required functions (nonlinear elastic material)

For a generic nonlinear elastic material that is not history-dependent, you only need to define the material properties, flag it as nonlinear, and implement the constitutive response.
No state tracking functions are needed:

```julia
is_linear(::MyNonlinearMaterial) = false
kinematics(::MyNonlinearMaterial) = FiniteStrain() # or SmallStrain()

function material_response(mat::MyNonlinearMaterial, F::Tensor{2,3}, state, dt, cache=nothing)
    # Compute P and dP_dF (or the small-strain equivalents)
    return P, dP_dF, state
end
```

To implement a hyperelastic material, simply subtype [`AbstractHyperelastic`](@ref) and implement the strain energy function `Ψ(C, material)` (see [The hyperelastic quick path](@ref)).

### Required functions (history-dependent material)

For a history-dependent (stateful) material model, you must explicitly define the state structure and the methods to create and manage its history:

```julia
struct MyMaterial <: AbstractMaterial end  # replace "Custom" with your name here and in the following
mutable struct CustomState <: AbstractMaterialState
    current::CustomStateInternal
    previous::CustomStateInternal
end

is_linear(::MyMaterial) = false
kinematics(::MyMaterial) = FiniteStrain()   # or SmallStrain()

create_state(::MyMaterial) = CustomState(CustomStateInternal(...), CustomStateInternal(...)) # creates a new state object

function update_state!(s::CustomState)  # must include all state variables
    s.previous = s.current
    return nothing
end

function revert_state!(s::CustomState)  # must include all state variables
    s.current = s.previous
    return nothing
end

function material_response(mat::MyMaterial, F::Tensor{2,3}, state::CustomState, dt, cache=nothing)
    # Read the committed history (state.previous), run the local update,
    # compute P and ∂P/∂F (analytic or AD), while not committing state.
    return P, dP_dF, new_state
end
```

The package's generic element routine stores the returned `new_state` as the trial state via [`set_trial!`](@ref).
By default, `set_trial!(state, new_state)` assigns `state.current = new_state`, supporting the `current`/`previous` pattern above.
Models with a different state layout override `set_trial!` (see `VEPD_Detrez2010` for a field-wise example); materials that write their trial in place return the passed `state`.

A cheap, tangent-free [`material_stress`](@ref) method is recommended for history models whose tangent is expensive (see `VEPD_Detrez2010` and `VEVP_Zhao2021_AD` for examples, as well as [Stress postprocessing](#Stress-postprocessing) below); otherwise the generic stress fallback evaluates `material_response` for postprocessing and discards the tangent and trial state (-> more expensive).

The "trial and commit state" pattern (`current` and `previous`) is convention for history-dependent models.
If the internal state is defined as an immutable struct (as shown in the `J2Plasticity` example below), the default `copy_state!` function automatically handles copying all fields.
Override `copy_state!` only when the default fallback is not enough (e.g., when a field is a mutable heap-allocated array that requires an explicit `deepcopy`).

The state object (`CustomState`) should be mutable when your functions update its fields in place, as shown above.
Immutable state objects are also possible, but then `update_state!(state)` **must** return a replacement `AbstractMaterialState`.
In this case, `update_states!(assembler)` stores that returned state back into the quadrature point state vector.
If the state cannot be updated in-place (e.g., an immutable struct) and `update_state!` returns `nothing`, the package assumes no state update is required.

### The hyperelastic quick path

For a stateless hyperelastic model, you can subtype [`AbstractHyperelastic`](@ref) and implement one function:

```julia
struct CustomRubber2026 <: AbstractHyperelastic
    μ::Float64
    κ::Float64
end

function FerriteSolidMechanics.Ψ(C, mp::CustomRubber2026)   # strain energy density Ψ(C)
    J = sqrt(det(C))
    Ic̄ = det(C)^(-1/3) * tr(C)
    return 0.5 * mp.μ * (Ic̄ - 3) + 0.5 * mp.κ * (J - 1)^2
end
```

If you follow this hyperelastic path, stress and tangent are automatically derived with `Tensors.hessian`/`Tensors.gradient` by the package, and element assembly, stress output, and the `PlaneStrain`/`PlaneStress` wrappers are automatically functional.
Optionally, every derived quantity can be overridden with analytic expressions (e.g. for speed) – one method at a time:
`_second_piola(C, mp)` (stress only), `constitutive_driver(C, mp)` (`S` and `∂S/∂C`), or `_finite_strain_pk1_tangent(F, mp, dt)` (`P` and `∂P/∂F` directly).
For example, the bundled `NeoHooke`, `ArrudaBoyce`, `MooneyRivlin`, and `Ogden` models are all `AbstractHyperelastic` subtypes, but only `NeoHooke` keeps the AD energy path, while the other three override the drivers analytically.

### Optional hooks (for `PlaneStrain` / `PlaneStress` wrappers)

If you want your 3D material to be wrappable by [`PlaneStrain`](@ref) or [`PlaneStress`](@ref), also implement

```julia
compute_PK1_3D(mat, F, dt, state) -> P̄
update_state_from_3D!(state, mat, F, dt)
```

The two in-plane wrappers call `compute_PK1_3D` to evaluate the 3D PK1, then embed the 2D gradient, and extract the in-plane parts.
Material models implement `update_state_from_3D!` to update its internal state via the trial/current variables using the converged 3D deformation gradient.
(Not to be confused with `update_states!`, which must commit the state after Newton converges.)
See [Wrappers](wrappers.md) for the math.

## The generic element routine

For a material implementing `material_response` + `kinematics`, the generic element routine does the following per quadrature point:

1. Compute the integration weight `α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)` (alpha_value is 1 by default if not specified).
2. Evaluate the strain measure (`ε = sym(∇u)` for `SmallStrain`, `F = I + ∇u` for `FiniteStrain`).
3. Call `material_response(material, ε_or_F, state, dt, cache)`.${}^1$
4. Store the returned trial state (see the three accepted return forms in the [`material_response`](@ref) docstring).
5. Add the weak-form contributions to `ke`/`re` (`δε ⊡ σ` / `δε ⊡ D ⊡ Δε` for small-strain, exploiting the symmetry of `D`; `∇δu ⊡ P` / `∇δu ⊡ ∂P∂F ⊡ ∇δu` for finite-strain).

*${}^1$ The `cache` argument is created once per element via [`allocate_material_cache`](@ref) (default `nothing`).
You can override this function to return a custom cache object if your material needs to reuse preallocated variables across quadrature points.*

## Custom element assembly

The package calls `_assemble_element!(ke, re, states, material, cellvalues, alphavalues, u, dt)` once per owned nonlinear cell.
The generic fallback described above provides this function for every model that includes the `material_response` hook.
We therefore recommend implementing a custom `_assemble_element!` method for your material only when the element structure itself is material-specific.
`VEVP_Zhao2021_AT` is a bundled example: its analytical tangent is assembled from nodal shape-gradient blocks (`calc_vevp_zhao_stiffness_block_jk`), which does not reduce to a quadrature point `∂P/∂F`.
Nonlocal regularization is another model class that may require a custom element assembly.
Mixed and multi-field formulations are currently out of reach of this hook: the assembler builds one `CellValues` per `SubDofHandler` from its single field interpolation, so a `SubDofHandler` carrying more than one field is rejected when the assembler is created.

Arguments passed to `_assemble_element!` are

- `ke`, `re`: the preallocated element stiffness matrix and residual vector
(A custom method must compute and add the element contributions to them.
`re` is `nothing` in the linear preassembly path, which only reaches materials with `is_linear(material) == true`; `ke` is never `nothing`.
There is no residual-only entry point that would skip `ke`: `compute_forces` wraps `stiffness_matrix` and discards the tangent.);
- `states`: `Vector{AbstractMaterialState}` of length `nquadpoints` for nonlinear materials, or `NoState()` in the linear preassembly path;
- `material`: the material model instance (e.g., your `MyMaterial` struct);
- `cellvalues`: the Ferrite `CellValues` object for the current SubDofHandler (already reinitialized for the cell);
- `alphavalues`: the integration weight scaling factors (if any) for the current SubDofHandler, or `nothing`;
- `u`: the element displacement vector;
- `dt`: the load step time increment.

### Element arrays

`ke` and `re` are a dense `Matrix{Float64}` and `Vector{Float64}` from a preallocated workspace, sized to the cell's own dof count (`size(ke, 1) == getnbasefunctions(cellvalues)`).

A custom `_assemble_element!` method is responsible for

- looping over quadrature points;
- evaluating strain or deformation gradient;
- computing stress and tangent;
- multiplying the quadrature point contributions by the integration weight via `alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)` (The generic assembler does not apply this automatically for custom element routines.);
- adding the weak-form contributions to the element arrays `ke` and `re`;
- writing trial state values to `state.current`
(Do not commit the state here; state commitment happens after the global Newton step converges via `update_states!`.).

The return value of `_assemble_element!` is ignored.
You must update the provided arrays and state objects in-place and return `nothing`.

Some bundled materials delegate the nested shape function loops to internal helper functions after computing the quadrature point stress and tangent.
These internal helpers expect specific kinematics: either a PK1 stress and `∂P/∂F` tangent combined with standard shape gradients for finite-strain, or a small-strain stress and `∂σ/∂ε` tangent combined with symmetric shape gradients.
However, material-specific updates, local Newton solves, wrapper condensation, and state writes must remain in your custom `_assemble_element!` method.

## State variables

State variables are allocated per quadrature point and follow a "trial and commit" pattern.
They represent the uncommitted trial state (`current`) and the last converged state (`previous`).
The assembler does not read or write the state variables directly; the material model owns them.

- `create_state` initializes `current` and `previous` to the same starting value.
- During the global Newton step (when calling `(try_)stiffness_matrix`), trial values are written into `current` at every quadrature point.
As a consequence, for models providing the `material_response` hook, the generic element routine stores them automatically via [`set_trial!`](@ref).
Custom `_assemble_element!` methods must update `current` manually.
- [`material_stress`](@ref) is used for [stress postprocessing](#Stress-postprocessing).
It reads the state and returns stress, but it must not commit the state or advance history.
- `update_state!` is called (by `update_states!(assembler)` from the user-code) for every quadrature point once per load step, after the global Newton iteration has converged.
It commits `current` to `previous`.
- Similarly, `revert_state!` is called by `revert_states!(assembler)` when a load step is rejected; it restores `current` from `previous`.
The user must explicitly call `revert_states!(assembler)` in their outer solver loop to perform this rollback when a load step is rejected (e.g., Newton does not converge).

## Example: Implementing J2Plasticity

A small non-trivial model is the small-strain J2 plasticity with linear isotropic hardening.
It is ported from the Ferrite.jl plasticity tutorial [1]; reading that tutorial side-by-side with this section is the recommended way to see the correspondence between pure Ferrite code and the `FerriteSolidMechanics.jl` hooks.

### Material struct and state

```julia
struct J2Plasticity{T, S <: SymmetricTensor{4, 3, T}} <: AbstractMaterial
    G::T   # shear modulus
    K::T   # bulk modulus
    σ₀::T  # initial yield limit
    H::T   # hardening modulus
    Dᵉ::S  # elastic stiffness tensor
end

struct J2PlasticityStateInternal{T}
    ϵᵖ::SymmetricTensor{2, 3, T, 6}  # plastic strain
    σ::SymmetricTensor{2, 3, T, 6}   # stress
    k::T                             # hardening variable
end

mutable struct J2PlasticityState <: AbstractMaterialState
    current::J2PlasticityStateInternal{Float64}
    previous::J2PlasticityStateInternal{Float64}
end
```

Note how `J2PlasticityState` explicitly implements the trial and commit pattern (via `current` and `previous`):
`J2PlasticityStateInternal` is immutable, while only the wrapper is mutable, and only so that `current` can be reassigned to a new internal struct each Newton iteration.
The type parameter `T` keeps the model wrappable via `PlaneStrain`/`PlaneStress`: they differentiate the 3D response with ForwardDiff, so the constitutive update below builds the internal state from strain and stress that arrive as `ForwardDiff.Dual` numbers while the committed history stays `Float64`.

### State hooks

We flag the material as nonlinear and define the required hooks to initialize the state object and manage its trial/commit updates:

```julia
is_linear(::J2Plasticity) = false

function create_state(::J2Plasticity)
    return J2PlasticityState(
        J2PlasticityStateInternal(zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), 0.0),
        J2PlasticityStateInternal(zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), 0.0),
    )
end

function update_state!(state::J2PlasticityState)
    state.previous = state.current
    return nothing
end

function revert_state!(state::J2PlasticityState)
    state.current = state.previous
    return nothing
end
```


### Local constitutive update

The constitutive update is handled by a single function as follows:

```julia
function compute_stress_tangent(ϵ::SymmetricTensor{2, 3}, material::J2Plasticity, state::J2PlasticityStateInternal)
    # unpack some material parameters
    G = material.G
    H = material.H

    # Trial-values
    σᵗ = material.Dᵉ ⊡ (ϵ - state.ϵᵖ)  # trial-stress
    sᵗ = dev(σᵗ)                        # deviatoric part of trial-stress
    J₂ = 0.5 * sᵗ ⊡ sᵗ                 # second invariant of sᵗ
    σᵗₑ = sqrt(3.0 * J₂)                # effective trial-stress (von Mises stress)
    σʸ = material.σ₀ + H * state.k      # Previous yield limit

    φᵗ = σᵗₑ - σʸ                       # Trial-value of the yield surface

    if φᵗ < 0.0                         # elastic loading
        T = eltype(σᵗ)
        ϵᵖ = convert(SymmetricTensor{2, 3, T, 6}, state.ϵᵖ)
        return σᵗ, material.Dᵉ, J2PlasticityStateInternal(ϵᵖ, σᵗ, convert(T, state.k))
    else                                # plastic loading
        h = H + 3G
        μ = φᵗ / h                      # plastic multiplier

        c1 = 1 - 3G * μ / σᵗₑ
        s = c1 * sᵗ                     # updated deviatoric stress
        σ = s + vol(σᵗ)                 # updated stress

        κ = H * (state.k + μ)           # drag stress
        σₑ = material.σ₀ + κ            # updated yield surface

        # ... algorithmic tangent D ...

        Δϵᵖ = 3 / 2 * μ / σₑ * s        # plastic strain increment
        ϵᵖ = state.ϵᵖ + Δϵᵖ             # plastic strain
        k = state.k + μ                 # hardening variable
        return σ, D, J2PlasticityStateInternal(ϵᵖ, σ, k)
    end
end
```

### Constitutive interface

```julia
kinematics(::J2Plasticity) = SmallStrain()

function material_response(mp::J2Plasticity, ε::SymmetricTensor{2,3}, state::J2PlasticityState, dt, cache=nothing)
    return compute_stress_tangent(ε, mp, state.previous)
end
```

`material_response` reads the committed history from `state.previous` and delegates to the constitutive update function.
The generic element routine receives the returned `(σ, D, new_state)` tuple and automatically stores `new_state` as the uncommitted trial state for the current Newton iteration (via `state.current = new_state`).

### Stress postprocessing

J2's tangent is cheap, so it defines **no** `material_stress`: the generic fallback (which evaluates `material_response` and discards the tangent) is therefore used.
Overriding pays off only when the stress has a cheaper path than the tangent, as in AD history models that return it from the local solve without differentiating.
If implemented for J2, the method would read:

```julia
function material_stress(mp::J2Plasticity, ε::SymmetricTensor{2,3}, state::J2PlasticityState, dt, cache=nothing)
    σ, _, _ = compute_stress_tangent(ε, mp, state.current)
    return σ
end
```

`material_stress` receives `ε` directly, reads `state.current` (the solver's latest values), and returns only `σ`; the tangent and updated state that `compute_stress_tangent` also returns are discarded, since postprocessing must not advance history.
The generic `_compute_stress_qp` fallback extracts `ε` from the element values and calls `material_stress`.

### [Wrapper hooks (for `PlaneStrain` / `PlaneStress`)](@id wrapper-hooks)

J2 plasticity is small-strain, so it needs **no wrapper code at all**: every `SmallStrain` material is wrappable automatically.
A generic fallback implements both hooks (`compute_PK1_3D` and `update_state_from_3D!`) by calling `material_response`: given the wrapper's embedded 3D `F̄`, it forms `ε = sym(F̄ - I)`, evaluates the response, and maps the Cauchy stress to PK1.
The package then runs:

```julia
# provided generically for any SmallStrain material – you do not write this
function compute_PK1_3D(mp, F̄, dt, state)
    ε = symmetric(F̄ - one(F̄))
    σ, _, _ = material_response(mp, ε, state, dt)
    return det(F̄) * σ ⋅ inv(F̄)'
end
```

Only **finite-strain** models must implement [`compute_PK1_3D`](@ref) and [`update_state_from_3D!`](@ref) themselves as there is no universal `F̄ → stress` conversion for them (see `VEVP_Zhao2021_AD` for a worked example).

Both wrappers differentiate `compute_PK1_3D` with ForwardDiff, so `F̄` carries `ForwardDiff.Dual` entries while the committed history remains `Float64`.
A model that constructs a state object or scratch cache mixing both types fails with a `MethodError`.
`J2Plasticity` stays wrappable by using generic number types for its internal state instead of hardcoding `Float64`.

## Tangent strategies

Either hand-derive the tangent to hard-code an **analytical tangent** or use **automatic differentiation** via `Tensors.gradient`:

```julia
# Finite-strain: ∂P/∂F
dP_dF = Tensors.gradient(F_ -> compute_PK1(mat, F_, dt, state), F)

# Small-strain: ∂σ/∂ε
dσ_dε = Tensors.gradient(ε_ -> compute_stress(mat, ε_, dt, state), ε)
```

### Declaring tangent symmetry (small-strain)

Small-strain materials may declare [`tangent_symmetry`](@ref) (optionally).
The default [`Unsymmetric`](@ref) integrates every entry of `ke`; [`MajorSymmetric`](@ref) integrates only the lower triangle and mirrors it, which measures about 1.4× faster on element assembly:

```julia
tangent_symmetry(::MyMaterial) = MajorSymmetric()   # only if D[i,j,k,l] == D[k,l,i,j]
```

Before declaring it, verify that the tangent `D` returned by `material_response` is major-symmetric at several strain states and on every branch (elastic and inelastic) of your model:

```julia
D ≈ permutedims(D, (3, 4, 1, 2))   # true ⟹ major-symmetric, safe to opt in
```

If the check passes everywhere, declare `MajorSymmetric()`.
If it fails on any branch, keep the default (`Unsymmetric()`).
Note that `SymmetricTensor{4}` carries only the *minor* symmetries (`D[i,j,k,l] == D[j,i,k,l] == D[i,j,l,k]`), which do not imply major symmetry, so the tensor type alone is not sufficient.

Isotropic elasticity and associated plasticity with isotropic hardening are major-symmetric, so the bundled `Hooke`, `Hooke2D`, and `J2Plasticity` opt in.
Non-associated flow, nonlinear kinematic hardening, and anisotropic damage are not, and must keep the default.
Declaring `MajorSymmetric()` wrongly discards the non-symmetric part of `D`: the residual stays correct while the tangent does not, so Newton converges either slowly or not at all.
A solution that does converge is still correct, but the `K` returned by `stiffness_matrix` is not (matters only if you use it beyond the Newton update).
Finite-strain materials are unaffected; `∂P/∂F` is never assumed symmetric.

## Testing material model implementations

It is recommended to implement three categories of tests:

1. **Material-point verification.**
   Evaluate `material_response` for prescribed strain values to verify the computed stresses and consistent tangent moduli.
   Finite-difference approximations can be used to validate the analytical or automatically differentiated tangent operator.
   This test does not require a finite element assembly, making it suitable for comparison against other implementations.
   Several models have dedicated test files for this purpose, e.g. `test/test_ogden.jl` or `test/test_vepd_detrez2010_tangent.jl`.

   ```julia
   # Example
   mat = J2Plasticity(200.0e3, 0.3, 200.0, 10.0e3)
   state = create_state(mat)
   ε = SymmetricTensor{2,3}((i, j) -> i == j == 1 ? 2.0e-3 : 0.0)
   σ, D, new = material_response(mat, ε, state, 0.0)
   set_trial!(state, new)
   update_state!(state)
   ```

2. **Equivalence to reference implementations.**
   Compare the element stiffness matrix and residual vector against a standard Ferrite cell loop under simple loading conditions.
   The script `test/test_generic_assembler_equivalence.jl` demonstrates this procedure.

3. **State management validation.**
   The internal state variable handling must be verified by advancing the state and subsequently invoking `revert_state!`.
   The final state must be identical to the initial state.
   The available tests in `test/test_state_management.jl` can serve as a template.

Instead of creating a dedicated file for each model, tests should generally be appended to the existing categorical test files in `test/` and run via `test/runtests.jl`.

### Test coverage requirements

Material models integrated into the core package (`src/models/`) require specific test coverage.
The following test integrations are prioritized, with the first two being mandatory.

1. **`test/test_material_execution_paths.jl`**
   A single-element assembly test must be conducted to ensure that `stiffness_matrix` and `compute_stresses` return finite, non-zero results.

2. **`test/test_state_management.jl`**
   Stateful models require a test scenario that induces a non-trivial state change, such as yielding.
   The test must subsequently verify the proper functioning of the `update_state!` and `revert_state!` procedures.

3. **`test/test_stress_extraction.jl`**
   The `compute_stresses` function must be deterministic and independent of the time increment for repeated evaluations on a committed state.

4. **`test/test_dimensionality_wrappers.jl`**
   Models providing `compute_PK1_3D` and `update_state_from_3D!` methods must be tested within the `PlaneStrain` and `PlaneStress` wrappers.
   The chosen parameters must trigger every code branch (e.g. both the elastic and the plastic path), because a `Float64`/`Dual` type mismatch in an unexecuted branch stays undetected.

The existing tests in the `test/` directory provide suitable templates.

### Documentation requirements

Each bundled model requires a documentation page in `docs/src/models/` following the structure and style of the existing pages (e.g. `ogden.md` or `vevp_moammm.md`).

## Experimental models

Experimental models located in `src/models/experimental/` are exempt from the strict naming conventions, as well as the testing and documentation requirements.
New contributions are welcome here directly: get a working model in first and only later adopt the bundled-model requirements if you decide to promote it.

## See also

- `src/Interfaces.jl`: Documentation of the required interfaces.
- `src/models/ArrudaBoyce.jl` and `src/models/MooneyRivlin.jl`: Simple stateless hyperelastic models that each support both `:AT` and `:AD` tangent options.
- `src/models/Ogden.jl`: A specialized analytical tangent involving spectral decomposition.
- `src/models/VEVP_Zhao2021_AT.jl` and `src/models/VEPD_Detrez2010.jl`: Stateful viscoelastic-(visco)plastic models with analytical and automatic differentiation tangents, respectively.

## References

[1] Ferrite.jl documentation. Plasticity tutorial. [url](https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/plasticity/)