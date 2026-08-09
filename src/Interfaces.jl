"""
    AbstractMaterial

Supertype for all material models used with GenericMaterialAssembler.
"""
abstract type AbstractMaterial end

"""
    AbstractMaterialState

Supertype for all quadrature point state objects that store internal history.
"""
abstract type AbstractMaterialState end

"""
    LocalAssemblyFailure

Abstract exception type for recoverable local assembly failures.
Public non-throwing helpers such as [`try_stiffness_matrix`](@ref) catch this type and return it to the caller, so an outer load step controller can reduce the step size and retry.

Those helpers catch only exceptions that opt in by subtyping `LocalAssemblyFailure`.
Exceptions from invalid inputs or buggy material methods are rethrown.
"""
abstract type LocalAssemblyFailure <: Exception end

"""
    RemoteAssemblyFailure()

Recoverable assembly failure marker that [`try_stiffness_matrix`](@ref) returns on MPI ranks which did not fail locally, when another rank reported a `LocalAssemblyFailure`.

The rank where the failure occurred reports the original material-specific exception as `result.error`.
All ranks receive `converged = false`, so an adaptive outer solver can reject the step collectively and retry with a smaller `dt`.
"""
struct RemoteAssemblyFailure <: LocalAssemblyFailure end

function Base.showerror(io::IO, ::RemoteAssemblyFailure)
    print(io, "recoverable assembly failure occurred on another MPI rank")
end

"""
    NoState

Placeholder state for materials that do not require internal history.
"""
struct NoState <: AbstractMaterialState end

"""
    is_linear(material)

Return `true` when `material` has a displacement-independent tangent stiffness.
`GenericMaterialAssembler` preassembles linear materials once.
"""
is_linear(::AbstractMaterial) = false

"""
    create_state(material)

Create the quadrature point state object of `material`.
A stateless material uses the default method, which returns `NoState()`.
"""
create_state(::AbstractMaterial) = NoState()

"""
    update_state!(state)

Commit one converged quadrature point state at the end of a load or time step.

`update_state!` either mutates a mutable state in place and returns `nothing`, or returns a new `AbstractMaterialState` for an immutable or replacement-style state.
[`update_states!`](@ref) stores a returned state back into the assembler state vector.
"""
update_state!(::AbstractMaterialState) = nothing

"""
    revert_state!(state)

Restore one quadrature point state to the previously committed value.

`revert_state!` either mutates a mutable state in place and returns `nothing`, or returns a new state value for a replacement-style state.
A caller that owns the state container is responsible for storing that returned value.
"""
revert_state!(::AbstractMaterialState) = nothing

"""
    copy_state!(dest, src)

Copy fields from `src` to `dest` recursively, using reflection.
`copy_state!` copies arrays in place where possible and returns immutable values, so a caller that owns the storage can replace the destination value.
This is the generic history-state fallback for arbitrary structs.
"""
@generated function copy_state!(dest::T, src::T) where {T}
    if !ismutabletype(T)
        return :(return src)
    end
    exprs = []
    for field in fieldnames(T)
        ft = fieldtype(T, field)
        if isbitstype(ft)
            push!(exprs, :(dest.$field = src.$field))
        else
            push!(exprs, quote
                copied = copy_state!(dest.$field, src.$field)
                if copied !== nothing && copied isa $ft
                    dest.$field = copied
                end
            end)
        end
    end
    return quote
        $(exprs...)
        return dest
    end
end

function copy_state!(dest::AbstractArray, src::AbstractArray)
    size(dest) == size(src) || throw(DimensionMismatch("copy_state! requires equal sizes, got dest $(size(dest)) and src $(size(src))"))
    for i in eachindex(src)
        if isbitstype(eltype(src))
            dest[i] = src[i]
        else
            copied = copy_state!(dest[i], src[i])
            if copied !== nothing && copied isa eltype(dest)
                dest[i] = copied
            end
        end
    end
    return dest
end

#######################################################
# quadrature point constitutive interface (UMAT-like) #
#######################################################

"""
    AbstractKinematics

Trait supertype describing which strain measure a material consumes.
[`kinematics`](@ref) returns it, and the generic element routine uses it to select what to evaluate at each quadrature point and which weak form to integrate.
Concrete traits: [`SmallStrain`](@ref), [`FiniteStrain`](@ref).
"""
abstract type AbstractKinematics end

"""
    SmallStrain <: AbstractKinematics

Material works on the small-strain tensor `ε = sym(∇u)`, a `SymmetricTensor{2,dim}`, and returns the Cauchy stress `σ` and the algorithmic tangent `D = ∂σ/∂ε`.
"""
struct SmallStrain <: AbstractKinematics end

"""
    FiniteStrain <: AbstractKinematics

Material works on the deformation gradient `F = I + ∇u`, a `Tensor{2,dim}`, and returns the first Piola–Kirchhoff stress `P` and the tangent `∂P/∂F`.
"""
struct FiniteStrain <: AbstractKinematics end

"""
    kinematics(material) -> AbstractKinematics

Declare the strain measure of a material: [`SmallStrain`](@ref) or [`FiniteStrain`](@ref).

> **Extension point.** Every material that uses the generic element routine, meaning every material that implements [`material_response`](@ref), must define `kinematics`.
> A material that provides its own [`_assemble_element!`](@ref) method does not need it.
"""
kinematics(mp::AbstractMaterial) = throw(ArgumentError("$(typeof(mp)) defines neither kinematics()/material_response() nor a custom _assemble_element! method. Implement material_response + kinematics (recommended), or a custom element routine."))

"""
    AbstractTangentSymmetry

Trait supertype describing whether a [`SmallStrain`](@ref) material's tangent `D = ∂σ/∂ε` is major-symmetric.
[`tangent_symmetry`](@ref) returns it, and the generic element routine uses it to select the tangent-integration loop.
Concrete traits: [`MajorSymmetric`](@ref), [`Unsymmetric`](@ref).
"""
abstract type AbstractTangentSymmetry end

"""
    MajorSymmetric <: AbstractTangentSymmetry

`D[i,j,k,l] == D[k,l,i,j]`, so the element stiffness is symmetric.
The element routine integrates only its lower triangle and mirrors the upper triangle.

Declaring this trait for a tangent that is *not* major-symmetric silently discards the antisymmetric part and yields a wrong element stiffness, so Newton converges slowly or not at all.
The residual stays correct, which makes the error easy to miss.
"""
struct MajorSymmetric <: AbstractTangentSymmetry end

"""
    Unsymmetric <: AbstractTangentSymmetry

No symmetry is assumed, so the element routine integrates every entry of `ke`.
This is the default, and the correct choice for non-associated flow, nonlinear kinematic hardening, and anisotropic damage.
"""
struct Unsymmetric <: AbstractTangentSymmetry end

"""
    tangent_symmetry(material) -> AbstractTangentSymmetry

Declare whether a [`SmallStrain`](@ref) material's tangent `D = ∂σ/∂ε` is major-symmetric.
Defaults to [`Unsymmetric`](@ref).

Returning [`MajorSymmetric`](@ref) halves the tangent-integration work, worth roughly 1.4× on small-strain element assembly.
The declaration is a promise about the values, not about the type.
`SymmetricTensor{4}` carries only the minor symmetries `D[i,j,k,l] == D[j,i,k,l] == D[i,j,l,k]`, which do not imply major symmetry.
Check before claiming it:

```julia
D ≈ permutedims(D, (3, 4, 1, 2))
```

[`FiniteStrain`](@ref) materials do not use this trait, because their `∂P/∂F` is never assumed symmetric.
"""
tangent_symmetry(::AbstractMaterial) = Unsymmetric()

"""
    material_response(material, ε_or_F, state, dt, cache=nothing) -> (stress, tangent, new_state)

Constitutive update at a single quadrature point.
A typical material implements `material_response`, [`kinematics`](@ref), and, for a history-dependent model, a state type.

# Arguments
- `ε_or_F`: strain measure matching `kinematics(material)` – `ε::SymmetricTensor{2,dim}` for [`SmallStrain`](@ref), `F::Tensor{2,dim}` for [`FiniteStrain`](@ref)
- `state`: the quadrature point state object, `NoState()` for a stateless material. Read the committed history from it, by convention the `previous`-side fields. Do not commit.
- `dt`: time increment of the current step, where `0.0` freezes viscous flow
- `cache`: scratch object from [`allocate_material_cache`](@ref), or `nothing`

# Returns
- `stress`: `σ` for SmallStrain, `P` for FiniteStrain
- `tangent`: `∂σ/∂ε` for SmallStrain, `∂P/∂F` for FiniteStrain
- `new_state`: the trial state produced by this evaluation

The generic element routine accepts three forms of `new_state`:

1. the passed `state` itself, for a stateless material or a material that wrote its trial fields in place
2. a snapshot of internal variables that is not an `AbstractMaterialState`, stored via [`set_trial!`](@ref)`(state, new_state)`
3. a new `AbstractMaterialState`, which replaces the quadrature point state slot and must have the same concrete type as the state it replaces

The generic element routine pins one state type per cell, taken from the first quadrature point.

`material_response` must be AD-transparent where the material relies on automatic differentiation.
It must throw a [`LocalAssemblyFailure`](@ref) subtype for a recoverable local-solve failure.

The `material_response` name and the [`AbstractMaterial`](@ref)/[`AbstractMaterialState`](@ref) pair come from [MaterialModels.jl](https://github.com/kimauth/MaterialModels.jl) by K. Auth and contributors.
K. A. Meyer's [MaterialModelsBase.jl](https://github.com/KnutAM/MaterialModelsBase.jl) was developed from that package as an implementation-independent interface.
The design here follows the MaterialModelsBase conventions: a single response function returning `(stress, tangent, new_state)`, strain-measure-driven stress/tangent pairs, and the cache hook.
Materials implementing the MaterialModelsBase interface can be used directly via [`FromMaterialModelsBase`](@ref).
"""
function material_response end

"""
    material_stress(material, ε_or_F, state, dt, cache=nothing) -> stress

Return the postprocessing stress at one quadrature point, without the tangent.

`material_stress` is the optional stress-only companion to [`material_response`](@ref).
The generic [`_compute_stress_qp`](@ref) fallback calls it to obtain the output stress: Cauchy `σ` for [`FiniteStrain`](@ref), small-strain `σ` for [`SmallStrain`](@ref).
The default method computes that stress from `material_response` at the passed state and discards the tangent and the trial state, so a plain constitutive material needs no method of its own.

Define a method when computing the tangent is expensive, or when the output stress follows a model-specific convention.
An AD-based history model, for example, can return the stress directly from its local solve and skip the automatic differentiation.
Override `material_stress` in preference to the lower-level [`_compute_stress_qp`](@ref): `material_stress` receives the strain measure `ε` or `F` directly, whereas `_compute_stress_qp` receives the `cellvalues` and the quadrature point index and leaves the strain extraction to the method.

A `material_stress` method must be deterministic and must not commit or evolve state.
A rate-dependent model, for example, passes `dt = 0.0` to its local solve by default, so it reports the stress at the current time and state without evolving state.
"""
function material_stress end

"""
    set_trial!(state, new_state)

Store the trial result `new_state` returned by [`material_response`](@ref) into the quadrature point `state` without committing it.
The default method covers the `current`/`previous` layout:

```julia
set_trial!(state, new) = (state.current = new; nothing)
```

Override it for a state with a different layout; `VEPD_Detrez2010` is a field-wise example.
The generic element routine does not call `set_trial!` when `material_response` returns the state itself or a replacement `AbstractMaterialState`.
"""
@inline set_trial!(state::AbstractMaterialState, new) = (state.current = new; nothing)

"""
    allocate_material_cache(material) -> cache

Allocate a reusable scratch object that is passed as the `cache` argument of [`material_response`](@ref).
The default method returns `nothing`.
Element assembly calls `allocate_material_cache` once per element, and the generic stress fallback calls it once per quadrature point.

Override it to reuse buffers across the quadrature points of an element, such as workspace arrays for a local Newton solve.
A material whose cache is expensive to build should also override [`material_stress`](@ref).
"""
allocate_material_cache(::AbstractMaterial) = nothing

"""
    AbstractHyperelastic <: AbstractMaterial

Supertype for stateless hyperelastic materials defined by a strain energy density `Ψ(C)`.
A subtype implements the energy function and receives the complete material interface:

```julia
struct MyModel <: AbstractHyperelastic
    μ::Float64
end
FerriteSolidMechanics.Ψ(C, mp::MyModel) = 0.5 * mp.μ * (tr(C) - 3)  # + volumetric part
```

Derived defaults, each overridable with an analytic expression:

- `constitutive_driver(C, mp)` – `S = 2∂Ψ/∂C` and `∂S/∂C = 2∂²Ψ/∂C²` via `Tensors.hessian`
- `_second_piola(C, mp)` – `S = 2∂Ψ/∂C` via `Tensors.gradient`
- [`material_response`](@ref), [`kinematics`](@ref) (= `FiniteStrain()`), [`_compute_stress_qp`](@ref), [`compute_PK1_3D`](@ref) and [`update_state_from_3D!`](@ref), so `PlaneStrain` and `PlaneStress` wrapping works without additional methods
"""
abstract type AbstractHyperelastic <: AbstractMaterial end

"""
    Ψ(C, material)

Strain energy density as a function of the right Cauchy–Green tensor `C = FᵀF`.
This is the extension point for [`AbstractHyperelastic`](@ref) materials.
Extend it as `FerriteSolidMechanics.Ψ`.
"""
function Ψ end

"""
    _assemble_element!(ke, re, states, material, cellvalues, alphavalues, u, dt)

Element-level extension hook for assembling an element stiffness matrix `ke` and internal force vector `re`.

Most materials do not implement `_assemble_element!`.
A generic fallback loops over the quadrature points, evaluates [`material_response`](@ref), and integrates the weak form selected by [`kinematics`](@ref).
Implement `_assemble_element!` directly only when the element structure itself is material-specific, such as a hand-derived tangent assembled from nodal shape-gradient blocks (`VEVP_Zhao2021_AT`) or nonlocal regularization.

This hook does not reach mixed or multi-field formulations.
The assembler builds one `CellValues` per `SubDofHandler` from its single field interpolation, so it rejects a `SubDofHandler` carrying more than one field when the assembler is created.

> **Extension point.** The leading underscore indicates that users normally call higher-level assembler functions, but this method is exported so material packages can extend it.
> The [Developer guide](@ref) walks through a complete worked example.

A custom method is responsible for:

- looping over the quadrature points
- evaluating the strain or the deformation gradient
- computing the stress and the tangent
- multiplying by the integration weight, including the `alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)` factor, which the assembler does not apply
- adding the contributions to `ke` and `re`
- writing trial values into the material-specific current/trial state fields, without committing them

The assembler calls `update_state!` only after the outer Newton step has converged.
A custom method mutates `ke`, `re`, and the state in place and returns `nothing`; its return value is ignored.
"""
function _assemble_element! end

"""
    _compute_stress_qp(material, cellvalues, alphavalues, qp, u_local, state, dt)

Material extension hook for postprocessing the stress at one quadrature point.

> **Extension point.** The leading underscore indicates that users normally call higher-level assembler functions, but this method is exported so material packages can extend it.

A generic fallback derives the stress from [`material_stress`](@ref), which itself defaults to [`material_response`](@ref) with the tangent and the trial state discarded.
A material implementing the constitutive interface therefore needs no `_compute_stress_qp` method.
For a tangent-free stress path, override the higher-level [`material_stress`](@ref) instead, which receives the strain measure directly.
Define a `_compute_stress_qp` method only when the output convention needs the `cellvalues` and quadrature point context itself, as in element-structured models.

A `_compute_stress_qp` method returns the Cauchy stress at the quadrature point for a finite-strain material, or the Cauchy-like stress for a small-strain material.
`compute_stresses` multiplies that value by `alpha_value(alphavalues, qp)` before storing it in the output array, so the material must not apply the alpha factor itself.
A linear cell that owns no state is redispatched from `state === nothing` to `NoState()`.
A material with neither a `_compute_stress_qp` method nor the `material_response`/`kinematics` pair throws an `ArgumentError` instead of returning a silent zero stress.
"""
function _compute_stress_qp end

# Shared interface for 2D wrappers (PlaneStrain/PlaneStress)
"""
    compute_PK1_3D(material, F, dt, state)

Return the 3D first Piola-Kirchhoff stress for the dimensionality wrappers.

> **Extension point.** A cross-package hook.
> A [`FiniteStrain`](@ref) material must provide a method to be wrapped by [`PlaneStrain`](@ref) or [`PlaneStress`](@ref).
> A [`SmallStrain`](@ref) material requires no method, because a fallback maps the stress returned by [`material_response`](@ref) to `P`.
> An [`AbstractHyperelastic`](@ref) subtype requires none either, because a fallback derives `P` from the strain energy `Ψ`.
"""
function compute_PK1_3D end

"""
    update_state_from_3D!(state, material, F, dt)

Update a wrapped material state after a 3D constitutive evaluation.

> **Extension point.** A cross-package hook.
> A [`FiniteStrain`](@ref) material must provide a method to be wrapped by [`PlaneStrain`](@ref) or [`PlaneStress`](@ref).
> A [`SmallStrain`](@ref) material requires no method, because a fallback takes the trial state from [`material_response`](@ref).
> An [`AbstractHyperelastic`](@ref) subtype requires none either, because it is stateless and the fallback performs no update.
> The method writes the material's current/trial variables for the converged 3D deformation gradient `F`, and a replacement-style state may return a new state value.
> The history commit still happens through `update_state!` and [`update_states!`](@ref).
"""
function update_state_from_3D! end

# Shared extension point for optional alpha scaling.
"""
    alpha_value(alphavalues, qp::Int) -> Real

Return the scaling factor for quadrature point `qp`.

The element routine multiplies this factor into the integration weight before assembling into the residual and the stiffness, and `compute_stresses` applies the same factor to its output:

```julia
α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)
```

[`create_alpha_values`](@ref) creates `alphavalues`, and the assembler forwards it as the positional `av` argument to every material's element routine.

When `create_assembler` is called with `ah=nothing`, `alphavalues` is `nothing` and the built-in fallback returns `1.0`, so no scaling is applied.

The `alpha_value` name, the `AlphaValues` companion type that [`create_alpha_values`](@ref) resolves by name, and the `α_dΩ` integration weight come from the CapriccioSimulation FE–MD coupling code:
J. Roksvaag. *CAPRICCIO - Tool to run concurrent Finite Element-Molecular Dynamics Simulations* (Version 3.0.0), Zenodo (2026).
<https://doi.org/10.5281/zenodo.18326736>
"""
function alpha_value end
alpha_value(::Nothing, ::Int) = 1.0

"""
    create_alpha_values(ah, cellvalues)

Build the per-cell alpha-values object that the assembler passes to [`alpha_value`](@ref) at every quadrature point.

The returned object `av` must satisfy `alpha_value(av, qp::Int) -> Real`.
The assembler calls `create_alpha_values` once per cell during construction and stores the result for reuse in every assembly pass.

# How to extend

Provide one of the following; the first match wins:

1. A method `FerriteSolidMechanics.create_alpha_values(ah::MyAlphaSource, cellvalues)`
2. A constructor `AlphaValues(ah, cellvalues)` in the same module that defines `typeof(ah)`, which the default fallback calls automatically

`create_alpha_values` throws an `ArgumentError` if it finds neither.

When `create_assembler` is called with `ah=nothing`, `create_alpha_values` receives `nothing` and returns `nothing`.
That pairs with the `alpha_value(::Nothing, ::Int) = 1.0` fallback, so no scaling is applied.
"""
function create_alpha_values(ah, cellvalues)
    mod = parentmodule(typeof(ah))
    if isdefined(mod, :AlphaValues)
        return getfield(mod, :AlphaValues)(ah, cellvalues)
    end
    throw(ArgumentError("Define create_alpha_values(ah, cellvalues) or $(mod).AlphaValues(ah, cellvalues)"))
end
create_alpha_values(::Nothing, cellvalues) = nothing

"""
    deformation_gradient(cellvalues, qp, u_local)

Compute the deformation gradient `F = I + ∇u` at quadrature point `qp` from the given `cellvalues` and the element-local displacement vector `u_local`.

`cellvalues` is a Ferrite `CellValues`. `qp` is the quadrature point index.
`u_local` is the displacement vector restricted to the current cell, in the order returned by `celldofs(cell)`.

Returns a `Tensor{2,dim}` with the element type of `u_local`, where `dim` is the spatial dimension.
For `Float64` displacements this is `Tensor{2,3,Float64,9}` in 3D or `Tensor{2,2,Float64,4}` in 2D.
"""
function deformation_gradient(cellvalues, qp, u_local)
    grad_u = function_gradient(cellvalues, qp, u_local)
    return one(grad_u) + grad_u
end

"""
    get_cells(dh::DofHandler) -> AbstractSet{Int}

Return the cell ids that `dh` assigns degrees of freedom to, the union of every `SubDofHandler` cellset.
Cells outside this set carry no DOFs.
[`create_assembler`](@ref) ignores them and validates a material dictionary's cellsets against exactly this set.
"""
get_cells(dh::DofHandler) = length(dh.subdofhandlers) == 1 ? only(dh.subdofhandlers).cellset : union([sdh.cellset for sdh in dh.subdofhandlers]...)

get_ip(dh::DofHandler, sdh_idx::Int) = only(dh.subdofhandlers[sdh_idx].field_interpolations)

# Unwrap a vectorized interpolation to its scalar base
_base_interpolation(ip) = hasfield(typeof(ip), :ip) ? ip.ip : ip

function get_default_quadrature_order_for_sdh(dh::DofHandler, sdh_idx::Int)
    ip = get_ip(dh, sdh_idx)
    base_ip = _base_interpolation(ip)
    p = Ferrite.getorder(base_ip)
    return p > 1 ? p + 1 : 2
end

function cell_values_for_sdh(dh::DofHandler, sdh_idx::Int, order)
    ip = get_ip(dh, sdh_idx)
    refshape = getrefshape(ip)
    qr = QuadratureRule{refshape}(order)
    # Pick a sample cell from the SDH cellset to determine geometric interpolation
    sample_cid = first(dh.subdofhandlers[sdh_idx].cellset)
    geom_ip = Ferrite.geometric_interpolation(typeof(dh.grid.cells[sample_cid]))
    return CellValues(qr, ip, geom_ip)
end


mpi_is_active() = MPI.Initialized() && !MPI.Finalized()
mpi_rank() = mpi_is_active() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
mpi_size() = mpi_is_active() ? MPI.Comm_size(MPI.COMM_WORLD) : 1

function mpi_allreduce!(data)
    if mpi_size() > 1
        MPI.Allreduce!(data, +, MPI.COMM_WORLD)
    end
    return data
end

function mpi_any!(flag::Vector{Int}, value::Bool)
    @assert length(flag) == 1 "mpi_any! expects a one-entry flag buffer"
    flag[1] = value ? 1 : 0
    if mpi_size() > 1
        MPI.Allreduce!(flag, +, MPI.COMM_WORLD)
    end
    return flag[1] > 0
end