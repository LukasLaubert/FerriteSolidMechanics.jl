# Package-side surface of the MaterialModelsBase.jl adapter. The actual
# interface methods live in ext/FerriteSolidMechanicsMaterialModelsBaseExt.jl and
# are activated by loading MaterialModelsBase (`import MaterialModelsBase`).
#
# MaterialModelsBase.jl is a package by Knut Andreas Meyer; this package's
# constitutive interface deliberately follows its conventions so that the
# adapter below is a thin, loss-free bridge.

"""
    FromMaterialModelsBase(material; kinematics=SmallStrain())

Wrap a material implementing the
[MaterialModelsBase.jl](https://github.com/KnutAM/MaterialModelsBase.jl)
interface, for example a model from
[MechanicalMaterialModels.jl](https://github.com/KnutAM/MechanicalMaterialModels.jl),
for use everywhere a FerriteSolidMechanics `AbstractMaterial` is accepted: `create_assembler`, `stiffness_matrix`, `compute_stresses`, `update_states!`, `try_stiffness_matrix`, and so on.

The `kinematics` keyword declares which strain measure the wrapped material consumes, since MaterialModelsBase selects it by dispatch on the input type:

- [`SmallStrain`](@ref), the default, passes `ε::SymmetricTensor{2,dim}` and expects `(σ, ∂σ∂ε, new_state)`
- [`FiniteStrain`](@ref) passes `F::Tensor{2,dim}` and expects `(P, ∂P∂F, new_state)`

The constructor is always available, while a package extension provides the interface methods behind it.
Load the weak dependency first:

```julia
import MaterialModelsBase                    # activates the extension
using MechanicalMaterialModels: Plastic, LinearElastic, Voce,
                                ArmstrongFrederick, NortonOverstress

mat = FromMaterialModelsBase(Plastic(;
    elastic   = LinearElastic(E=210.0e3, ν=0.3),
    yield     = 100.0,
    isotropic = Voce(Hiso=50.0e3, κ∞=100.0),
    kinematic = ArmstrongFrederick(Hkin=200.0e3, β∞=200.0),
    overstress = NortonOverstress(tstar=1.0, nexp=2.0),
))
fem = create_assembler(mat, dh, ch)
```

Prefer `import MaterialModelsBase` over `using MaterialModelsBase`: both packages export a `material_response` function with the same semantics by design, so `using` both makes the unqualified name ambiguous.

Neither MaterialModelsBase nor MechanicalMaterialModels is registered in the General registry at the time of writing, so install them by URL: `Pkg.add(url="https://github.com/KnutAM/MaterialModelsBase.jl")`.
MechanicalMaterialModels additionally depends on `Newton.jl`, which must be added first via `Pkg.add(url="https://github.com/KnutAM/Newton.jl")`.

`FromMaterialModelsBase` stores the wrapped material's immutable state snapshots in an [`MMBState`](@ref) trial/commit pair, so `update_states!` and `revert_states!` work as for any bundled material.
A local convergence failure, `MaterialModelsBase.MaterialConvergenceError`, is rethrown as [`MaterialModelsBaseConvergenceError`](@ref), a [`LocalAssemblyFailure`](@ref), so `try_stiffness_matrix` and `TimeStepController`-driven adaptive stepping work unchanged.
"""
struct FromMaterialModelsBase{K<:AbstractKinematics,M} <: AbstractMaterial
    material::M
end

function FromMaterialModelsBase(material; kinematics::AbstractKinematics=SmallStrain())
    return FromMaterialModelsBase{typeof(kinematics),typeof(material)}(material)
end

kinematics(::FromMaterialModelsBase{K}) where {K} = K()

"""
    MMBState(current, previous)

Trial/commit state pair for [`FromMaterialModelsBase`](@ref).
Both fields hold immutable MaterialModelsBase state snapshots, and the default [`set_trial!`](@ref) semantics apply.
`update_state!` commits `previous ← current`, and `revert_state!` rolls back `current ← previous`.
"""
mutable struct MMBState{S} <: AbstractMaterialState
    current::S
    previous::S
end

update_state!(state::MMBState) = (state.previous = state.current; nothing)
revert_state!(state::MMBState) = (state.current = state.previous; nothing)

"""
    MaterialModelsBaseConvergenceError(error)

Recoverable [`LocalAssemblyFailure`](@ref) wrapping a
`MaterialModelsBase.MaterialConvergenceError` thrown by a wrapped
material's local solve, so that [`try_stiffness_matrix`](@ref) reports
`converged = false` instead of aborting the outer solve.
"""
struct MaterialModelsBaseConvergenceError <: LocalAssemblyFailure
    error::Exception
end

function Base.showerror(io::IO, err::MaterialModelsBaseConvergenceError)
    print(io, "wrapped MaterialModelsBase material failed to converge locally: ")
    showerror(io, err.error)
end

# Helpful failure when the extension is not loaded (create_state then falls
# back to NoState(), which routes the generic element kernel here).
function material_response(w::FromMaterialModelsBase, strain, state, dt, cache=nothing)
    throw(ArgumentError(
        "FromMaterialModelsBase requires the MaterialModelsBase package extension. " *
        "Run `import MaterialModelsBase` (install via " *
        "Pkg.add(url=\"https://github.com/KnutAM/MaterialModelsBase.jl\")) before assembling."))
end