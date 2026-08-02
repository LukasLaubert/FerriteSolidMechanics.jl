# Package extension bridging MaterialModelsBase.jl materials into the
# FerriteSolidMechanics constitutive interface. Activated by loading
# MaterialModelsBase (`import MaterialModelsBase`).
#
# The two interfaces were designed to line up (FerriteSolidMechanics follows the
# conventions established by K. A. Meyer's MaterialModelsBase.jl), so the
# bridge is thin:
#   - FerriteSolidMechanics passes the whole trial/commit state object; here we
#     store the wrapped material's immutable snapshots in an MMBState pair
#     and hand `state.previous` to MaterialModelsBase.
#   - The returned snapshot is FerriteSolidMechanics' trial form (stored by the
#     generic element routine via `set_trial!`, i.e. `state.current = new`).
#   - MaterialModelsBase.MaterialConvergenceError is rethrown as a
#     recoverable LocalAssemblyFailure.
module FerriteSolidMechanicsMaterialModelsBaseExt

using FerriteSolidMechanics: FerriteSolidMechanics, FromMaterialModelsBase, MMBState,
    MaterialModelsBaseConvergenceError
import MaterialModelsBase as MMB

function FerriteSolidMechanics.create_state(w::FromMaterialModelsBase)
    s0 = MMB.initial_material_state(w.material)
    return MMBState(s0, s0)
end

FerriteSolidMechanics.allocate_material_cache(w::FromMaterialModelsBase) =
    MMB.allocate_material_cache(w.material)

function FerriteSolidMechanics.material_response(w::FromMaterialModelsBase, strain, state::MMBState, dt, cache=nothing)
    mmb_cache = cache === nothing ? MMB.allocate_material_cache(w.material) : cache
    try
        stress, tangent, new_state = MMB.material_response(w.material, strain, state.previous, dt, mmb_cache)
        return stress, tangent, new_state
    catch err
        err isa MMB.MaterialConvergenceError && throw(MaterialModelsBaseConvergenceError(err))
        rethrow()
    end
end

end