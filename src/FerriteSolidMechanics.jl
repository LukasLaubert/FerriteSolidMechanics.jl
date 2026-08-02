module FerriteSolidMechanics

using Ferrite
using Tensors
using LinearAlgebra
using MPI
using SparseArrays
using OhMyThreads
using OhMyThreads: @local, @set

# Single export list. Do not add inline `export` statements in the included files
# Keep every exported name here for reviewability
export AbstractMaterial, AbstractMaterialState, NoState,
       AbstractHyperelastic,
       LocalAssemblyFailure, RemoteAssemblyFailure,
       is_linear, create_state, update_state!, revert_state!, copy_state!,
       material_response, material_stress, kinematics, set_trial!, allocate_material_cache,
       AbstractKinematics, SmallStrain, FiniteStrain,
       AbstractTangentSymmetry, MajorSymmetric, Unsymmetric, tangent_symmetry,
       _assemble_element!, _compute_stress_qp,
       deformation_gradient,
       alpha_value, create_alpha_values,
       compute_PK1_3D, update_state_from_3D!,
       TimeStepController, TimeStepControllerExhausted, accept_step!, reject_step!, reset_controller!,
       GenericMaterialAssembler, create_assembler, stiffness_matrix, try_stiffness_matrix, compute_forces, compute_stresses, update_states!, revert_states!,
       AbstractLoad, LoadHandler, Traction, Pressure, NodalForce, BodyForce, external_forces!,
       recommended_blas_threads, recommended_solve_settings, estimated_replicated_memory_gb,
       distributed_solve, DistributedSolveError,
       Hooke, Hooke2D, NeoHooke, ArrudaBoyce, MooneyRivlin, Ogden, J2Plasticity,
       VEPD_Detrez2010, VEPD_Detrez2010ConvergenceError, VEVP_Zhao2021_AD, VEVP_Zhao2021_AT, VEVP_MOAMMM, VEVP_MOAMMMConvergenceError,
       PlaneStrain, PlaneStress, PlaneStressConvergenceError,
       FromMaterialModelsBase, MMBState, MaterialModelsBaseConvergenceError

include("Interfaces.jl")
include("DistributedSolve.jl")
include("SolveSettings.jl")
include("TimeStepController.jl")
include("ModelHelpers.jl")
include("MaterialModelsBaseSupport.jl")
include("GenericMaterialAssembler.jl")
include("Loads.jl")
include("PlaneStrainStress.jl")

include("models/LinearElasticity.jl")
include("models/NeoHooke.jl")
include("models/ArrudaBoyce.jl")
include("models/MooneyRivlin.jl")
include("models/Ogden.jl")
include("models/J2Plasticity.jl")
include("models/VEVP_Zhao2021_AD.jl")
include("models/VEVP_MOAMMM.jl")
include("models/VEPD_Detrez2010.jl")
include("models/VEVP_Zhao2021_AT.jl")

module Experimental
    using Ferrite
    using Tensors
    using LinearAlgebra
    # `using` for names a model calls; `import` for hooks it extends:
    # without `import`, `function material_response(…)` silently creates a local function instead of extending the parent's
    using ..FerriteSolidMechanics: AbstractMaterial, AbstractMaterialState, NoState, LocalAssemblyFailure, alpha_value, lame_parameters, _vepd_crystalline_stress, _vepd_network_stress, assemble_pk1_tangent!,
                              AbstractHyperelastic, SmallStrain, FiniteStrain, MajorSymmetric, Unsymmetric, set_trial!
    import ..FerriteSolidMechanics: is_linear, create_state, update_state!, revert_state!, copy_state!, _assemble_element!, _compute_stress_qp, compute_PK1_3D, update_state_from_3D!, deformation_gradient,
                               material_response, kinematics, material_stress, allocate_material_cache, tangent_symmetry, Ψ

    include("models/experimental/VEVP_Zhao2021_AD_Simplified.jl")
    include("models/experimental/VEPD_Detrez2010_Optimized.jl")
    include("models/experimental/VEPD_Detrez2010_Implicit.jl")
    include("models/experimental/VEPD_Detrez2010_ExactVisco.jl")
    include("models/experimental/VEPD_Detrez2010_ClosedCVEndStep.jl")
    include("models/experimental/VEVP_Zhao2021_AT_Matlab.jl")
    include("models/experimental/VEVP_MOAMMM_VarsN.jl")
end

end