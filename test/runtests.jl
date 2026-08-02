using Test

@testset "FerriteSolidMechanics test suite" begin
    include("test_interface_contracts.jl")
    include("test_docstrings.jl")
    include("test_time_step_controller.jl")
    include("test_generic_assembler_equivalence.jl")
    include("test_dimensionality_wrappers.jl")
    include("test_material_execution_paths.jl")
    include("test_arruda_boyce.jl")
    include("test_mooney_rivlin.jl")
    include("test_ogden.jl")
    include("test_vepd_detrez2010_tangent.jl")
    include("test_vepd_detrez2010_histories.jl")
    include("test_vevp_zhao2021_tangent.jl")
    include("test_vevp_zhao2021_histories.jl")
    include("test_stress_extraction.jl")
    include("test_state_management.jl")
    include("test_loads.jl")
    # Spawn subprocesses: thread and rank counts are fixed at process start
    include("test_thread_determinism.jl")
    include("test_mpi.jl")
    include("test_example_integration.jl")
    # The MaterialModelsBase adapter needs the MaterialModelsBase.jl and MechanicalMaterialModels.jl packages.
    # Install by URL, see docs/src/models/index.md); testset is skipped when they are not available in the active environment.
    if Base.find_package("MaterialModelsBase") !== nothing &&
       Base.find_package("MechanicalMaterialModels") !== nothing
        include("test_materialmodelsbase_adapter.jl")
    else
        @info "Skipping MaterialModelsBase adapter tests (MaterialModelsBase / MechanicalMaterialModels not installed)"
    end
    # The MUMPS extension (distributed_solve) needs MUMPS.jl, a Unix/cluster binary.
    if Sys.isunix() && Base.find_package("MUMPS") !== nothing
        include("test_mumps_solve.jl")
    else
        @info "Skipping MUMPS distributed_solve tests (needs a Unix platform with MUMPS installed)"
    end
end