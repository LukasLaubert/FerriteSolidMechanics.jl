using Test
using Ferrite
using FerriteSolidMechanics

struct ImmutableStateMaterial <: AbstractMaterial end

struct ImmutableCommitState <: AbstractMaterialState
    current::Float64
    previous::Float64
end

FerriteSolidMechanics.create_state(::ImmutableStateMaterial) = ImmutableCommitState(0.0, 0.0)

function FerriteSolidMechanics.update_state!(state::ImmutableCommitState)
    return ImmutableCommitState(state.current, state.current)
end

function FerriteSolidMechanics._assemble_element!(ke, re, states, ::ImmutableStateMaterial, cv, av, u, dt)
    fill!(ke, 0.0)
    re !== nothing && fill!(re, 0.0)
    for qp in 1:getnquadpoints(cv)
        state = states[qp]::ImmutableCommitState
        states[qp] = ImmutableCommitState(state.previous + 1.0, state.previous)
    end
    return nothing
end

@testset "Interface contracts" begin
    @testset "copy_state! replaces immutable array elements" begin
        dest = [ImmutableCommitState(0.0, 0.0)]
        src = [ImmutableCommitState(2.0, 1.0)]

        copy_state!(dest, src)

        @test dest[1] == src[1]
    end

    @testset "update_states! accepts replacement states" begin
        grid = generate_grid(Quadrilateral, (1, 1))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
        close!(dh)

        ch = ConstraintHandler(dh)
        close!(ch)

        fem = create_assembler(ImmutableStateMaterial(), dh, ch)
        workspace_kes = [workspace.kes[1] for workspace in fem._workspaces]
        workspace_res = [workspace.res[1] for workspace in fem._workspaces]
        workspace_cvs = [workspace.cvs[1] for workspace in fem._workspaces]

        stiffness_matrix(fem, zeros(ndofs(dh)))
        stiffness_matrix(fem, zeros(ndofs(dh)))

        # Workspace buffers must be reused, not reallocated, across calls
        @test all(fem._workspaces[i].kes[1] === workspace_kes[i] for i in eachindex(workspace_kes))
        @test all(fem._workspaces[i].res[1] === workspace_res[i] for i in eachindex(workspace_res))
        @test all(fem._workspaces[i].cvs[1] === workspace_cvs[i] for i in eachindex(workspace_cvs))

        for state in fem.states[1]
            @test state.current == 1.0
            @test state.previous == 0.0
        end

        update_states!(fem)

        for state in fem.states[1]
            @test state.current == 1.0
            @test state.previous == 1.0
        end
    end

    @testset "recommended_blas_threads is dimension- and layout-keyed" begin
        grid2d = generate_grid(Quadrilateral, (1, 1))
        dh2d = DofHandler(grid2d)
        add!(dh2d, :u, Lagrange{RefQuadrilateral,1}()^2)
        close!(dh2d)

        grid3d = generate_grid(Hexahedron, (1, 1, 1))
        dh3d = DofHandler(grid3d)
        add!(dh3d, :u, Lagrange{RefHexahedron,1}()^3)
        close!(dh3d)

        # 2D frontal blocks are too small to thread -> 1; 3D dense fronts want ncores
        @test recommended_blas_threads(dh2d; ncores=8) == 1
        @test recommended_blas_threads(dh2d) == 1
        @test recommended_blas_threads(dh3d; ncores=8) == 8

        # Ranks sharing a node would oversubscribe it, so the 3D count collapses to 1
        @test recommended_blas_threads(dh3d; ncores=8, ranks_per_node=1) == 8
        @test recommended_blas_threads(dh3d; ncores=8, ranks_per_node=4) == 1
        @test recommended_blas_threads(dh2d; ncores=8, ranks_per_node=4) == 1
        # Serial (no MPI ranks sharing the node) keeps the pre-existing value
        @test recommended_blas_threads(dh3d; ncores=8) ==
              recommended_blas_threads(dh3d; ncores=8, ranks_per_node=1)
    end

    @testset "recommended_solve_settings reproduces the measured layouts" begin
        # Meshes big enough to sit on the correct side of the 3.7k-dof 3D crossover.
        dh3d = DofHandler(generate_grid(Hexahedron, (12, 12, 12)))
        add!(dh3d, :u, Lagrange{RefHexahedron,1}()^3)
        close!(dh3d)
        dh3d_small = DofHandler(generate_grid(Hexahedron, (4, 4, 4)))
        add!(dh3d_small, :u, Lagrange{RefHexahedron,1}()^3)
        close!(dh3d_small)
        dh2d = DofHandler(generate_grid(Quadrilateral, (200, 20)))
        add!(dh2d, :u, Lagrange{RefQuadrilateral,1}()^2)
        close!(dh2d)

        # 3D above the crossover: distributed_solve, many ranks, one BLAS thread each
        s = recommended_solve_settings(dh3d; cores_per_node=72, mumps_available=true)
        @test s.solver === :distributed_solve
        @test s.ranks_per_node == 18
        @test s.threads_per_rank == 4
        @test s.blas_threads == 1

        # 3D without MUMPS: one rank keeps a single factor and gives BLAS the node
        s = recommended_solve_settings(dh3d; cores_per_node=72)
        @test s.solver === :replicated_lu
        @test s.ranks_per_node == 1
        @test s.threads_per_rank == 72
        @test s.blas_threads == 72
        @test s.blas_threads == recommended_blas_threads(dh3d; ncores=72, ranks_per_node=1)

        # Below the crossover the replicated solve is chosen even with MUMPS present
        @test recommended_solve_settings(dh3d_small; cores_per_node=72,
                                         mumps_available=true).solver === :replicated_lu

        # 2D never threads BLAS, and assembly_bound shifts it toward more ranks
        s = recommended_solve_settings(dh2d; cores_per_node=72, mumps_available=true)
        @test s.solver === :replicated_lu
        @test s.blas_threads == 1
        @test s.ranks_per_node == 9
        @test recommended_solve_settings(dh2d; cores_per_node=72,
                                         assembly_bound=true).ranks_per_node == 18

        # Memory is reported per rank and only judged against a stated budget
        @test recommended_solve_settings(dh3d; cores_per_node=72).fits === missing
        @test recommended_solve_settings(dh3d; cores_per_node=72,
                                         memory_per_node_gb=256).fits === true
        @test recommended_solve_settings(dh3d; cores_per_node=72,
                                         memory_per_node_gb=1).fits === false

        # Single-core machines must still return a runnable layout
        for dh in (dh2d, dh3d), mumps in (false, true)
            s = recommended_solve_settings(dh; cores_per_node=1, mumps_available=mumps)
            @test s.ranks_per_node >= 1 && s.threads_per_rank >= 1 && s.blas_threads >= 1
        end
    end

    @testset "with MUMPS available the recommendation always fits" begin
        # The budget must be evaluated at the layout that is returned. Testing it at a
        # different rank count let the two disagree: `assembly_bound` doubles the 2D ranks,
        # so a grid could be judged to fit at 9 and be recommended at 18, returning
        # replicated_lu together with fits == false while MUMPS sat unused.
        dh2d = DofHandler(generate_grid(Quadrilateral, (60, 60)))
        add!(dh2d, :u, Lagrange{RefQuadrilateral,1}()^2)
        close!(dh2d)
        dh3d = DofHandler(generate_grid(Hexahedron, (12, 12, 12)))
        add!(dh3d, :u, Lagrange{RefHexahedron,1}()^3)
        close!(dh3d)

        for dh in (dh2d, dh3d), cores in (8, 36, 72), gb in (1, 30, 50, 80, 256, 1024),
            ab in (false, true)
            s = recommended_solve_settings(dh; cores_per_node=cores, memory_per_node_gb=gb,
                                           mumps_available=true, assembly_bound=ab)
            s.solver === :replicated_lu && @test s.fits !== false
        end

        # The case that exposed it, kept explicit so a regression names itself.
        s = recommended_solve_settings(dh2d; cores_per_node=72, memory_per_node_gb=50,
                                       mumps_available=true, assembly_bound=true)
        @test s.solver === :distributed_solve
    end

    @testset "the memory budget classifies every observed cluster outcome" begin
        # Every 3D replicated-LU configuration the scaling campaign actually ran on a
        # 256 GB node, and whether it survived. `fits` must not call a killed run safe.
        # (ndofs, ranks, survived)
        observed = [(46875, 36, false),   # OOM-killed, J2 measured ~12 GB/rank
                    (46875, 9, true),
                    (46875, 1, true),
                    (107811, 9, false),   # OOM-killed
                    (107811, 1, true),
                    (273375, 1, true)]    # 62.5 GB/rank measured, ran
        spread = FerriteSolidMechanics._MATERIAL_MEMORY_SPREAD
        for (n, ranks, survived) in observed
            budget = spread * ranks * estimated_replicated_memory_gb(n, 3)
            verdict = budget <= 0.85 * 256
            survived || @test verdict === false   # a killed run must never read as fitting
        end
        # The surviving single-rank runs must still be reported as fitting, or the
        # helper would push everyone onto MUMPS needlessly.
        for (n, ranks) in ((46875, 1), (107811, 1), (273375, 1))
            @test spread * ranks * estimated_replicated_memory_gb(n, 3) <= 0.85 * 256
        end
    end

    @testset "DistributedSolveError reports both failure modes" begin
        # The extension is Unix-only, but the error type it throws lives in src/ and
        # a driver's recovery path keys off it, so the contract is checked everywhere.
        @test DistributedSolveError(:factorize, -9) isa LocalAssemblyFailure
        reported = DistributedSolveError(:factorize, -9)
        @test reported.relative_residual === NaN     # 2-arg form: no residual measured
        @test occursin("-9", sprint(showerror, reported))

        # MUMPS reporting success while returning an unusable du was observed on a
        # cluster run; that path carries the measured backward error instead of a code.
        unverified = DistributedSolveError(:verify, 0, 3.2e-3)
        @test unverified isa LocalAssemblyFailure
        msg = sprint(showerror, unverified)
        @test occursin("K * du = residual", msg)
        @test occursin("0.0032", msg)
    end

    @testset "estimated_replicated_memory_gb grows superlinearly and tracks measurement" begin
        # Fitted points from the cluster runs the model was built on (NeoHooke, GB).
        @test estimated_replicated_memory_gb(46875, 3) ≈ 7.15 rtol = 0.15
        @test estimated_replicated_memory_gb(107811, 3) ≈ 18.06 rtol = 0.15
        @test estimated_replicated_memory_gb(290642, 2) ≈ 4.26 rtol = 0.15

        # 3D grows as ndofs^1.43, so doubling the dofs costs clearly more than 2x
        r = (estimated_replicated_memory_gb(2_000_000, 3) - 1.6) /
            (estimated_replicated_memory_gb(1_000_000, 3) - 1.6)
        @test 2.5 < r < 2.9
        # 2D is near-linear, and cheaper than 3D at equal dofs
        @test estimated_replicated_memory_gb(500_000, 2) < estimated_replicated_memory_gb(500_000, 3)
    end
end
