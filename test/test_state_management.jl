using Test
using Ferrite
using FerriteSolidMechanics
using Tensors

const ZHAO_ARGS = (1.0, 100.0, 2, 0.1, 0.1, 1.0, 0.5, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0)

function _zhao_single_cell_problem(mat)
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    fem = create_assembler(mat, dh, ch; quadrature_order=1)
    u = [1.0e-3 * sin(i) for i in 1:ndofs(dh)]
    return fem, u
end

function _zhao_internal_matches(a, b)
    return isapprox(a.muVk, b.muVk; atol=1e-12, rtol=1e-12) &&
           isapprox(a.strain_maxk, b.strain_maxk; atol=1e-12, rtol=1e-12) &&
           all(isapprox.(a.Cik, b.Cik; atol=1e-12, rtol=1e-12))
end

@testset "State management" begin
    mustafa_args = (6, 800.0, 250.0, 0.25, 0.3, 2.0, 1.0, 40.0, 10.0, 5.0, 1.0, 45.0, 10.0, 5.0, 1.0, 1.0, 0.1, 0.01, [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0], [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3], [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0], [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3])

    mat = PlaneStress(VEPD_Detrez2010(
        2000.0, 0.3, 10.0, 5.0, 0.1,
        1.0, 1.0, 1.0, 10.0,
        [100.0, 50.0], [0.1, 1.0],
    ))

    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0], [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> [0.0], [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.05 * t], [1]))
    close!(ch)

    fem = create_assembler(mat, dh, ch)
    state0 = deepcopy(fem.states[1][1])
    u = zeros(ndofs(dh))
    update!(ch, 1.0)
    apply!(u, ch)
    stiffness_matrix(fem, u; dt=1.0)
    @test fem.states[1][1].F33_current != state0.F33_current
    revert_state!(fem.states[1][1])
    @test fem.states[1][1].F33_current ≈ state0.F33_current
    stiffness_matrix(fem, u; dt=1.0)
    update_states!(fem)
    @test fem.states[1][1].F33_previous ≈ fem.states[1][1].F33_current

    mixed_grid = Grid(
        Ferrite.AbstractCell[
            Quadrilateral((1, 2, 5, 4)),
            Triangle((2, 3, 5)),
            Triangle((3, 6, 5)),
        ],
        [Node(Vec(0.0, 0.0)), Node(Vec(1.0, 0.0)), Node(Vec(2.0, 0.0)),
         Node(Vec(0.0, 1.0)), Node(Vec(1.0, 1.0)), Node(Vec(2.0, 1.0))],
    )
    mdh = DofHandler(mixed_grid)
    add!(SubDofHandler(mdh, Set([1])), :u, Lagrange{RefQuadrilateral,1}()^2)
    add!(SubDofHandler(mdh, Set([2, 3])), :u, Lagrange{RefTriangle,1}()^2)
    close!(mdh)
    mch = ConstraintHandler(mdh)
    close!(mch)
    mfem = create_assembler(PlaneStrain(VEVP_MOAMMM(mustafa_args...)), mdh, mch)
    @test length(mfem.states[1]) == 4
    @test length(mfem.states[2]) == 3

    @testset "PlaneStrain wrapped state reverts" begin
        ps_grid = generate_grid(Quadrilateral, (1, 1))
        ps_dh = DofHandler(ps_grid)
        add!(ps_dh, :u, Lagrange{RefQuadrilateral,1}()^2)
        close!(ps_dh)
        ps_ch = ConstraintHandler(ps_dh)
        close!(ps_ch)

        ps_fem = create_assembler(PlaneStrain(J2Plasticity(100.0, 0.3, 1.0e-4, 10.0)), ps_dh, ps_ch; quadrature_order=1)
        ps_state = ps_fem.states[1][1]
        ps_initial = deepcopy(ps_state)
        ps_u = [1.0e-3 * sin(i) for i in 1:ndofs(ps_dh)]

        stiffness_matrix(ps_fem, ps_u; dt=1.0)
        @test ps_state.current != ps_initial.current
        revert_state!(ps_state)
        @test ps_state.current == ps_initial.previous
    end

    @testset "VEVP_MOAMMM custom Maxwell branch counts" begin
        for nbr in (0, 1, 3, 8, 10)
            KK = collect(range(40.0; step=5.0, length=nbr))
            k = collect(range(0.1; step=0.1, length=nbr))
            GG = collect(range(15.0; step=2.5, length=nbr))
            g = collect(range(0.2; step=0.2, length=nbr))
            mat_n = VEVP_MOAMMM(mustafa_args[1:18]..., KK, k, GG, g)
            state = create_state(mat_n)
            _, upd = FerriteSolidMechanics.solve_local_vevp_moammm(one(Tensor{2,3,Float64,9}), 0.1, mat_n, state)

            @test mat_n.nbr == nbr
            @test length(state.current_AA) == nbr
            @test length(state.current_BB) == nbr
            @test length(state.previous_AA) == nbr
            @test length(state.previous_BB) == nbr
            @test length(upd.AA) == nbr
            @test length(upd.BB) == nbr
            @test all(a -> all(isfinite, a), upd.AA)
            @test all(isfinite, upd.BB)

            # Mirror the assembly path: write a trial state, then commit it.
            state.current_Fvp = upd.Fvp
            state.current_Eve = upd.Eve
            state.current_gma = upd.gma
            state.current_b   = upd.b
            state.current_AA  = upd.AA
            state.current_BB  = upd.BB
            update_state!(state)
            @test state.previous_Fvp == upd.Fvp
            @test state.previous_Eve == upd.Eve
            @test state.previous_gma == upd.gma
            @test state.previous_b   == upd.b
            @test state.previous_AA  == upd.AA
            @test state.previous_BB  == upd.BB

            # Snapshot committed vector contents before mutating current values;
            # `upd.AA` and `upd.BB` intentionally alias `state.current_*` here.
            prev_AA_snap = copy(state.previous_AA)
            prev_BB_snap = copy(state.previous_BB)
            # Mutating current vectors must not affect committed previous vectors.
            for i in 1:nbr
                state.current_AA[i] = one(Tensor{2,3,Float64,9})
                state.current_BB[i] = -12345.0
            end
            @test state.previous_AA == prev_AA_snap
            @test state.previous_BB == prev_BB_snap

            revert_state!(state)
            @test state.current_Fvp == upd.Fvp
            @test state.current_Eve == upd.Eve
            @test state.current_gma == upd.gma
            @test state.current_b   == upd.b
            @test state.current_AA  == upd.AA
            @test state.current_BB  == upd.BB
        end

        @test_throws ArgumentError VEVP_MOAMMM(mustafa_args[1:18]..., [40.0], [0.1, 1.0], [15.0], [0.2])
    end

    @testset "minimal constructor validation" begin
        @test_throws ArgumentError VEPD_Detrez2010(
            2000.0, 0.3, 10.0, 5.0, 0.1, 1.0, 1.0, 1.0, 10.0,
            [100.0], [0.1, 1.0],
        )
        @test_throws ArgumentError VEPD_Detrez2010(
            2000.0, 0.3, 10.0, 5.0, 0.1, 1.0, 1.0, 1.0, 10.0,
            [100.0], [0.0],
        )
        @test_throws ArgumentError VEPD_Detrez2010(
            2000.0, 0.3, 10.0, 5.0, 0.1, 1.0, 1.0, 1.0, 10.0,
            [100.0], [0.1]; plastic_update=:invalid,
        )
        @test_throws ArgumentError VEPD_Detrez2010(
            2000.0, 0.3, 10.0, 5.0, 0.1, 1.0, 1.0, 1.0, 10.0,
            [100.0], [0.1]; maxwell_update=:invalid,
        )
        @test_throws ArgumentError VEVP_Zhao2021_AD(1.0, 100.0, 1, 0.1, 0.1, 1.0, 0.5, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0)
        @test_throws ArgumentError VEVP_Zhao2021_AT(1.0, 100.0, 1, 0.1, 0.1, 1.0, 0.5, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0)
    end

    @testset "Zhao state commit and revert" begin
        for mat in (VEVP_Zhao2021_AD(ZHAO_ARGS...), VEVP_Zhao2021_AT(ZHAO_ARGS...))
            fem, u = _zhao_single_cell_problem(mat)
            state = fem.states[1][1]
            initial = deepcopy(state)

            stiffness_matrix(fem, u; dt=1.0)
            @test !_zhao_internal_matches(state.current, initial.current)

            revert_state!(state)
            @test _zhao_internal_matches(state.current, initial.previous)

            stiffness_matrix(fem, u; dt=1.0)
            update_states!(fem)
            @test _zhao_internal_matches(state.previous, state.current)
        end
    end
end
