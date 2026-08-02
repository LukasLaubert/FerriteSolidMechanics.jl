# Tests for the MaterialModelsBase.jl adapter (FromMaterialModelsBase).
#
# These tests need the unregistered packages MaterialModelsBase.jl and
# MechanicalMaterialModels.jl (install by URL, see docs/src/models/index.md).
# runtests.jl includes this file only when both are loadable, so the main
# suite stays self-contained; run standalone with e.g.
#   julia --project=<env with both packages + FerriteSolidMechanics> test/test_materialmodelsbase_adapter.jl
using Test
using Ferrite
using FerriteSolidMechanics
using Tensors
import MaterialModelsBase
# Explicit-name using: MechanicalMaterialModels re-exports MaterialModelsBase's
# material_response, which would clash with FerriteSolidMechanics' export.
using MechanicalMaterialModels: Plastic, LinearElastic, Voce, ArmstrongFrederick, NortonOverstress

# A deliberately "hard" model: von Mises viscoplasticity with nonlinear
# isotropic (Voce) + nonlinear kinematic (Armstrong-Frederick) hardening and
# Norton overstress (rate-dependent).
function _hard_mmb_plastic()
    return Plastic(;
        elastic    = LinearElastic(E=210.0e3, ν=0.3),
        yield      = 100.0,
        isotropic  = Voce(Hiso=50.0e3, κ∞=100.0),
        kinematic  = ArmstrongFrederick(Hkin=200.0e3, β∞=200.0),
        overstress = NortonOverstress(tstar=1.0, nexp=2.0),
    )
end

function _mmb_fe_problem()
    grid = generate_grid(Hexahedron, (2, 2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.002 * t], [1]))
    close!(ch)
    return dh, ch
end

@testset "MaterialModelsBase adapter" begin
    @testset "material-point equivalence with direct MMB call" begin
        mmb_mat = _hard_mmb_plastic()
        w = FromMaterialModelsBase(mmb_mat)  # SmallStrain default
        @test kinematics(w) isa SmallStrain

        state = create_state(w)
        @test state isa MMBState

        ε = SymmetricTensor{2,3}((i, j) -> i == j == 1 ? 2.0e-3 : (i == j ? -0.6e-3 : 1.0e-4))
        dt = 0.1
        σ_w, D_w, new_w = material_response(w, ε, state, dt)
        σ_d, D_d, new_d = MaterialModelsBase.material_response(mmb_mat, ε, state.previous, dt)
        @test σ_w == σ_d
        @test D_w == D_d
        @test new_w == new_d
        @test norm(σ_d) > 0.0

        # Rate dependence through the adapter (Norton overstress): different
        # dt must give a different stress beyond yield.
        σ_fast, _, _ = material_response(w, ε, state, 1.0e-3)
        @test !(σ_fast ≈ σ_w)

        # Trial/commit/revert round-trip via the standard state lifecycle.
        set_trial!(state, new_w)
        @test state.current == new_w
        update_state!(state)
        @test state.previous == new_w
        σ_after_commit, _, _ = material_response(w, ε, state, dt)
        @test !(σ_after_commit == σ_w)  # committed hardening changes the response
        revert_state!(state)
        @test state.current == state.previous
    end

    @testset "assembly, Newton step, stress output, state lifecycle" begin
        dh, ch = _mmb_fe_problem()
        w = FromMaterialModelsBase(_hard_mmb_plastic())
        fem = create_assembler(w, dh, ch)

        u = zeros(ndofs(dh))
        update!(ch, 1.0)
        apply!(u, ch)
        dt = 0.05

        # A couple of Newton iterations; the residual norm (on free dofs) must drop.
        local rnorm_first = 0.0
        local rnorm_last = Inf
        for iter in 1:4
            K, r = stiffness_matrix(fem, u; dt=dt)
            apply_zero!(K, r, ch)
            rnorm = norm(r)
            iter == 1 && (rnorm_first = rnorm)
            rnorm_last = rnorm
            rnorm < 1e-8 && break
            u .-= K \ r
        end
        @test rnorm_last < 1e-6 * max(rnorm_first, 1.0)

        update_states!(fem)
        σ = copy(compute_stresses(fem, u; dt=0.0))
        @test all(s -> all(isfinite, s), σ)
        @test maximum(s -> norm(s), σ) > 0.0

        # States committed: previous == current on an owned cell.
        state1 = fem.states[fem.owned_nonlinear_cells[1]][1]
        @test state1 isa MMBState
        @test state1.current == state1.previous

        # revert_states! after further trial assembly restores committed history.
        committed = state1.previous
        stiffness_matrix(fem, 1.5 .* u; dt=dt)
        revert_states!(fem)
        @test state1.current == committed
    end

    @testset "extension guard message without loaded state" begin
        w = FromMaterialModelsBase(_hard_mmb_plastic())
        # Passing a non-MMBState routes to the in-package stub with the
        # informative error (simulates use without the extension loaded).
        ε = zero(SymmetricTensor{2,3})
        @test_throws ArgumentError material_response(w, ε, NoState(), 0.0)
    end
end
