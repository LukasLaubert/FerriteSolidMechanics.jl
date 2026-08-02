using Test
using Ferrite
using FerriteSolidMechanics
using FerriteSolidMechanics.Experimental: VEPD_Detrez2010_Optimized, VEPD_Detrez2010_Implicit, VEPD_Detrez2010_ExactVisco, VEPD_Detrez2010_ClosedCVEndStep, VEVP_Zhao2021_AD_Simplified, VEVP_Zhao2021_AT_Matlab, VEVP_MOAMMM_VarsN
using LinearAlgebra
using SparseArrays

const VEPD_PARAMETERS = (;
    E=200.0, ν=0.3, R0=10.0, Q=5.0, b=0.1,
    α=1.0, β=1.0, n_ab=1.0, μ_ab=10.0,
    G=[10.0, 5.0], τ=[0.5, 2.0],
)

const MUSTAFA_PARAMETERS = (
    6, 80.0, 40.0, 0.25, 0.3, 2.0, 1.0,
    40.0, 10.0, 5.0, 1.0, 45.0, 10.0, 5.0, 1.0,
    1.0, 0.1, 0.01,
    fill(40.0, 8),
    [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3],
    fill(15.0, 8),
    [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3],
)

const ZHAO_PARAMETERS = (1.0, 100.0, 2, 0.1, 0.1, 1.0, 0.5, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0)
const ZHAO_DT_SCALE_PARAMETERS = (1.0, 100.0, 2, 0.1, 0.1, 1.0, 0.5, 1.0e-3, 1.0e-3, 1.0, 1.0, 0.0, 0.0, 1.0)

struct DummyLocalAssemblyFailure <: LocalAssemblyFailure end

# Minor-symmetric but NOT major-symmetric: a SymmetricTensor{4} guarantees only
# D[i,j,k,l] == D[j,i,k,l] == D[i,j,l,k]. Mirroring a half ke loop is invalid here.
const UNSYMMETRIC_D = SymmetricTensor{4,3}((i, j, k, l) ->
    (i == j && k == l) ? (i == 1 && k == 2 ? 5.0 : 1.0) : (i == k && j == l ? 0.5 : 0.0))

struct UnsymmetricTangentMaterial <: AbstractMaterial end
FerriteSolidMechanics.kinematics(::UnsymmetricTangentMaterial) = SmallStrain()
FerriteSolidMechanics.material_response(::UnsymmetricTangentMaterial, ε::SymmetricTensor{2,3}, state, dt, cache=nothing) =
    (UNSYMMETRIC_D ⊡ ε, UNSYMMETRIC_D, state)

const PRODUCTION_MATERIAL_CASES = [
    (name="Hooke", material=Hooke(100.0, 0.3), dim=3),
    (name="Hooke2D", material=Hooke2D(100.0, 0.3), dim=2),
    (name="NeoHooke", material=NeoHooke(100.0, 0.3), dim=3),
    (name="ArrudaBoyce", material=ArrudaBoyce(2.0, 60.0, 20.0), dim=3),
    (name="MooneyRivlin", material=MooneyRivlin(1.2, 0.4, 50.0), dim=3),
    (name="Ogden", material=Ogden([1.2, 0.4], [2.0, -1.3], 50.0), dim=3),
    (name="J2Plasticity", material=J2Plasticity(100.0, 0.3, 1.0, 10.0), dim=3),
    (name="VEPD_Detrez2010", material=VEPD_Detrez2010(200.0, 0.3, 10.0, 5.0, 0.1, 1.0, 1.0, 1.0, 10.0, [10.0, 5.0], [0.5, 2.0]), dim=3),
    (name="VEVP_Zhao2021_AD", material=VEVP_Zhao2021_AD(ZHAO_PARAMETERS...), dim=3),
    (name="VEVP_Zhao2021_AT", material=VEVP_Zhao2021_AT(ZHAO_PARAMETERS...), dim=3),
    (name="VEVP_MOAMMM", material=VEVP_MOAMMM(MUSTAFA_PARAMETERS...), dim=3),
]

const EXPERIMENTAL_MATERIAL_CASES = [
    (name="VEPD_Detrez2010_Optimized", material=VEPD_Detrez2010_Optimized(; VEPD_PARAMETERS...), dim=3),
    (name="VEPD_Detrez2010_Implicit", material=VEPD_Detrez2010_Implicit(; VEPD_PARAMETERS...), dim=3),
    (name="VEPD_Detrez2010_ExactVisco", material=VEPD_Detrez2010_ExactVisco(; VEPD_PARAMETERS...), dim=3),
    (name="VEPD_Detrez2010_ClosedCVEndStep", material=VEPD_Detrez2010_ClosedCVEndStep(; VEPD_PARAMETERS...), dim=3),
    (name="VEVP_Zhao2021_AD_Simplified", material=VEVP_Zhao2021_AD_Simplified(1.0, 100.0, 2, 0.1, 0.1, 1.0, 0.5, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0), dim=3),
    (name="VEVP_Zhao2021_AT_Matlab", material=VEVP_Zhao2021_AT_Matlab(ZHAO_PARAMETERS...), dim=3),
    (name="VEVP_MOAMMM_VarsN", material=VEVP_MOAMMM_VarsN(MUSTAFA_PARAMETERS...), dim=3),
]

function _single_cell_material_problem(dim::Int)
    if dim == 2
        grid = generate_grid(Quadrilateral, (1, 1))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    else
        grid = generate_grid(Hexahedron, (1, 1, 1))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    end
    close!(dh)
    ch = ConstraintHandler(dh)
    close!(ch)
    return grid, dh, ch
end

# Assembled tangent vs central-difference Jacobian of the residual (relative
# Frobenius). Valid because assembly only writes trial state, never commits;
# the final base evaluation restores trial state at u for a later commit.
function _tangent_matches_fd(fem, u, dt, K_dense; h=1.0e-5, rtol=1.0e-6)
    K_fd = similar(K_dense)
    for j in eachindex(u)
        u_plus = copy(u); u_plus[j] += h
        u_minus = copy(u); u_minus[j] -= h
        r_plus = copy(stiffness_matrix(fem, u_plus; dt=dt)[2])
        r_minus = copy(stiffness_matrix(fem, u_minus; dt=dt)[2])
        @views K_fd[:, j] .= (r_plus .- r_minus) ./ (2h)
    end
    stiffness_matrix(fem, u; dt=dt)
    return norm(K_fd .- K_dense) <= rtol * norm(K_dense)
end

function _check_material_assembly_and_stress_extraction(mat; dim::Int=3, dt::Float64=0.1)
    grid, dh, ch = _single_cell_material_problem(dim)
    fem = create_assembler(mat, dh, ch; quadrature_order=1)
    u = [1.0e-2 * sin(i) for i in 1:ndofs(dh)]

    K, r = stiffness_matrix(fem, u; dt=dt)
    @test all(isfinite, r)
    # Before update_states! so K and the difference quotient share the same history
    @test _tangent_matches_fd(fem, u, dt, Matrix(K))

    update_states!(fem)
    stresses = compute_stresses(fem, u; dt=dt)
    @test size(stresses, 2) == getncells(grid)
    @test all(s -> all(isfinite, s), stresses)
    return nothing
end

function _zhao_state_for_dt_scale(mat; dt::Float64=1.0)
    _, dh, ch = _single_cell_material_problem(3)
    fem = create_assembler(mat, dh, ch; quadrature_order=1)
    u = [1.0e-2 * cos(i) for i in 1:ndofs(dh)]
    stiffness_matrix(fem, u; dt=dt)
    return deepcopy(fem.states[1][1].current)
end

function _zhao_states_match(a, b)
    return isapprox(a.muVk, b.muVk; atol=1e-12, rtol=1e-12) &&
           isapprox(a.strain_maxk, b.strain_maxk; atol=1e-12, rtol=1e-12) &&
           all(isapprox.(a.Cik, b.Cik; atol=1e-12, rtol=1e-12))
end

@testset "production material assembly and stress extraction" begin
    for case in PRODUCTION_MATERIAL_CASES
        @testset "$(case.name)" begin
            _check_material_assembly_and_stress_extraction(case.material; dim=case.dim)
        end
    end
end

@testset "tangent symmetry trait" begin
    # Default must be Unsymmetric: claiming MajorSymmetric here drops the
    # antisymmetric part of D and leaves the assembled K ~65% off the true Jacobian
    mat = UnsymmetricTangentMaterial()
    @test tangent_symmetry(mat) isa Unsymmetric
    @test !(UNSYMMETRIC_D ≈ permutedims(UNSYMMETRIC_D, (3, 4, 1, 2)))

    _, dh, ch = _single_cell_material_problem(3)
    fem = create_assembler(mat, dh, ch; quadrature_order=2)
    u = [1.0e-3 * sin(i) for i in 1:ndofs(dh)]
    K, r = stiffness_matrix(fem, u)
    @test all(isfinite, r)
    @test _tangent_matches_fd(fem, u, 0.0, Matrix(K))

    # A wrong MajorSymmetric claim corrupts ke silently, so pin the property itself
    ε3_elastic = SymmetricTensor{2,3}((i, j) -> i == j == 1 ? 1.0e-5 : 0.0)
    ε3_plastic = SymmetricTensor{2,3}((i, j) -> i == j == 1 ? 5.0e-3 : 0.0)
    ε2 = SymmetricTensor{2,2}((i, j) -> i == j == 1 ? 1.0e-4 : 0.0)
    j2 = J2Plasticity(200.0e3, 0.3, 200.0, 10.0e3)
    for (mp, ε) in ((Hooke(210.0e3, 0.3), ε3_elastic), (Hooke2D(210.0e3, 0.3), ε2),
                    (j2, ε3_elastic), (j2, ε3_plastic))
        @test tangent_symmetry(mp) isa MajorSymmetric
        _, D, _ = material_response(mp, ε, create_state(mp), 0.0)
        @test D ≈ permutedims(D, (3, 4, 1, 2))
    end
    # the plastic branch must actually have been exercised above
    σ_pl, _, _ = material_response(j2, ε3_plastic, create_state(j2), 0.0)
    @test sqrt(1.5) * norm(dev(σ_pl)) > j2.σ₀
end

@testset "VEPD_Detrez2010 update mode assembly paths" begin
    for plastic_update in (:end_step, :path_substepped), maxwell_update in (:closed_form_cv, :objective_rate)
        mat = VEPD_Detrez2010(200.0, 0.3, 10.0, 5.0, 0.1, 1.0, 1.0, 1.0, 10.0,
                         [10.0, 5.0], [0.5, 2.0]; plastic_update, maxwell_update)
        @testset "$plastic_update / $maxwell_update" begin
            _check_material_assembly_and_stress_extraction(mat; dim=3, dt=0.1)
        end
    end
end

@testset "Zhao dt_scale affects state evolution" begin
    for constructor in (VEVP_Zhao2021_AD, VEVP_Zhao2021_AT)
        state_default = _zhao_state_for_dt_scale(constructor(ZHAO_DT_SCALE_PARAMETERS...; dt_scale=1.0))
        state_scaled = _zhao_state_for_dt_scale(constructor(ZHAO_DT_SCALE_PARAMETERS...; dt_scale=10.0))
        @test !_zhao_states_match(state_default, state_scaled)
    end
end

@testset "recoverable task exception classification" begin
    recoverable = DummyLocalAssemblyFailure()
    bug = ArgumentError("not recoverable")

    @test FerriteSolidMechanics._recoverable_local_assembly_failure(recoverable) === recoverable
    @test FerriteSolidMechanics._recoverable_local_assembly_failure(CompositeException([recoverable])) === recoverable
    @test FerriteSolidMechanics._recoverable_local_assembly_failure(CompositeException([recoverable, DummyLocalAssemblyFailure()])) === recoverable
    @test FerriteSolidMechanics._recoverable_local_assembly_failure(bug) === nothing
    @test FerriteSolidMechanics._recoverable_local_assembly_failure(CompositeException([recoverable, bug])) === nothing
end

@testset "VEPD_Detrez2010 local nonconvergence is recoverable" begin
    # Pathological parameters force the local plastic Newton solve to exhaust
    # its iteration budget; this exercises the recoverable failure path.
    mat = VEPD_Detrez2010(0.0, 0.3, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, Float64[], Float64[])
    _, dh, ch = _single_cell_material_problem(3)
    fem = create_assembler(mat, dh, ch; quadrature_order=1)
    result = try_stiffness_matrix(fem, zeros(ndofs(dh)); dt=1.0)

    @test result.converged == false
    @test result.K === nothing
    @test result.r === nothing
    @test result.error isa VEPD_Detrez2010ConvergenceError
end

@testset "VEVP_MOAMMM local nonconvergence is recoverable" begin
    # Pathological parameters force the local plastic Newton solve to exhaust
    # its iteration budget; this exercises the recoverable failure path.
    mat = VEVP_MOAMMM(
        6, 80.0, 40.0, 3.0, -0.99, 1.0e-12, 0.05,
        1.0e-9, 10.0, 5.0, 1.0, 1.0, 10.0, 5.0, 1.0,
        1.0, 0.1, 0.01,
        Float64[], Float64[], Float64[], Float64[],
    )
    _, dh, ch = _single_cell_material_problem(3)
    fem = create_assembler(mat, dh, ch; quadrature_order=1)
    u = [1.0e-3 * sin(i) for i in 1:ndofs(dh)]
    result = try_stiffness_matrix(fem, u; dt=1.0e-9)

    @test result.converged == false
    @test result.K === nothing
    @test result.r === nothing
    @test result.error isa VEVP_MOAMMMConvergenceError
    @test isfinite(result.error.plastic_multiplier)
    @test isfinite(result.error.residual)
end

@testset "experimental material assembly and stress extraction" begin
    for case in EXPERIMENTAL_MATERIAL_CASES
        @testset "$(case.name)" begin
            _check_material_assembly_and_stress_extraction(case.material; dim=case.dim)
        end
    end
end