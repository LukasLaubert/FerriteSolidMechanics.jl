using Test
using Ferrite
using FerriteSolidMechanics
using LinearAlgebra
using Tensors

const OGDEN_TERM_CASES = (
    (mu=[1.2], alpha=[2.0], kappa=80.0),
    (mu=[1.2, 0.4], alpha=[2.0, -1.3], kappa=80.0),
    (mu=[1.2, 0.4, -0.08], alpha=[2.0, -1.3, 5.0], kappa=80.0),
    (mu=[1.2, 0.4, -0.08, 0.03], alpha=[2.0, -1.3, 5.0, -4.0], kappa=80.0),
)

const OGDEN_TEST_CASE = OGDEN_TERM_CASES[3]

_ogden_material(case) = Ogden(case.mu, case.alpha, case.kappa)
_ogden_test_material() = _ogden_material(OGDEN_TEST_CASE)

function _ogden_mixed_mode_F()
    values = [
        1.18  0.20  0.05
       -0.08  0.92  0.12
        0.04 -0.03  1.10
    ]
    return Tensor{2,3,Float64}((i, j) -> values[i, j])
end

function _ogden_mixed_mode_H()
    values = [
         0.03 -0.02  0.01
         0.04  0.02 -0.03
        -0.01  0.03  0.02
    ]
    return Tensor{2,3,Float64}((i, j) -> values[i, j])
end

function _ogden_simple_shear_F(gamma)
    return Tensor{2,3,Float64}((i, j) -> i == j ? 1.0 : (i == 1 && j == 2 ? gamma : 0.0))
end

function _ogden_simple_shear_H()
    return Tensor{2,3,Float64}((i, j) -> i == j ? 0.01 : (i == 1 && j == 2 ? 0.03 : 0.0))
end

function _ogden_ad_pk1_tangent(F, mat)
    dP_dF, P = Tensors.gradient(F_ -> FerriteSolidMechanics._pk1_from_second_piola(F_, mat), F, :all)
    return P, dP_dF
end

function _ogden_directional_tangent_ok(mat, F, H; h=1.0e-6, rtol=3.0e-5, atol=3.0e-7)
    _, dP_dF = FerriteSolidMechanics._ogden_pk1_tangent(F, mat)
    P_plus = compute_PK1_3D(mat, F + h * H, 0.0, NoState())
    P_minus = compute_PK1_3D(mat, F - h * H, 0.0, NoState())
    finite_difference = (P_plus - P_minus) / (2h)
    return isapprox(dP_dF ⊡ H, finite_difference; rtol, atol)
end

function _ogden_response_finite(mat, Fs)
    return all(Fs) do F
        P, dP = FerriteSolidMechanics._ogden_pk1_tangent(F, mat)
        all(isfinite, P) && all(isfinite, dP)
    end
end

function _single_cell_ogden_problem(dim::Int)
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

function _ogden_assembly_ok(mat, dim; u_scale=0.0)
    grid, dh, ch = _single_cell_ogden_problem(dim)
    fem = create_assembler(mat, dh, ch; quadrature_order=1)
    u = u_scale == 0.0 ? zeros(ndofs(dh)) : [u_scale * sin(i) for i in 1:ndofs(dh)]
    K, r = stiffness_matrix(fem, u; dt=1.0)
    update_states!(fem)
    stresses = compute_stresses(fem, u; dt=1.0)
    return all(isfinite, K.nzval) && all(isfinite, r) && size(stresses, 2) == getncells(grid) && all(s -> all(isfinite, s), stresses)
end

@testset "Ogden tangents and term counts" begin
    mixed_F = _ogden_mixed_mode_F()
    mixed_H = _ogden_mixed_mode_H()
    shear_F = _ogden_simple_shear_F(2.5)
    shear_H = _ogden_simple_shear_H()
    identity_F = one(Tensor{2,3,Float64})
    for (n, case) in enumerate(OGDEN_TERM_CASES)
        mat = _ogden_material(case)
        @test length(mat.μ) == n && length(mat.α) == n
        @test _ogden_response_finite(mat, (identity_F, mixed_F, shear_F))
        @test _ogden_directional_tangent_ok(mat, mixed_F, mixed_H)
        @test _ogden_directional_tangent_ok(mat, shear_F, shear_H)
    end
    mat = _ogden_test_material()
    P_at, dP_at = FerriteSolidMechanics._ogden_pk1_tangent(mixed_F, mat)
    P_ad, dP_ad = _ogden_ad_pk1_tangent(mixed_F, mat)
    @test isapprox(P_at, P_ad; rtol=1.0e-10, atol=1.0e-10)
    @test isapprox(dP_at ⊡ mixed_H, dP_ad ⊡ mixed_H; rtol=1.0e-10, atol=1.0e-10)
    _, dP_ad_identity = _ogden_ad_pk1_tangent(identity_F, mat)
    @test !all(isfinite, dP_ad_identity)
end

@testset "Ogden Mooney–Rivlin limiting case" begin
    mu = 2.4
    kappa = 50.0
    ogden = Ogden(mu, 2.0, kappa)
    mooney = MooneyRivlin(mu / 2, 0.0, kappa; tangent=:AT)
    F = _ogden_mixed_mode_F()
    P_ogden, dP_ogden = FerriteSolidMechanics._ogden_pk1_tangent(F, ogden)
    P_mooney, dP_mooney = FerriteSolidMechanics._mooney_rivlin_pk1_tangent(F, mooney)
    @test isapprox(P_ogden, P_mooney; rtol=1.0e-12, atol=1.0e-12)
    @test isapprox(dP_ogden, dP_mooney; rtol=1.0e-11, atol=1.0e-11)
end

@testset "Ogden constructor validation" begin
    @test Ogden(1.2, 2.0, 50.0) isa Ogden{1}
    @test Ogden((1.2, 0.3), (2.0, -1.0), 50.0) isa Ogden{2}
    @test_throws ArgumentError Ogden(Float64[], Float64[], 50.0)
    @test_throws ArgumentError Ogden([1.0, 2.0], [2.0], 50.0)
    @test_throws ArgumentError Ogden([NaN], [2.0], 50.0)
    @test_throws ArgumentError Ogden([1.0], [NaN], 50.0)
    @test_throws ArgumentError Ogden([1.0], [0.0], 50.0)
    @test_throws ArgumentError Ogden([1.0], [2.0], NaN)
    @test_throws ArgumentError Ogden([1.0], [2.0], 0.0)
    @test_throws ArgumentError Ogden([-1.0], [2.0], 50.0)
    @test_throws ArgumentError Ogden([1.0, 1.0], [2.0, -3.0], 50.0)
    @test Ogden([-0.2, 1.0], [-2.0, 1.5], 50.0) isa Ogden
end

@testset "Ogden specialized wrapper assembly" begin
    base = _ogden_test_material()
    @test _ogden_assembly_ok(PlaneStrain(base), 2; u_scale=0.0)
    @test _ogden_assembly_ok(PlaneStrain(base), 2; u_scale=2.0e-2)
    @test _ogden_assembly_ok(PlaneStress(base), 2; u_scale=0.0)
    @test _ogden_assembly_ok(PlaneStress(base), 2; u_scale=2.0e-2)
end
