using Test
using FerriteSolidMechanics
using LinearAlgebra
using Tensors

const MOONEY_RIVLIN_TEST_PARAMETERS = (1.2, 0.4, 50.0)

function _mooney_rivlin_mixed_mode_F()
    values = [
        1.18  0.20  0.05
       -0.08  0.92  0.12
        0.04 -0.03  1.10
    ]
    return Tensor{2,3,Float64}((i, j) -> values[i, j])
end

function _mooney_rivlin_mixed_mode_H()
    values = [
         0.03 -0.02  0.01
         0.04  0.02 -0.03
        -0.01  0.03  0.02
    ]
    return Tensor{2,3,Float64}((i, j) -> values[i, j])
end

function _mooney_rivlin_simple_shear_F(γ)
    return Tensor{2,3,Float64}((i, j) -> i == j ? 1.0 : (i == 1 && j == 2 ? γ : 0.0))
end

function _mooney_rivlin_simple_shear_H()
    return Tensor{2,3,Float64}((i, j) -> i == j ? 0.01 : (i == 1 && j == 2 ? 0.03 : 0.0))
end

function _check_mooney_rivlin_directional_tangent(F, H; tangent=:AT, h=1.0e-6, rtol=2.0e-5, atol=2.0e-7)
    mat = MooneyRivlin(MOONEY_RIVLIN_TEST_PARAMETERS...; tangent)
    _, dP_dF = FerriteSolidMechanics._mooney_rivlin_pk1_tangent(F, mat)
    P_plus = compute_PK1_3D(mat, F + h * H, 0.0, NoState())
    P_minus = compute_PK1_3D(mat, F - h * H, 0.0, NoState())
    finite_difference = (P_plus - P_minus) / (2h)
    linearized = dP_dF ⊡ H
    @test linearized ≈ finite_difference rtol=rtol atol=atol
end

function _check_mooney_rivlin_tangent_modes_agree(F, H; rtol=1.0e-12, atol=1.0e-12)
    mat_at = MooneyRivlin(MOONEY_RIVLIN_TEST_PARAMETERS...; tangent=:AT)
    mat_ad = MooneyRivlin(MOONEY_RIVLIN_TEST_PARAMETERS...; tangent=:AD)

    P_at, dP_at = FerriteSolidMechanics._mooney_rivlin_pk1_tangent(F, mat_at)
    P_ad, dP_ad = FerriteSolidMechanics._mooney_rivlin_pk1_tangent(F, mat_ad)

    @test P_at ≈ P_ad rtol=rtol atol=atol
    @test (dP_at ⊡ H) ≈ (dP_ad ⊡ H) rtol=rtol atol=atol
end

@testset "MooneyRivlin tangent checks" begin
    mixed_F = _mooney_rivlin_mixed_mode_F()
    mixed_H = _mooney_rivlin_mixed_mode_H()
    shear_F = _mooney_rivlin_simple_shear_F(2.5)
    shear_H = _mooney_rivlin_simple_shear_H()

    @test det(mixed_F) > 0.0
    @test det(shear_F) > 0.0

    @testset "mixed mode stretch/shear" begin
        _check_mooney_rivlin_tangent_modes_agree(mixed_F, mixed_H)
        _check_mooney_rivlin_directional_tangent(mixed_F, mixed_H; tangent=:AT)
        _check_mooney_rivlin_directional_tangent(mixed_F, mixed_H; tangent=:AD)
    end

    @testset "high-strain simple shear" begin
        _check_mooney_rivlin_tangent_modes_agree(shear_F, shear_H)
        _check_mooney_rivlin_directional_tangent(shear_F, shear_H; tangent=:AT)
        _check_mooney_rivlin_directional_tangent(shear_F, shear_H; tangent=:AD)
    end
end

@testset "MooneyRivlin constructor validation" begin
    @test_throws ArgumentError MooneyRivlin(NaN, 1.0, 50.0)
    @test_throws ArgumentError MooneyRivlin(1.0, NaN, 50.0)
    @test_throws ArgumentError MooneyRivlin(1.0, 1.0, NaN)
    @test_throws ArgumentError MooneyRivlin(-1.0, 1.0, 50.0)
    @test_throws ArgumentError MooneyRivlin(1.0, 1.0, 0.0)
    @test_throws ArgumentError MooneyRivlin(1.0, 1.0, -50.0)
    @test_throws ArgumentError MooneyRivlin(1.0, 1.0, 50.0; tangent=:invalid)
    @test MooneyRivlin(-0.2, 1.0, 50.0) isa MooneyRivlin
    @test MooneyRivlin(1.2, 0.4, 50.0).tangent == :AD
end
