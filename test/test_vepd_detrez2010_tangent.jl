using Test
using FerriteSolidMechanics
using Tensors

function _detrez_tangent_material()
    return VEPD_Detrez2010(200.0, 0.3, 10.0, 5.0, 0.1, 0.4, 1.0, 2.0, 5.0, [3.0], [1.5])
end

_tdot(A) = sqrt(A ⊡ A)

function _check_detrez_directional_tangent(F, H; dt, rtol=2e-4, atol=2e-7)
    mat = _detrez_tangent_material()
    state = create_state(mat)
    h = 1.0e-6

    dP_dF = Tensors.gradient(F_ -> Tensor{2,3}(compute_PK1_3D(mat, F_, dt, state)), F)
    dP_ad = dP_dF ⊡ H
    P_plus = Tensor{2,3}(compute_PK1_3D(mat, F + h * H, dt, state))
    P_minus = Tensor{2,3}(compute_PK1_3D(mat, F - h * H, dt, state))
    dP_fd = (P_plus - P_minus) / (2h)

    @test _tdot(dP_ad - dP_fd) <= atol + rtol * max(_tdot(dP_ad), _tdot(dP_fd))
end

@testset "VEPD_Detrez2010 directional tangent checks" begin
    F_elastic = Tensor{2,3}((1.01, 0.01, 0.0,
                             0.00, 0.995, 0.0,
                             0.00, 0.00, 0.998))
    H_elastic = Tensor{2,3}((0.7, 0.2, 0.0,
                             0.1, -0.4, 0.0,
                             0.0, 0.0, 0.3))
    _check_detrez_directional_tangent(F_elastic, H_elastic; dt=0.2)

    F_shear = Tensor{2,3}((1.02, 0.04, 0.0,
                           0.01, 0.99, 0.0,
                           0.00, 0.00, 1.00))
    H_shear = Tensor{2,3}((0.2, -0.3, 0.0,
                           0.4, 0.1, 0.0,
                           0.0, 0.0, -0.2))
    _check_detrez_directional_tangent(F_shear, H_shear; dt=0.5)
end
