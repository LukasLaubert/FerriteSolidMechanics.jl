using Test
using Ferrite
using FerriteSolidMechanics
using LinearAlgebra

const ZHAO_TANGENT_PARAMETERS = (
    1.0, 100.0, 3, 0.05, 0.2, 1.4, 0.0,
    1.0e-3, 1.0e-3, 1.0, 1.0, 0.0, 0.0, 1.0,
)

const ZHAO_INACTIVE_FLOW_PARAMETERS = (
    1.0, 100.0, 3, 0.05, 0.2, 1.4, 0.0,
    1.0e3, 1.0e3, 1.0, 1.0, 1.0e3, 1.0e3, 1.0,
)

function _zhao_single_cell_problem()
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)
    ch = ConstraintHandler(dh)
    close!(ch)
    return dh, ch
end

function _zhao_stiffness_and_residual(constructor, dh, ch, u; dt, parameters)
    mat = constructor(parameters...)
    fem = create_assembler(mat, dh, ch; quadrature_order=1)
    return stiffness_matrix(fem, u; dt=dt)
end

function _zhao_assembled_tangent_matches(constructor, parameters; rtol=2e-4, atol=2e-8)
    dh, ch = _zhao_single_cell_problem()
    u = [1.0e-3 * cos(0.4 * i) for i in 1:ndofs(dh)]
    direction = [sin(0.7 * i) for i in 1:ndofs(dh)]
    direction ./= norm(direction)
    h = 1.0e-6
    dt = 0.5

    K, _ = _zhao_stiffness_and_residual(constructor, dh, ch, u; dt, parameters)
    _, r_plus = _zhao_stiffness_and_residual(constructor, dh, ch, u .+ h .* direction; dt, parameters)
    _, r_minus = _zhao_stiffness_and_residual(constructor, dh, ch, u .- h .* direction; dt, parameters)

    dr_linearized = K * direction
    dr_fd = (r_plus - r_minus) ./ (2h)
    return norm(dr_linearized - dr_fd) <= atol + rtol * max(norm(dr_linearized), norm(dr_fd))
end

@testset "VEVP_Zhao2021 assembled tangent checks" begin
    @test _zhao_assembled_tangent_matches(VEVP_Zhao2021_AD, ZHAO_TANGENT_PARAMETERS)
    @test _zhao_assembled_tangent_matches(VEVP_Zhao2021_AT, ZHAO_INACTIVE_FLOW_PARAMETERS)

    # The analytical AT stiffness follows the CAPRICCIO/Zhao assembly path,
    # but is not fully finite-difference consistent for active flow.
end
