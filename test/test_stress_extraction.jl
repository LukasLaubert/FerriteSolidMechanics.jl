using Test
using Ferrite
using FerriteSolidMechanics
using LinearAlgebra
using Tensors

struct LinearMissingStressMaterial <: AbstractMaterial end
struct LinearNoStateStressMaterial <: AbstractMaterial end

FerriteSolidMechanics.is_linear(::Union{LinearMissingStressMaterial,LinearNoStateStressMaterial}) = true
FerriteSolidMechanics.create_state(::Union{LinearMissingStressMaterial,LinearNoStateStressMaterial}) = NoState()

function FerriteSolidMechanics._assemble_element!(ke, re, state, ::Union{LinearMissingStressMaterial,LinearNoStateStressMaterial}, cv, av, u, dt)
    fill!(ke, 0.0)
    return nothing
end

function FerriteSolidMechanics._compute_stress_qp(::LinearNoStateStressMaterial, cv, av, qp, u_local, state::NoState, dt=0.0)
    return one(deformation_gradient(cv, qp, u_local))
end

@testset "stateful stress postprocessing does not advance history" begin
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0], [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> [0.0], [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "front"), (x, t) -> [0.0], [3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.02 * t], [1]))
    close!(ch)

    mat = VEPD_Detrez2010(
        2000.0, 0.3, 10.0, 5.0, 0.1,
        1.0, 1.0, 1.0, 10.0,
        [100.0, 50.0], [0.1, 1.0],
    )
    fem = create_assembler(mat, dh, ch; quadrature_order=2)
    u = zeros(ndofs(dh))
    update!(ch, 1.0)
    apply!(u, ch)

    stiffness_matrix(fem, u; dt=1.0)
    update_states!(fem)

    stresses_dt = copy(compute_stresses(fem, u; dt=1.0))
    stresses_zero = copy(compute_stresses(fem, u; dt=0.0))

    @test stresses_dt ≈ stresses_zero atol=1e-10 rtol=1e-10
    @test norm(stresses_dt[1, 1]) > 0.0
end

@testset "stateless stress redispatch is explicit" begin
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)
    ch = ConstraintHandler(dh)
    close!(ch)

    u = zeros(ndofs(dh))

    fem = create_assembler(LinearNoStateStressMaterial(), dh, ch)
    stresses = compute_stresses(fem, u)
    @test stresses[1, 1] == one(Tensor{2,3})

    # A material with neither a _compute_stress_qp method nor the
    # material_response/kinematics pair must fail loudly (the generic stress
    # fallback asks for kinematics and raises an informative ArgumentError).
    fem_missing = create_assembler(LinearMissingStressMaterial(), dh, ch)
    @test_throws ArgumentError compute_stresses(fem_missing, u)

end

@testset "mixed-grid stress output keeps grid cell positions" begin
    grid = Grid(
        Ferrite.AbstractCell[
            Quadrilateral((1, 2, 5, 4)),
            Triangle((2, 3, 5)),
            Triangle((3, 6, 5)),
        ],
        [Node(Vec(0.0, 0.0)), Node(Vec(1.0, 0.0)), Node(Vec(2.0, 0.0)),
         Node(Vec(0.0, 1.0)), Node(Vec(1.0, 1.0)), Node(Vec(2.0, 1.0))],
    )
    dh = DofHandler(grid)
    add!(SubDofHandler(dh, Set([1])), :u, Lagrange{RefQuadrilateral,1}()^2)
    add!(SubDofHandler(dh, Set([2, 3])), :u, Lagrange{RefTriangle,1}()^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    fem = create_assembler(Hooke2D(100.0, 0.3), dh, ch; quadrature_order=2)
    u = collect(range(0.0, 0.01; length=ndofs(dh)))
    stresses = compute_stresses(fem, u)

    @test size(stresses) == (4, 3)
    @test norm(stresses[1, 1]) > 0.0
    @test norm(stresses[1, 2]) > 0.0
    @test stresses[4, 2] == zero(eltype(stresses))
    @test stresses[4, 3] == zero(eltype(stresses))
end

@testset "VEVP_MOAMMM postprocessing contract" begin
    # Regression test for the bare-Mustafa postprocessing contract:
    # finite stresses, dt-independent output, and deterministic repeated calls.
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> [0.0], [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> [0.0], [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "front"),  (x, t) -> [0.0], [3]))
    # 0.2 strain at t=1 -> drives the state plastic for these parameters
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"),  (x, t) -> [0.2 * t], [1]))
    close!(ch)

    mat = VEVP_MOAMMM(
        6, 80.0, 40.0, 0.25, 0.3, 2.0, 1.0,
        40.0, 10.0, 5.0, 1.0, 45.0, 10.0, 5.0, 1.0,
        1.0, 0.1, 0.01,
        [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
        [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3],
        [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3],
    )
    fem = create_assembler(mat, dh, ch; quadrature_order=2)
    u = zeros(ndofs(dh))
    update!(ch, 1.0)
    apply!(u, ch)

    # Drive into the plastic regime
    stiffness_matrix(fem, u; dt=1.0)
    update_states!(fem)

    stresses_dt   = copy(compute_stresses(fem, u; dt=1.0))
    stresses_zero = copy(compute_stresses(fem, u; dt=0.0))

    # stresses_dt/zero are arrays of Tensor{2,3,Float64,9}; isfinite isn't
    # defined on Tensor, so iterate the inner components.
    @test all(s -> all(isfinite, s), stresses_dt)
    @test all(s -> all(isfinite, s), stresses_zero)
    @test stresses_dt ≈ stresses_zero atol=1e-10 rtol=1e-10
    @test norm(stresses_dt[1, 1]) > 0.0

    # Determinism: two back-to-back calls without an intervening assembly
    # must give the same result (the postprocessing reads the committed state).
    stresses_again = copy(compute_stresses(fem, u; dt=0.0))
    @test stresses_again ≈ stresses_zero atol=1e-12 rtol=1e-12
end

@testset "VEVP_MOAMMM constructor validation" begin
    # Non-positive eta or p_exp would silently produce Inf in the corrector's
    # (eta_over_dt * G)^p_exp term. The constructor must reject these.
    # Positional layout matches the working call above:
    # (order, KK_inf, GG_inf, alpha, nu_p, eta, p_exp,
    #  sigmac0, hc1, hc2, hcexp, sigmat0, ht1, ht2, htexp,
    #  hb0, hb1, hb2, KK, k, GG, g)  -- 22 entries total
    common = (
        6, 80.0, 40.0, 0.25, 0.3,      # 1-5:   order, KK_inf, GG_inf, alpha, nu_p
        2.0, 1.0,                       # 6-7:   eta, p_exp
        40.0, 10.0, 5.0, 1.0, 45.0, 10.0, 5.0, 1.0,   # 8-15:  sigmac0, hc1, hc2, hcexp, sigmat0, ht1, ht2, htexp
        1.0, 0.1, 0.01,                 # 16-18: hb0, hb1, hb2
        [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0],   # 19: KK
        [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3],        # 20: k
        [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],    # 21: GG
        [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3],        # 22: g
    )
    # eta = 0 is rejected
    @test_throws ArgumentError VEVP_MOAMMM(common[1:5]..., 0.0, common[7:22]...)
    # eta < 0 is rejected
    @test_throws ArgumentError VEVP_MOAMMM(common[1:5]..., -1.0, common[7:22]...)
    # p_exp = 0 is rejected
    @test_throws ArgumentError VEVP_MOAMMM(common[1:6]..., 0.0, common[8:22]...)
    # p_exp < 0 is rejected
    @test_throws ArgumentError VEVP_MOAMMM(common[1:6]..., -1.0, common[8:22]...)
end
