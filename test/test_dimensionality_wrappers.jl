using Test
using Ferrite
using LinearAlgebra
using FerriteSolidMechanics
using Tensors

struct NonConvergingPlaneStressMaterial <: AbstractMaterial end

function FerriteSolidMechanics.compute_PK1_3D(::NonConvergingPlaneStressMaterial, F::Tensor{2,3,T}, dt, state) where T
    return Tensor{2,3,T}((zero(T), zero(T), zero(T),
                         zero(T), zero(T), zero(T),
                         zero(T), zero(T), one(T)))
end

FerriteSolidMechanics.update_state_from_3D!(state, ::NonConvergingPlaneStressMaterial, F, dt) = nothing

struct SlowPlaneStressMaterial <: AbstractMaterial end

function FerriteSolidMechanics.compute_PK1_3D(::SlowPlaneStressMaterial, F::Tensor{2,3,T}, dt, state) where T
    return Tensor{2,3,T}((zero(T), zero(T), zero(T),
                         zero(T), zero(T), zero(T),
                         zero(T), zero(T), F[3, 3] - T(2)))
end

FerriteSolidMechanics.update_state_from_3D!(state, ::SlowPlaneStressMaterial, F, dt) = nothing

const WRAPPER_ZHAO_ARGS = (1.0, 100.0, 2, 0.1, 0.1, 1.0, 0.5, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0)

function _single_element_response(mat, dim; u_max=1e-5, plane_stress=false)
    grid = dim == 2 ? generate_grid(Quadrilateral, (1, 1)) : generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, dim == 2 ? Lagrange{RefQuadrilateral,1}()^2 : Lagrange{RefHexahedron,1}()^3)
    close!(dh)

    ch = ConstraintHandler(dh)
    if dim == 2
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0], [1]))
        add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> [0.0], [2]))
        add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [u_max * t], [1]))
    else
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0], [1]))
        add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> [0.0], [2]))
        add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [u_max * t], [1]))
        !plane_stress && add!(ch, Dirichlet(:u, getfacetset(grid, "front"), (x, t) -> [0.0], [3]))
        !plane_stress && add!(ch, Dirichlet(:u, getfacetset(grid, "back"), (x, t) -> [0.0], [3]))
    end
    close!(ch)

    fem = create_assembler(mat, dh, ch)
    u = zeros(ndofs(dh))
    update!(ch, 1.0)
    apply!(u, ch)
    for iter in 1:15
        K, r = stiffness_matrix(fem, u; dt=1.0)
        apply_zero!(K, r, ch)
        if norm(r) < 1e-10
            break
        end
        u .-= K \ r
    end
    update_states!(fem)
    return compute_stresses(fem, u; dt=1.0)[1, 1]
end

@testset "Dimensionality wrappers" begin
    E, nu = 200000.0, 0.3
    hooke = Hooke(E, nu)
    F = Tensor{2,3}((1.01, 0.02, 0.0,
                     0.01, 0.99, 0.0,
                     0.0, 0.0, 1.02))
    σ = hooke.C ⊡ symmetric(F - one(F))
    @test compute_PK1_3D(hooke, F, 0.0, NoState()) ≈ det(F) * σ ⋅ inv(F)'

    native_strain = _single_element_response(Hooke2D(E, nu; plane_stress=false), 2; u_max=1e-5)
    wrapped_strain = _single_element_response(PlaneStrain(Hooke(E, nu)), 2; u_max=1e-5)
    @test native_strain[1, 1] ≈ wrapped_strain[1, 1] rtol=1e-10

    native_stress = _single_element_response(Hooke2D(E, nu; plane_stress=true), 2; u_max=1e-5)
    wrapped_stress = _single_element_response(PlaneStress(Hooke(E, nu)), 2; u_max=1e-5)
    @test native_stress[1, 1] ≈ wrapped_stress[1, 1] rtol=1e-10

    neo_stress = _single_element_response(PlaneStress(NeoHooke(100.0, 0.3)), 2; u_max=0.01)
    @test isfinite(neo_stress[1, 1])
    @test neo_stress[1, 1] > 0.0

    arruda_stress = _single_element_response(PlaneStress(ArrudaBoyce(2.0, 60.0, 20.0)), 2; u_max=0.02)
    @test all(isfinite, arruda_stress)
    @test arruda_stress[1, 1] > 0.0

    arruda_strain = _single_element_response(PlaneStrain(ArrudaBoyce(2.0, 60.0, 20.0)), 2; u_max=0.02)
    @test all(isfinite, arruda_strain)
    @test arruda_strain[1, 1] > 0.0

    mooney_stress = _single_element_response(PlaneStress(MooneyRivlin(1.2, 0.4, 50.0)), 2; u_max=0.02)
    @test all(isfinite, mooney_stress)
    @test mooney_stress[1, 1] > 0.0

    mooney_strain = _single_element_response(PlaneStrain(MooneyRivlin(1.2, 0.4, 50.0)), 2; u_max=0.02)
    @test all(isfinite, mooney_strain)
    @test mooney_strain[1, 1] > 0.0

    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)
    ch = ConstraintHandler(dh)
    close!(ch)
    fem = create_assembler(PlaneStress(NonConvergingPlaneStressMaterial()), dh, ch)
    u = zeros(ndofs(dh))
    @test_throws PlaneStressConvergenceError stiffness_matrix(fem, u; dt=1.0)

    result = try_stiffness_matrix(fem, u; dt=1.0)
    @test result.converged == false
    @test result.K === nothing
    @test result.r === nothing
    @test result.error isa PlaneStressConvergenceError
    @test result.error.reason == :small_newton_derivative

    @test_throws ArgumentError PlaneStress(Hooke(1.0, 0.3); tol=0.0)
    @test_throws ArgumentError PlaneStress(Hooke(1.0, 0.3); maxiter=0)

    fem_limited = create_assembler(PlaneStress(SlowPlaneStressMaterial(); maxiter=1), dh, ch)
    limited = try_stiffness_matrix(fem_limited, u; dt=1.0)
    @test limited.converged == false
    @test limited.error isa PlaneStressConvergenceError
    @test limited.error.reason == :local_newton_nonconvergence
    @test limited.error.iterations == 1

    fem_zhao_ad_plane_strain = create_assembler(PlaneStrain(VEVP_Zhao2021_AD(WRAPPER_ZHAO_ARGS...)), dh, ch; quadrature_order=1)
    zhao_u = [1.0e-4 * cos(i) for i in 1:ndofs(dh)]
    K_zhao, r_zhao = stiffness_matrix(fem_zhao_ad_plane_strain, zhao_u; dt=0.1)
    @test all(isfinite, K_zhao.nzval)
    @test all(isfinite, r_zhao)
    update_states!(fem_zhao_ad_plane_strain)
    zhao_stresses = compute_stresses(fem_zhao_ad_plane_strain, zhao_u; dt=0.1)
    @test all(s -> all(isfinite, s), zhao_stresses)

    zhao_at = VEVP_Zhao2021_AT(WRAPPER_ZHAO_ARGS...)
    fem_zhao_at = create_assembler(PlaneStrain(zhao_at), dh, ch; quadrature_order=1)
    K_zhao_at, r_zhao_at = stiffness_matrix(fem_zhao_at, zhao_u; dt=0.1)
    @test all(isfinite, K_zhao_at.nzval)
    @test all(isfinite, r_zhao_at)
    update_states!(fem_zhao_at)
    zhao_at_stresses = compute_stresses(fem_zhao_at, zhao_u; dt=0.1)
    @test all(s -> all(isfinite, s), zhao_at_stresses)

    fem_zhao_at_plane_stress = create_assembler(PlaneStress(zhao_at), dh, ch)
    @test_throws ArgumentError stiffness_matrix(fem_zhao_at_plane_stress, u; dt=1.0)
end

@testset "wrapped J2Plasticity stays AD-transparent in the elastic branch" begin
    # Only the elastic branch carries history through unpromoted, so only it can
    # mix Float64 history with the Dual stress the wrappers' AD pass creates
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)
    ch = ConstraintHandler(dh)
    close!(ch)

    u = [1.0e-3 * sin(i) for i in 1:ndofs(dh)]
    elastic_j2 = J2Plasticity(100.0, 0.3, 1.0, 10.0)   # σ₀=1.0 ⇒ elastic here
    plastic_j2 = J2Plasticity(100.0, 0.3, 1.0e-4, 10.0) # σ₀=1e-4 ⇒ plastic here

    for wrapper in (PlaneStrain, PlaneStress)
        for mat in (elastic_j2, plastic_j2)
            fem = create_assembler(wrapper(mat), dh, ch; quadrature_order=1)
            K, r = stiffness_matrix(fem, u; dt=1.0)
            @test all(isfinite, K.nzval)
            @test all(isfinite, r)
        end
    end
end
