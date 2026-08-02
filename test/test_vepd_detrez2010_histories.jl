using Test
using FerriteSolidMechanics
using Tensors

const DETREZ_POLYBUTENE = (
    167.0, 0.27, 3.24, 28.6, 20.8,
    0.59, 41.7, 4.5, 3.58,
    [101.0, 21.0], [10.0, 100.0],
)

function _detrez_iso_tension(lambda)
    lateral = inv(sqrt(lambda))
    return Tensor{2,3}((lambda, 0.0, 0.0,
                        0.0, lateral, 0.0,
                        0.0, 0.0, lateral))
end

function _detrez_step!(state, mat, F, dt)
    sigma, Fp, p, Cv = FerriteSolidMechanics.solve_local_vepd(F, dt, mat, state)
    state.current_Fp = Tensors.value(Fp)
    state.current_p = Tensors.value(p)
    for i in eachindex(state.current_Cv)
        state.current_Cv[i] = Tensors.value(Cv[i])
    end
    update_state!(state)
    return sigma
end

function _detrez_history_summary(Fs, dts)
    mat = VEPD_Detrez2010(DETREZ_POLYBUTENE...)
    state = create_state(mat)
    sigma11 = Float64[]
    ps = Float64[]
    Ds = Float64[]
    for i in eachindex(dts)
        sigma = _detrez_step!(state, mat, Fs[i + 1], dts[i])
        p = state.previous_p
        D = min(1.0, mat.α * (1.0 - exp(-mat.β * p)))
        push!(sigma11, sigma[1, 1])
        push!(ps, p)
        push!(Ds, D)
    end
    return (
        n=length(dts),
        final_sigma11=last(sigma11),
        min_sigma11=minimum(sigma11),
        max_sigma11=maximum(sigma11),
        max_p=maximum(ps),
        max_D=maximum(Ds),
        sigma11=sigma11,
    )
end

@testset "VEPD_Detrez2010 polybutene history regressions" begin
    monotonic = _detrez_history_summary(
        [_detrez_iso_tension(1 + 0.10 * i / 40) for i in 0:40],
        fill(2.5, 40),
    )
    @test monotonic.n == 40
    @test monotonic.final_sigma11 ≈ 12.1374196 rtol=1e-6
    @test monotonic.max_p ≈ 0.0264596121 rtol=1e-6
    @test monotonic.max_D ≈ 0.394265977 rtol=1e-6
    @test monotonic.final_sigma11 == monotonic.max_sigma11

    cyclic = _detrez_history_summary(
        [_detrez_iso_tension(1 + 0.04 * sin(2π * i / 80)) for i in 0:80],
        fill(2.5, 80),
    )
    @test cyclic.n == 80
    @test cyclic.final_sigma11 ≈ 2.09503192 rtol=1e-6
    @test cyclic.min_sigma11 ≈ -7.35356121 rtol=1e-6
    @test cyclic.max_sigma11 ≈ 6.82237286 rtol=1e-6
    @test cyclic.max_p ≈ 0.00928498489 rtol=1e-6
    @test cyclic.max_D ≈ 0.189409917 rtol=1e-6

    relaxation = _detrez_history_summary(
        vcat([_detrez_iso_tension(1 + 0.08 * i / 20) for i in 0:20],
             fill(_detrez_iso_tension(1.08), 40)),
        fill(2.5, 60),
    )
    hold_start = relaxation.sigma11[20]
    hold_drop = hold_start - relaxation.final_sigma11
    @test relaxation.n == 60
    @test relaxation.final_sigma11 ≈ 8.00480101 rtol=1e-6
    @test relaxation.max_sigma11 ≈ 12.313449 rtol=1e-6
    @test relaxation.max_p ≈ 0.0191003811 rtol=1e-6
    @test relaxation.max_D ≈ 0.323962668 rtol=1e-6
    @test hold_drop ≈ 4.30864799 rtol=1e-6
    @test relaxation.final_sigma11 < hold_start
end

@testset "VEPD_Detrez2010 update mode options execute" begin
    F = _detrez_iso_tension(1.05)
    for plastic_update in (:end_step, :path_substepped), maxwell_update in (:closed_form_cv, :objective_rate)
        mat = VEPD_Detrez2010(DETREZ_POLYBUTENE...; plastic_update, maxwell_update)
        state = create_state(mat)
        P = FerriteSolidMechanics.compute_PK1_3D(mat, F, 1.0, state)
        @test isfinite(P[1, 1])

        FerriteSolidMechanics.update_state_from_3D!(state, mat, F, 1.0)
        @test isfinite(state.current_p)
        @test state.current_F[1, 1] ≈ F[1, 1]
        @test all(isfinite(state.current_Cv[i][1, 1]) for i in eachindex(state.current_Cv))
        @test all(isfinite(state.current_σv[i][1, 1]) for i in eachindex(state.current_σv))
    end
end
