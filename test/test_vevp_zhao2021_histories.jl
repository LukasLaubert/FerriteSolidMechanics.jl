using Test
using FerriteSolidMechanics
using Tensors

const ZHAO_HISTORY_PARAMETERS = (
    1.0, 100.0, 3, 0.05, 0.2, 1.4, 0.0,
    1.0e-3, 1.0e-3, 1.0, 1.0, 0.0, 0.0, 1.0,
)

function _zhao_iso_tension(lambda)
    lateral = inv(sqrt(lambda))
    return Tensor{2,3}((lambda, 0.0, 0.0,
                        0.0, lateral, 0.0,
                        0.0, 0.0, lateral))
end

function _zhao_step!(state, mat::VEVP_Zhao2021_AD, F, dt)
    sigma, Cik, muV, strain_max = FerriteSolidMechanics.solve_local_vevp_zhao_exact(F, dt, mat, state)
    state.current.Cik = Tensors.value.(Cik)
    state.current.muVk = Tensors.value(muV)
    state.current.strain_maxk = Tensors.value(strain_max)
    update_state!(state)
    return sigma
end

function _zhao_step!(state, mat::VEVP_Zhao2021_AT, F, dt)
    state_new, Sk, _, _ = FerriteSolidMechanics.set_vevp_zhao_stress_S_C(
        F, inv(F), dt * mat.vevp_dt_scale, mat, state.previous, 1,
    )
    state.current = state_new
    update_state!(state)
    return Tensor{2,3}(Matrix(F) * Matrix(Sk) * Matrix(F)' / det(F))
end

function _zhao_history_summary(mat, Fs, dts)
    state = create_state(mat)
    sigma11 = Float64[]
    muV = Float64[]
    strain_max = Float64[]
    for i in eachindex(dts)
        sigma = _zhao_step!(state, mat, Fs[i + 1], dts[i])
        push!(sigma11, sigma[1, 1])
        push!(muV, state.previous.muVk)
        push!(strain_max, state.previous.strain_maxk)
    end
    return (
        final_sigma11=last(sigma11),
        min_sigma11=minimum(sigma11),
        max_sigma11=maximum(sigma11),
        final_muV=last(muV),
        final_strain_max=last(strain_max),
        sigma11=sigma11,
    )
end

@testset "VEVP_Zhao2021 history regressions" begin
    histories = (
        monotonic=(
            Fs=[_zhao_iso_tension(1 + 0.06 * i / 30) for i in 0:30],
            dts=fill(1.0, 30),
        ),
        cyclic=(
            Fs=[_zhao_iso_tension(1 + 0.025 * sin(2pi * i / 60)) for i in 0:60],
            dts=fill(1.0, 60),
        ),
        relaxation=(
            Fs=vcat([_zhao_iso_tension(1 + 0.05 * i / 20) for i in 0:20],
                    fill(_zhao_iso_tension(1.05), 30)),
            dts=fill(1.0, 50),
        ),
    )

    ad = VEVP_Zhao2021_AD(ZHAO_HISTORY_PARAMETERS...)
    at = VEVP_Zhao2021_AT(ZHAO_HISTORY_PARAMETERS...)

    ad_monotonic = _zhao_history_summary(ad, histories.monotonic.Fs, histories.monotonic.dts)
    @test ad_monotonic.final_sigma11 ≈ 0.12014716987066784 rtol=1e-8
    @test ad_monotonic.max_sigma11 == ad_monotonic.final_sigma11
    @test ad_monotonic.final_muV ≈ 1.3999999848571238 rtol=1e-8
    @test ad_monotonic.final_strain_max ≈ 0.09133679494462199 rtol=1e-8

    at_monotonic = _zhao_history_summary(at, histories.monotonic.Fs, histories.monotonic.dts)
    @test at_monotonic.final_sigma11 ≈ 0.12544022461570387 rtol=1e-8
    @test at_monotonic.max_sigma11 == at_monotonic.final_sigma11
    @test at_monotonic.final_muV ≈ ad_monotonic.final_muV rtol=1e-12
    @test at_monotonic.final_strain_max ≈ ad_monotonic.final_strain_max rtol=1e-12

    ad_cyclic = _zhao_history_summary(ad, histories.cyclic.Fs, histories.cyclic.dts)
    @test ad_cyclic.final_sigma11 ≈ 1.5679024174630532e-5 rtol=1e-8 atol=1e-12
    @test ad_cyclic.min_sigma11 ≈ -0.05001153013778528 rtol=1e-8
    @test ad_cyclic.max_sigma11 ≈ 0.05001096770026412 rtol=1e-8
    @test ad_cyclic.final_muV ≈ 1.3999999999999815 rtol=1e-12

    at_cyclic = _zhao_history_summary(at, histories.cyclic.Fs, histories.cyclic.dts)
    @test at_cyclic.final_sigma11 ≈ 0.007351768462042977 rtol=1e-8
    @test at_cyclic.min_sigma11 ≈ -0.05091570175429548 rtol=1e-8
    @test at_cyclic.max_sigma11 ≈ 0.050858573507784964 rtol=1e-8
    @test at_cyclic.final_muV ≈ ad_cyclic.final_muV rtol=1e-12

    ad_relaxation = _zhao_history_summary(ad, histories.relaxation.Fs, histories.relaxation.dts)
    @test ad_relaxation.final_sigma11 ≈ 0.10007936507935446 rtol=1e-8
    @test ad_relaxation.max_sigma11 ≈ 0.10009365089436396 rtol=1e-8
    @test ad_relaxation.final_muV ≈ 1.3999999372425007 rtol=1e-8

    at_relaxation = _zhao_history_summary(at, histories.relaxation.Fs, histories.relaxation.dts)
    @test at_relaxation.final_sigma11 ≈ 0.10007936507935414 rtol=1e-8
    @test at_relaxation.max_sigma11 ≈ 0.10677630881041991 rtol=1e-8
    @test at_relaxation.final_muV ≈ ad_relaxation.final_muV rtol=1e-12
end
