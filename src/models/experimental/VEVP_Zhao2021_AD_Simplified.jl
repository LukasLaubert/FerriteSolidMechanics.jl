
using LinearAlgebra
using Tensors

####################
# Material structs #
####################

"""
    VEVP_Zhao2021_AD_Simplified(muE, kappa, NV, beta, alpha, muVe, muVN,
                         tauHat1, tauHatN, m1, mN, taucut1, taucutN, muV;
                         dt_scale=1.0)

Experimental comparison variant of `VEVP_Zhao2021_AD`.
The parameters have the same meanings, with ASCII names for `tauHat` and `taucut`.
`dt_scale` scales the time step.
"""
struct VEVP_Zhao2021_AD_Simplified <: AbstractMaterial
    vevp_muE::Float64
    vevp_kappa::Float64
    vevp_NV::Int
    vevp_beta::Float64
    vevp_alpha::Float64
    vevp_muVe::Float64
    vevp_muVN::Float64
    vevp_tauHat1::Float64
    vevp_tauHatN::Float64
    vevp_m1::Float64
    vevp_mN::Float64
    vevp_taucut1::Float64
    vevp_taucutN::Float64
    vevp_muV::Float64
    vevp_dt_scale::Float64
end

function VEVP_Zhao2021_AD_Simplified(muE, kappa, NV, beta, alpha, muVe, muVN, tauHat1, tauHatN,
                              m1, mN, taucut1, taucutN, muV; dt_scale=1.0)
    NV >= 2 || throw(ArgumentError("VEVP_Zhao2021_AD_Simplified: NV must be >= 2"))
    return VEVP_Zhao2021_AD_Simplified(muE, kappa, NV, beta, alpha, muVe, muVN, tauHat1,
                                tauHatN, m1, mN, taucut1, taucutN, muV, dt_scale)
end

mutable struct VEVP_Zhao2021_AD_SimplifiedStateInternal
    Cik::Vector{SymmetricTensor{2,3,Float64,6}}
    muVk::Float64
    strain_maxk::Float64
end

mutable struct VEVP_Zhao2021_AD_SimplifiedState <: AbstractMaterialState
    current::VEVP_Zhao2021_AD_SimplifiedStateInternal
    previous::VEVP_Zhao2021_AD_SimplifiedStateInternal
end

##########################
# AbstractMaterial hooks #
##########################

is_linear(::VEVP_Zhao2021_AD_Simplified) = false

function create_state(mat::VEVP_Zhao2021_AD_Simplified)
    NV = mat.vevp_NV
    muV = mat.vevp_muV

    function init_internal()
        Cik = [one(SymmetricTensor{2,3}) for _ in 1:NV]
        muVk = muV
        strain_maxk = 0.0
        return VEVP_Zhao2021_AD_SimplifiedStateInternal(Cik, muVk, strain_maxk)
    end

    return VEVP_Zhao2021_AD_SimplifiedState(init_internal(), init_internal())
end

function update_state!(state::VEVP_Zhao2021_AD_SimplifiedState)
    copy_state!(state.previous, state.current)
end

function revert_state!(state::VEVP_Zhao2021_AD_SimplifiedState)
    copy_state!(state.current, state.previous)
end

#################################
# Core constitutive mathematics #
#################################

function solve_local_vevp_zhao(F::Tensor{2,3,T}, dt::T_dt, mat::VEVP_Zhao2021_AD_Simplified, state::VEVP_Zhao2021_AD_SimplifiedState) where {T,T_dt}
    prm = mat
    prev = state.previous

    dt_scaled = dt * mat.vevp_dt_scale

    # 1. Kinematics
    J = det(F)
    C = tdot(F) # F' * F
    C_inv = inv(C)
    J2 = max(det(C), 1e-12)
    C_star = J2^(-1 / 3) * C

    # 2. Maximum Strain Tracking (from F*F')
    B = symmetric(F ⋅ F')
    vals = eigen(B).values
    λs = sqrt.(max.(vals, 1e-14))
    strain_maxk0 = abs(maximum(λs) / minimum(λs) - 1.0)
    strain_maxkp1 = max(T(strain_maxk0), T(prev.strain_maxk))

    # 3. Parameters and Branch Logic
    NV = prm.vevp_NV
    muVk = T(prev.muVk)
    muVN = prm.vevp_muVN
    mu1 = muVk * 2.0 / NV - muVN

    # 4. Elastic Response
    SE_dev = prm.vevp_muE * C_inv ⋅ dev(C_star)
    SE_vol = prm.vevp_kappa * J * (J - 1.0) * C_inv

    Sk_dev = SE_dev
    Cikp1 = Vector{SymmetricTensor{2,3,T}}(undef, NV)
    gamma_D_all = zeros(T, NV)

    for j in 1:NV
        tauHat_j = prm.vevp_tauHat1 + (prm.vevp_tauHatN - prm.vevp_tauHat1) / (NV - 1) * (j - 1)
        m_j = prm.vevp_m1 + (prm.vevp_mN - prm.vevp_m1) / (NV - 1) * (j - 1)
        muVi_j = (abs(muVN) < 1e-6) ? (muVk / NV) : (mu1 + (muVN - mu1) / (NV - 1) * (j - 1))
        taucut_j = prm.vevp_taucut1 + (prm.vevp_taucutN - prm.vevp_taucut1) / (NV - 1) * (j - 1)

        invCik_j = inv(prev.Cik[j])
        SV_trial_j = muVi_j * C_inv ⋅ dev(C_star ⋅ invCik_j)

        tau_trial_norm = norm(dev(F ⋅ SV_trial_j ⋅ F'))

        ramp1 = max(0.0, tau_trial_norm / tauHat_j - max(0.0, taucut_j - prm.vevp_beta * strain_maxkp1))
        gamma_D = ramp1^m_j
        gamma_factor = (tau_trial_norm > 1e-14) ? (gamma_D / tau_trial_norm) : zero(T)

        # Semi-implicit update for Cik
        Cikp1_nd = prev.Cik[j] + gamma_factor * muVi_j * dt_scaled * C_star
        det_nd = det(Cikp1_nd)
        Cikp1[j] = (max(eps(), det_nd))^(-1 / 3) * Cikp1_nd

        Sk_dev += SV_trial_j
        if j == 1
            gamma_D_all[1] = gamma_D
        end
    end

    # 5. Damage/Softening Update
    muVkp1 = muVk + prm.vevp_alpha * (prm.vevp_muVe - muVk) * gamma_D_all[1] * dt_scaled

    Sk = Sk_dev + SE_vol
    sigma = (F ⋅ Sk ⋅ F') / J
    return sigma, Cikp1, muVkp1, strain_maxkp1
end

function compute_vevp_zhao_PK1(F::Tensor{2,3,T}, dt, mat::VEVP_Zhao2021_AD_Simplified, state::VEVP_Zhao2021_AD_SimplifiedState) where T
    sigma, _, _, _ = solve_local_vevp_zhao(F, dt, mat, state)
    return det(F) * sigma ⋅ inv(F)'
end

function _assemble_element!(ke, re, states::Vector{<:AbstractMaterialState}, mp::VEVP_Zhao2021_AD_Simplified, cellvalues, alphavalues, u, dt)
    for qp in 1:getnquadpoints(cellvalues)
        state = states[qp]::VEVP_Zhao2021_AD_SimplifiedState
        α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)
        F = deformation_gradient(cellvalues, qp, u)

        P = compute_vevp_zhao_PK1(F, dt, mp, state)
        dP_dF = Tensors.gradient(F_ -> compute_vevp_zhao_PK1(F_, dt, mp, state), F)

        # Update state (Float64)
        _, Cik_v, muV_v, smax_v = solve_local_vevp_zhao(Tensors.value(F), dt, mp, state)
        state.current.Cik = Tensors.value.(Cik_v)
        state.current.muVk = Tensors.value(muV_v)
        state.current.strain_maxk = Tensors.value(smax_v)

        assemble_pk1_tangent!(ke, re, P, dP_dF, cellvalues, qp, α_dΩ)
    end
end

function _compute_stress_qp(mp::VEVP_Zhao2021_AD_Simplified, cellvalues, alphavalues, qp, u_local, state::VEVP_Zhao2021_AD_SimplifiedState, dt=0.0)
    F = deformation_gradient(cellvalues, qp, u_local)
    sigma, _, _, _ = solve_local_vevp_zhao(F, dt, mp, state)
    return sigma
end

# 2D wrapper interface for VEVP_Zhao2021_AD_Simplified
function compute_PK1_3D(mp::VEVP_Zhao2021_AD_Simplified, F::Tensor{2,3,T}, dt, state::VEVP_Zhao2021_AD_SimplifiedState) where T
    return compute_vevp_zhao_PK1(F, dt, mp, state)
end

function update_state_from_3D!(state::VEVP_Zhao2021_AD_SimplifiedState, mp::VEVP_Zhao2021_AD_Simplified, F::Tensor{2,3}, dt)
    _, Cik_v, muV_v, smax_v = solve_local_vevp_zhao(F, dt, mp, state)
    state.current.Cik = Tensors.value.(Cik_v)
    state.current.muVk = Tensors.value(muV_v)
    state.current.strain_maxk = Tensors.value(smax_v)
    return nothing
end
