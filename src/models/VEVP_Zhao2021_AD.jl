# --- 1. Material struct ---

"""
    VEVP_Zhao2021_AD(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV;
               dt_scale=1.0)

Viscoelastic-viscoplastic material model after Zhao with automatic differentiation for the algorithmic tangent.
Uses multiple inelastic modules and a semi-implicit internal-variable update.

# Note: not interchangeable with `VEVP_Zhao2021_AT`

Uses the same parameterization and semi-implicit update structure as `VEVP_Zhao2021_AT`, but builds the final PK2 stress from a different basis.
This struct uses the converged `SVⱼ` (new `μVₖ₊₁`, new `Cᵢₖ₊₁`), so the returned stress is colocated with the AD tangent.
`VEVP_Zhao2021_AT` uses the trial `SV_trial{j}` (old `μVₖ`, old `Cᵢₖ_inv`), matching the CAPRICCIO/Zhao analytical-tangent path.
The AD path also applies small determinant and principal stretch floors.
The same `F`, `dt`, parameters, and history state therefore give different PK2 stresses from the two models.

# References

- W. Zhao, M. Ries, P. Steinmann, S. Pfaller.
  *A viscoelastic-viscoplastic constitutive model for glassy polymers
  informed by molecular dynamics simulations.*
  International Journal of Solids and Structures **226–227** (2021) 111071.
  <https://doi.org/10.1016/j.ijsolstr.2021.111071>
- W. Zhao, R. Xiao, S. Pfaller, P. Steinmann.
  *Modeling strain hardening in glassy polymers based on the
  microscopic mechanisms revealed by molecular dynamic simulations.*
  Journal of the Mechanics and Physics of Solids **206** (2026) 106384.
  <https://doi.org/10.1016/j.jmps.2025.106384>

The implementation-level reference for this code path is part of the
CAPRICCIO FE–MD coupling tool. This file is a translated/adapted Julia
implementation with package-specific changes and explicit permission for
MIT distribution in FerriteSolidMechanics.jl:
Pfaller, S., Ries, M., Zhao, W., Bauer, C., Weber, F., & Laubert, L.
*CAPRICCIO - Tool to run concurrent Finite Element-Molecular Dynamics
Simulations* (Version 2.0.1), Zenodo (2024).
<https://doi.org/10.5281/zenodo.12606758>

### Parameters
`NV` must be at least 2 as branch parameters are interpolated between the branch-1 and branch-N end points.

- `μE` – Elastic branch shear modulus
- `κ` – Bulk modulus
- `NV` – Number of inelastic modules (requires NV ≥ 2; module parameters are interpolated between the module-1 and module-NV endpoints)
- `β` – Yield-threshold softening coefficient; larger values drop the threshold sooner
- `α` – Inelastic shear modulus evolution rate; higher values approach μVe faster
- `μVe` – Target total inelastic shear modulus approached by the μV evolution
- `μVN` – Module-NV shear modulus endpoint; near zero gives uniform modules
- `τ̂₁`, `τ̂N` – Flow-resistance stress-scale endpoints in the power-flow law; larger values reduce the viscoplastic flow rate
- `m₁`, `mN` – Stress exponents in the power-flow law; larger values increase the nonlinearity of the viscoplastic rate
- `τcut₁`, `τcutN` – Initial yield thresholds for the modules; subtracted from the trial-stress ratio before the power-law ramp
- `μV` – Initial total inelastic shear modulus

### Keyword arguments
- `dt_scale` – multiplies `dt` inside the constitutive update so the time step is expressed in the unit used during parameter calibration.
  Default `1.0` leaves `dt` unscaled.
  When the calibration unit and the driver unit differ, set it to the conversion factor: parameters calibrated in seconds with a driver passing `dt` in nanoseconds give `dt_scale = 1e-9`.
"""
struct VEVP_Zhao2021_AD <: AbstractMaterial
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

# Outer constructor exposing the `dt_scale` kwarg while preserving the
# 14-argument positional API used by existing code and examples.
function VEVP_Zhao2021_AD(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV;
                    dt_scale=1.0)
    NV >= 2 || throw(ArgumentError("VEVP_Zhao2021_AD: NV must be >= 2"))
    return VEVP_Zhao2021_AD(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV,
                      dt_scale)
end

# --- 2. State struct ---

mutable struct VEVP_Zhao2021_ADStateInternal
    Cik::Vector{SymmetricTensor{2,3,Float64,6}}
    muVk::Float64
    strain_maxk::Float64
end

mutable struct VEVP_Zhao2021_ADState <: AbstractMaterialState
    current::VEVP_Zhao2021_ADStateInternal
    previous::VEVP_Zhao2021_ADStateInternal
end

# --- 3. API hooks ---

is_linear(::VEVP_Zhao2021_AD) = false

function create_state(mat::VEVP_Zhao2021_AD)
    NV = mat.vevp_NV
    muV = mat.vevp_muV

    function init_internal()
        Cik = [one(SymmetricTensor{2,3}) for _ in 1:NV]
        muVk = muV
        strain_maxk = 0.0
        return VEVP_Zhao2021_ADStateInternal(Cik, muVk, strain_maxk)
    end

    return VEVP_Zhao2021_ADState(init_internal(), init_internal())
end

function update_state!(state::VEVP_Zhao2021_ADState)
    copy_state!(state.previous, state.current)
end

function revert_state!(state::VEVP_Zhao2021_ADState)
    copy_state!(state.current, state.previous)
end

# --- 4. AD tangent wrapper ---

function compute_vevp_zhao_exact_PK1(F::Tensor{2,3,T}, dt, mat::VEVP_Zhao2021_AD, state::VEVP_Zhao2021_ADState) where T
    sigma, _, _, _ = solve_local_vevp_zhao_exact(F, dt, mat, state)
    return det(F) * sigma ⋅ inv(F)'
end

# --- 5. Main driver ---

function solve_local_vevp_zhao_exact(F::Tensor{2,3,T}, dt::T_dt, mat::VEVP_Zhao2021_AD, state::VEVP_Zhao2021_ADState) where {T,T_dt}
    prm = mat
    prev = state.previous

    # Apply the user-supplied `dt_scale` kwarg (default 1.0; pass 1e-9 for
    # MD driver loops that provide `dt` in nanoseconds while the calibration
    # used seconds).
    dt_scaled = dt * mat.vevp_dt_scale

    # 1. Kinematics
    J = det(F)
    C = tdot(F)
    C_inv = inv(C)
    J2 = max(det(C), 1e-12)
    C_star = J2^(-1 / 3) * C

    # 2. Maximum Strain Tracking
    # eigen on the value only: skips differentiating eigen and omits the
    # strain ratio sensitivity for β ≠ 0 (exact for β = 0), like VEVP_Zhao2021_AT
    Fv = Tensors.value(F)
    Bv = symmetric(Fv ⋅ Fv')
    vals = eigen(Bv).values
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

    Cikp1 = Vector{SymmetricTensor{2,3,T}}(undef, NV)
    gamma_D_all = zeros(T, NV)

    # First pass: Compute the updated internal variables based on trial stresses
    for j in 1:NV
        tauHat_j = prm.vevp_tauHat1 + (prm.vevp_tauHatN - prm.vevp_tauHat1) / (NV - 1) * (j - 1)
        m_j = prm.vevp_m1 + (prm.vevp_mN - prm.vevp_m1) / (NV - 1) * (j - 1)
        muVi_j = (abs(muVN) < 1e-6) ? (muVk / NV) : (mu1 + (muVN - mu1) / (NV - 1) * (j - 1))
        taucut_j = prm.vevp_taucut1 + (prm.vevp_taucutN - prm.vevp_taucut1) / (NV - 1) * (j - 1)

        invCik_j = inv(prev.Cik[j])
        SV_trial_j = muVi_j * C_inv ⋅ dev(C_star ⋅ invCik_j)

        # value-branched sqrt: plain norm() NaNs the AD tangent at zero deviator (F ∝ I)
        tau_dev = dev(F ⋅ SV_trial_j ⋅ F')
        tau2 = tau_dev ⊡ tau_dev
        tau_trial_norm = Tensors.value(tau2) > 1e-28 ? sqrt(tau2) : zero(tau2)

        ramp1 = max(0.0, tau_trial_norm / tauHat_j - max(0.0, taucut_j - prm.vevp_beta * strain_maxkp1))
        gamma_D = ramp1^m_j
        gamma_factor = (tau_trial_norm > 1e-14) ? (gamma_D / tau_trial_norm) : zero(T)

        # Semi-implicit update for Cik
        Cikp1_nd = prev.Cik[j] + gamma_factor * muVi_j * dt_scaled * C_star
        det_nd = det(Cikp1_nd)
        Cikp1[j] = (max(eps(), det_nd))^(-1 / 3) * Cikp1_nd

        if j == 1
            gamma_D_all[1] = gamma_D
        end
    end

    # 5. Damage/Softening Update
    muVkp1 = muVk + prm.vevp_alpha * (prm.vevp_muVe - muVk) * gamma_D_all[1] * dt_scaled

    # 6. Final Stress Computation based on UPDATED state (Yields full algorithmic tangent via AD)
    mu1_p1 = muVkp1 * 2.0 / NV - muVN
    Sk_dev = SE_dev
    for j in 1:NV
        muVi_j = (abs(muVN) < 1e-6) ? (muVkp1 / NV) : (mu1_p1 + (muVN - mu1_p1) / (NV - 1) * (j - 1))
        invCikp1_j = inv(Cikp1[j])
        SV_j = muVi_j * C_inv ⋅ dev(C_star ⋅ invCikp1_j)
        Sk_dev += SV_j
    end

    Sk = Sk_dev + SE_vol
    sigma = (F ⋅ Sk ⋅ F') / J
    return sigma, Cikp1, muVkp1, strain_maxkp1
end

kinematics(::VEVP_Zhao2021_AD) = FiniteStrain()

function material_response(mp::VEVP_Zhao2021_AD, F::Tensor{2,3}, state::VEVP_Zhao2021_ADState, dt, cache=nothing)
    sigma, Cik_v, muV_v, smax_v = solve_local_vevp_zhao_exact(F, dt, mp, state)
    P = det(F) * sigma ⋅ inv(F)'
    dP_dF = Tensors.gradient(F_ -> compute_vevp_zhao_exact_PK1(F_, dt, mp, state), F)
    return P, dP_dF, VEVP_Zhao2021_ADStateInternal(Cik_v, muV_v, smax_v)
end

function material_stress(mp::VEVP_Zhao2021_AD, F::Tensor{2,3}, state::VEVP_Zhao2021_ADState, dt, cache=nothing)
    current = VEVP_Zhao2021_ADState(state.current, state.current)
    sigma, _, _, _ = solve_local_vevp_zhao_exact(F, 0.0, mp, current)  # pass dt=0.0 to not advance history in postprocessing (!)
    return sigma
end

# 2D wrapper interface for VEVP_Zhao2021_AD
function compute_PK1_3D(mp::VEVP_Zhao2021_AD, F::Tensor{2,3,T}, dt, state::VEVP_Zhao2021_ADState) where T
    return compute_vevp_zhao_exact_PK1(F, dt, mp, state)
end

function update_state_from_3D!(state::VEVP_Zhao2021_ADState, mp::VEVP_Zhao2021_AD, F::Tensor{2,3}, dt)
    _, Cik_v, muV_v, smax_v = solve_local_vevp_zhao_exact(F, dt, mp, state)
    state.current.Cik = Tensors.value.(Cik_v)
    state.current.muVk = Tensors.value(muV_v)
    state.current.strain_maxk = Tensors.value(smax_v)
    return nothing
end
