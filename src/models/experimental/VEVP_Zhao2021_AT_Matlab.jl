# Experimental: Matrix-storage translation of the CAPRICCIO/Zhao set_vevp_stress_S_C.m and set_vevp_stiffness_matrix_SC_gp.m routines. The canonical `VEVP_Zhao2021_AT` in src/models/ is the SymT-based variant for new work.
using LinearAlgebra
using Tensors

####################
# Material structs #
####################

"""
    VEVP_Zhao2021_AT_MatlabStateInternal

Internal state variables for the VEVP model at a single quadrature point.
"""
mutable struct VEVP_Zhao2021_AT_MatlabStateInternal
    Cik::Vector{Matrix{Float64}}
    Cik_inv::Vector{Matrix{Float64}}
    Cikp1_nd::Vector{Matrix{Float64}}
    muVk::Float64
    strain_maxk::Float64
end

"""
    VEVP_Zhao2021_AT_MatlabState <: AbstractMaterialState

Wraps `VEVP_Zhao2021_AT_MatlabStateInternal` for the `AbstractMaterialState` interface.
Holds `current` (trial) and `previous` (converged) states.
"""
mutable struct VEVP_Zhao2021_AT_MatlabState <: AbstractMaterialState
    current::VEVP_Zhao2021_AT_MatlabStateInternal
    previous::VEVP_Zhao2021_AT_MatlabStateInternal
end

"""
    VEVP_Zhao2021_AT_MatlabStiffnessDataGP

Temporary storage for stiffness calculation data at a quadrature point.
"""
struct VEVP_Zhao2021_AT_MatlabStiffnessDataGP
    kappa::Float64
    J::Float64
    muE::Float64
    NV::Int
    C::Matrix{Float64}
    S::Matrix{Float64}
    S_dev::Matrix{Float64}
    C_star::Matrix{Float64}
    Finv::Matrix{Float64}
    FinvT::Matrix{Float64}
    muVki::Vector{Float64}
    Ci_inv::Vector{Matrix{Float64}}
    gamma_D::Vector{Float64}
    Cikp1_nd::Vector{Matrix{Float64}}
    dt::Float64
    flag_implicit::Int
end

"""
    VEVP_Zhao2021_AT_Matlab(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV;
               dt_scale=1.0)

Viscoelastic-viscoplastic material model after Zhao with an assemble-time constitutive update.
Provides an analytically derived stiffness matrix for finite-strain problems.

# Note: not interchangeable with `VEVP_Zhao2021_AD`

Uses a Matrix-storage translation of the reference routines with
explicit permission for MIT distribution in FerriteSolidMechanics.jl
(`set_vevp_stress_S_C.m` + `set_vevp_stiffness_matrix_SC_gp.m`).
This model builds the final PK2 stress from the trial `SV_trial{j}` (old `μVₖ`, old `Cᵢₖ_inv`) and pairs it with a hand-derived analytical tangent.
`VEVP_Zhao2021_AD` builds it from the converged `SVⱼ` (new `μVₖ₊₁`, new `Cᵢₖ₊₁`) and obtains the tangent by automatic differentiation.
The same `F`, `dt`, parameters, and history state therefore give different PK2 stresses from the two models.

### Parameters
`NV` must be at least 2, because branch parameters are interpolated between the branch-1 and branch-N end points.

- `μE` – Elastic branch shear modulus
- `κ` – Bulk modulus
- `NV` – Number of inelastic modules
- `β` – Yield-threshold softening coefficient
- `α` – Inelastic shear modulus evolution rate
- `μVe` – Target total inelastic shear modulus
- `μVN` – Module-N shear modulus endpoint; near zero gives uniform modules
- `τ̂₁`, `τ̂N` – Flow-resistance stress-scale endpoints
- `m₁`, `mN` – Stress exponent endpoints for modules 1 and N
- `τcut₁`, `τcutN` – Initial effective yield-threshold endpoints for modules 1 and N
- `μV` – Initial total inelastic shear modulus

### Keyword arguments
- `dt_scale` – multiplies `dt` inside the constitutive update so the time step is expressed in the unit used during parameter calibration.
  Default `1.0` leaves `dt` unscaled.
  When the calibration unit and the driver unit differ, set it to the conversion factor: parameters calibrated in seconds with a driver passing `dt` in nanoseconds give `dt_scale = 1e-9`.
  The original MD-calibrated parameter set uses that nanosecond convention.
"""
struct VEVP_Zhao2021_AT_Matlab <: AbstractMaterial
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
function VEVP_Zhao2021_AT_Matlab(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV;
                    dt_scale=1.0)
    NV >= 2 || throw(ArgumentError("VEVP_Zhao2021_AT_Matlab: NV must be >= 2"))
    return VEVP_Zhao2021_AT_Matlab(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV,
                      dt_scale)
end

##########################
# AbstractMaterial hooks #
##########################

is_linear(::VEVP_Zhao2021_AT_Matlab) = false

function create_state(mat::VEVP_Zhao2021_AT_Matlab)
    NV = mat.vevp_NV
    muV = mat.vevp_muV

    function init_internal()
        Cik = [Matrix{Float64}(I, 3, 3) for _ in 1:NV]
        Cik_inv = [Matrix{Float64}(I, 3, 3) for _ in 1:NV]
        Cikp1_nd = [Matrix{Float64}(I, 3, 3) for _ in 1:NV]
        muVk = muV
        strain_maxk = 0.0
        return VEVP_Zhao2021_AT_MatlabStateInternal(Cik, Cik_inv, Cikp1_nd, muVk, strain_maxk)
    end

    return VEVP_Zhao2021_AT_MatlabState(init_internal(), init_internal())
end

function update_state!(state::VEVP_Zhao2021_AT_MatlabState)
    # Commit the current state to previous for the next load step
    copy_state!(state.previous, state.current)
end

function revert_state!(state::VEVP_Zhao2021_AT_MatlabState)
    copy_state!(state.current, state.previous)
end

#######################
# Ferrite integration #
#######################

"""
    _assemble_element!(ke, re, states, mp::VEVP_Zhao2021_AT_Matlab, cellvalues, alphavalues, u, dt)

Assembles the stiffness matrix `ke` and residual vector `re` for a VEVP element.
`states` is a vector of `VEVP_Zhao2021_AT_MatlabState`, one per quadrature point.
"""
function _assemble_element!(ke, re, states::Vector{<:AbstractMaterialState}, mp::VEVP_Zhao2021_AT_Matlab, cellvalues, alphavalues, u, dt)
    n_basefuncs = getnbasefunctions(cellvalues)

    # Unit scaling: apply the user-supplied `dt_scale` kwarg (default 1.0).
    # Note: scaling is also applied below in `_compute_stress_qp`.
    dt_scaled = dt * mp.vevp_dt_scale

    for qp in 1:getnquadpoints(cellvalues)
        state = states[qp]
        state_old = state.previous
        α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)

        # Deformation gradient
        F = deformation_gradient(cellvalues, qp, u)
        F_mat = Matrix(F)
        Finv_mat = inv(F_mat)

        # Constitutive update (Semi-implicit = 1)
        new_internal_state, Sk, Sk_dev, gamma_D = set_vevp_zhao_stress_S_C(F_mat, Finv_mat, dt_scaled, mp, state_old, 1)

        # Store for next iteration
        state.current = new_internal_state

        # Internal Force Contribution
        P_mat = F_mat * Sk
        P = Tensor{2,3}(P_mat)

        if re !== nothing
            for i in 1:n_basefuncs
                ∇δui = shape_gradient(cellvalues, qp, i)
                re[i] += (∇δui ⊡ P) * α_dΩ
            end
        end

        # Stiffness Matrix Contribution
        stiff_data = prepare_vevp_zhao_stiffness_data_gp(Sk_dev, Sk, gamma_D, new_internal_state, F_mat, Finv_mat, mp, state_old, dt_scaled, 1)

        n_nodes = n_basefuncs ÷ 3
        # Precompute scalar shape function gradients
        DNs = [Vector(shape_gradient(cellvalues, qp, (n - 1) * 3 + 1)[1, :]) for n in 1:n_nodes]

        for a in 1:n_nodes
            DN_a = DNs[a]
            for b in 1:n_nodes
                DN_b = DNs[b]
                K_ab = calc_vevp_zhao_stiffness_block_jk(stiff_data, DN_a, DN_b)
                K_ab .*= α_dΩ

                r_idx, c_idx = (a - 1) * 3, (b - 1) * 3
                for r in 1:3, c in 1:3
                    ke[r_idx+r, c_idx+c] += K_ab[r, c]
                end
            end
        end
    end
end

"""
    _compute_stress_qp(mp::VEVP_Zhao2021_AT_Matlab, cellvalues, alphavalues, qp, u_local, state, dt)

Computes the Cauchy stress at a quadrature point for output.
"""
function _compute_stress_qp(mp::VEVP_Zhao2021_AT_Matlab, cellvalues, alphavalues, qp, u_local, state::VEVP_Zhao2021_AT_MatlabState, dt=0.0)
    state_old = state.current

    F = deformation_gradient(cellvalues, qp, u_local)
    F_mat = Matrix(F)
    Finv_mat = inv(F_mat)
    J = det(F)

    _, Sk, _, _ = set_vevp_zhao_stress_S_C(F_mat, Finv_mat, 0.0, mp, state_old, 1)

    σ_mat = (F_mat * Sk * F_mat') / J
    return Tensor{2,3}(σ_mat)
end

#################################
# Core constitutive mathematics #
#################################

function set_vevp_zhao_stress_S_C(F, F_inv, dt, prm_vevp::VEVP_Zhao2021_AT_Matlab, state_eg::VEVP_Zhao2021_AT_MatlabStateInternal, flag_implicit::Int)
    # Internal variables from old state
    Cik = state_eg.Cik
    Cik_inv = state_eg.Cik_inv
    muVk = state_eg.muVk
    strain_maxk = state_eg.strain_maxk

    # Parameters
    beta = prm_vevp.vevp_beta
    muE = prm_vevp.vevp_muE
    kappa = prm_vevp.vevp_kappa
    NV = prm_vevp.vevp_NV
    tauHat1 = prm_vevp.vevp_tauHat1
    tauHatN = prm_vevp.vevp_tauHatN
    m1 = prm_vevp.vevp_m1
    mN = prm_vevp.vevp_mN
    taucut1 = prm_vevp.vevp_taucut1
    taucutN = prm_vevp.vevp_taucutN
    muVe = prm_vevp.vevp_muVe
    alpha = prm_vevp.vevp_alpha
    muVN = prm_vevp.vevp_muVN
    mu1 = muVk * 2 / NV - muVN

    # Eigenvalue logic
    eigs = sqrt.(eigvals(F * F'))
    strain_maxk0 = abs(maximum(eigs) / minimum(eigs) - 1)
    strain_maxkp1 = max(strain_maxk0, strain_maxk)

    tauHat = zeros(NV)
    m = zeros(NV)
    muVi = zeros(NV)
    taucut = zeros(NV)
    SV_trial = Vector{Matrix{Float64}}(undef, NV)
    SV = Vector{Matrix{Float64}}(undef, NV)
    gamma_D = zeros(NV)
    gamma_factor = zeros(NV)
    Cikp1 = Vector{Matrix{Float64}}(undef, NV)
    Cikp1_inv = Vector{Matrix{Float64}}(undef, NV)
    Cikp1_nd = Vector{Matrix{Float64}}(undef, NV)

    Jk = det(F)
    Jk_13 = Jk^(-1 / 3)
    Jk_23 = Jk_13^2

    C = F' * F
    C_star = C * Jk_23
    C_inv = F_inv * F_inv'

    SE_dev = muE * C_inv * get_deviate_matrix_zhao(C_star)
    SE_vol = kappa * Jk * (Jk - 1) * C_inv

    for j = 1:NV
        tauHat[j] = tauHat1 + (tauHatN - tauHat1) / (NV - 1) * (j - 1)
        m[j] = m1 + (mN - m1) / (NV - 1) * (j - 1)
        if abs(muVN) < 1e-6
            muVi[j] = muVk / NV
        else
            muVi[j] = mu1 + (muVN - mu1) / (NV - 1) * (j - 1)
        end
        taucut[j] = taucut1 + (taucutN - taucut1) / (NV - 1) * (j - 1)

        SV_trial[j] = muVi[j] * C_inv * get_deviate_matrix_zhao(C_star * Cik_inv[j])
        tau_trial_norm = get_matrix_norm_zhao(F * SV_trial[j] * F')

        if tau_trial_norm == 0
            gamma_factor[j] = 0
        else
            gamma_factor[j] = 1 / tau_trial_norm
        end

        gamma_D[j] = (get_ramp_value_zhao(tau_trial_norm / tauHat[j] - get_ramp_value_zhao(taucut[j] - beta * strain_maxkp1)))^m[j]

        if flag_implicit == 0
            # Explicit Euler
            dot_Cikj = gamma_factor[j] * gamma_D[j] * muVi[j] * C * SV_trial[j] * Cik[j]
            Cikp1[j] = Cik[j] + dot_Cikj * dt
            Cikp1_inv[j] = inv(Cikp1[j])
            SV[j] = SV_trial[j]
            Cikp1_nd[j] = Cikp1[j]
        elseif flag_implicit == 1
            # Semi-implicit
            Cikp1_nd[j] = Cik[j] + gamma_factor[j] * gamma_D[j] * muVi[j] * dt * C_star
            J_Cikp1 = det(Cikp1_nd[j])
            Cikp1[j] = J_Cikp1^(-1 / 3) * Cikp1_nd[j]
            Cikp1_inv[j] = inv(Cikp1[j])
            SV[j] = muVi[j] * C_inv * get_deviate_matrix_zhao(C_star * Cikp1_inv[j])
        end
    end

    muVkp1 = muVk + alpha * (muVe - muVk) * gamma_D[1] * dt
    gamma_D = gamma_D .* gamma_factor

    Sk_dev = copy(SE_dev)
    for j = 1:NV
        Sk_dev += SV_trial[j]
    end
    Sk = Sk_dev + SE_vol

    state_vevp_temp_eg = VEVP_Zhao2021_AT_MatlabStateInternal(Cikp1, Cikp1_inv, Cikp1_nd, muVkp1, strain_maxkp1)

    return state_vevp_temp_eg, Sk, Sk_dev, gamma_D
end

function prepare_vevp_zhao_stiffness_data_gp(S_dev, S, gamma_D, state_vevp_temp_eg::VEVP_Zhao2021_AT_MatlabStateInternal, F, Finv, prm_vevp::VEVP_Zhao2021_AT_Matlab, state_eg::VEVP_Zhao2021_AT_MatlabStateInternal, dt, flag_implicit::Int)
    kappa = prm_vevp.vevp_kappa
    NV = prm_vevp.vevp_NV
    muE = prm_vevp.vevp_muE
    muVN = prm_vevp.vevp_muVN

    Cikp1_inv = state_vevp_temp_eg.Cik_inv
    Cikp1_nd = state_vevp_temp_eg.Cikp1_nd
    Cik_inv = state_eg.Cik_inv
    muVk = state_eg.muVk

    muVki = zeros(NV)
    if abs(muVN) < 1e-6
        muVki .= muVk / NV
    else
        muV1 = muVk * 2 / NV - muVN
        for j = 1:NV
            muVki[j] = muV1 + (muVN - muV1) / (NV - 1) * (j - 1)
        end
    end

    C = F' * F
    J = det(F)
    C_star = C * J^(-2 / 3)

    Ci_inv = (flag_implicit == 0) ? Cik_inv : Cikp1_inv

    return VEVP_Zhao2021_AT_MatlabStiffnessDataGP(kappa, J, muE, NV, Matrix(C), Matrix(S), Matrix(S_dev), Matrix(C_star), Matrix(Finv), Matrix(Finv'), muVki, Ci_inv, gamma_D, Cikp1_nd, dt, flag_implicit)
end

function calc_vevp_zhao_stiffness_block_jk(data::VEVP_Zhao2021_AT_MatlabStiffnessDataGP, DN_j_vec, DN_k_vec)
    DN_j = Array(DN_j_vec)
    DN_k = Array(DN_k_vec)

    # Elastic part
    CS_DNj = (data.C * data.S) * DN_j
    CSdev_DNj = (data.C * data.S_dev) * DN_j
    Cstar_DNj = data.C_star * DN_j
    Cstar_DNk = data.C_star * DN_k

    Kgpe_jk_pull_back = -DN_k * CS_DNj' - (2.0 / 3.0) * CSdev_DNj * DN_k' + (data.kappa * data.J * (2 * data.J - 1)) * (DN_j * DN_k') +
                        data.muE * (DN_k * Cstar_DNj') - (2.0 / 3.0) * data.muE * (DN_j * Cstar_DNk') + (data.muE * dot(DN_k, DN_j)) * data.C_star

    # Inelastic part
    Kgpi_pull_back = zeros(3, 3)
    for j = 1:data.NV
        M = data.C_star * data.Ci_inv[j]
        Kgpi_pull_back .+= data.muVki[j] * (DN_k * (M * DN_j)' - (2.0 / 3.0) * DN_j * (M * DN_k)' + dot(DN_k, data.Ci_inv[j] * DN_j) * data.C_star)
    end

    # Implicit term
    Kgpi_im_pull_back = zeros(3, 3)
    if data.flag_implicit == 1
        for j = 1:data.NV
            Ce = data.C_star * data.Ci_inv[j]
            Ce_DNk = Ce * DN_k
            Ce_DNj = Ce * DN_j
            Ce2_DNk = (Ce * Ce) * DN_k
            Ce2_DNj = (Ce * Ce) * DN_j
            M_dev = Ce - tr(Ce) / 3.0 * I
            factor = -data.muVki[j]^2 * data.gamma_D[j] * data.dt * det(data.Cikp1_nd[j])^(-1 / 3)

            t1 = Ce_DNk * Ce_DNj'
            t2 = -(2.0 / 3.0) * DN_j * Ce2_DNk'
            t3 = -(2.0 / 3.0) * Ce2_DNj * DN_k'
            t4 = (2.0 / 9.0 * tr(Ce * Ce)) * (DN_j * DN_k')
            t5 = -(2.0 / 3.0) * (M_dev * DN_j) * (M_dev * DN_k)'
            t6 = (Ce * data.C_star) * dot(DN_j, data.Ci_inv[j] * DN_k)

            Kgpi_im_pull_back .+= factor * (t1 + t2 + t3 + t4 + t5 + t6)
        end
    end

    return data.FinvT * (Kgpi_im_pull_back + Kgpi_pull_back + Kgpe_jk_pull_back) * data.Finv
end

####################
# Helper utilities #
####################

function get_deviate_matrix_zhao(A)
    return A - tr(A) / 3.0 * I
end

function get_matrix_norm_zhao(T)
    dev_T = get_deviate_matrix_zhao(T)
    return sqrt(tr(dev_T * dev_T'))
end

function get_ramp_value_zhao(x)
    return 0.5 * (x + abs(x))
end

function get_trace_matrix_zhao(A)
    return tr(A)
end
