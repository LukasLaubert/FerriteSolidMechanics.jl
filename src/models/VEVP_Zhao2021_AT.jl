# --- 1. Material struct ---

"""
    VEVP_Zhao2021_AT(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV;
               dt_scale=1.0)

Viscoelastic-viscoplastic material model after Zhao with an assemble-time constitutive update.
Provides an analytically derived stiffness matrix for finite-strain problems.

# Note: not interchangeable with `VEVP_Zhao2021_AD`

Uses the same parameterization and semi-implicit update structure as `VEVP_Zhao2021_AD`, but builds the final PK2 stress from a different basis.
This struct uses the trial `SV_trial{j}` (old `μVₖ`, old `Cᵢₖ_inv`) with a hand-derived analytical tangent.
`VEVP_Zhao2021_AD` uses the converged `SVⱼ` (new `μVₖ₊₁`, new `Cᵢₖ₊₁`) and obtains the tangent by AD.
The same `F`, `dt`, parameters, and history state therefore give different PK2 stresses from the two models.

An archived Matrix-storage translation of the CAPRICCIO/Zhao routines is preserved as `FerriteSolidMechanics.Experimental.VEVP_Zhao2021_AT_Matlab`.

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

# References

- W. Zhao, M. Ries, P. Steinmann, S. Pfaller.
  *A viscoelastic-viscoplastic constitutive model for glassy polymers informed by molecular dynamics simulations.*
  International Journal of Solids and Structures **226-227** (2021) 111071.
  <https://doi.org/10.1016/j.ijsolstr.2021.111071>
- W. Zhao, R. Xiao, S. Pfaller, P. Steinmann.
  *Modeling strain hardening in glassy polymers based on the microscopic mechanisms revealed by molecular dynamic simulations.*
  Journal of the Mechanics and Physics of Solids **206** (2026) 106384.
  <https://doi.org/10.1016/j.jmps.2025.106384>

The implementation-level reference for this code path is part of the CAPRICCIO FE–MD coupling tool.
This file is a translated/adapted Julia implementation with package-specific changes and explicit permission for MIT distribution in FerriteSolidMechanics.jl:
Pfaller, S., Ries, M., Zhao, W., Bauer, C., Weber, F., & Laubert, L.
*CAPRICCIO - Tool to run concurrent Finite Element-Molecular Dynamics Simulations* (Version 2.0.1), Zenodo (2024).
<https://doi.org/10.5281/zenodo.12606758>
"""
struct VEVP_Zhao2021_AT <: AbstractMaterial
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
function VEVP_Zhao2021_AT(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV;
    dt_scale=1.0)
    NV >= 2 || throw(ArgumentError("VEVP_Zhao2021_AT: NV must be >= 2"))
    return VEVP_Zhao2021_AT(μE, κ, NV, β, α, μVe, μVN, τ̂₁, τ̂N, m₁, mN, τcut₁, τcutN, μV,
        dt_scale)
end

# --- 2. State struct ---

# Shorthand for the per-QP symmetric 3×3 storage used throughout this
# model (Tensors.jl's 6-component symmetric representation).
const SymT = SymmetricTensor{2,3,Float64,6}

"""
    VEVP_Zhao2021_ATStateInternal

Internal state variables for the VEVP model at a single quadrature point.
The per-branch right Cauchy–Green tensors, their inverses, and the pre-determinant-rescale intermediate are stored as `SymmetricTensor{2,3,Float64,6}`, Tensors.jl's 6-component symmetric storage.
That representation costs `(18·NV + 2) × 2` doubles per quadrature point, against `(27·NV + 2) × 2` doubles for the equivalent `Matrix{Float64}` storage.
It also makes the per-branch arithmetic in the forward solve faster.
"""
mutable struct VEVP_Zhao2021_ATStateInternal
    Cik::Vector{SymT}
    Cik_inv::Vector{SymT}
    Cikp1_nd::Vector{SymT}
    muVk::Float64
    strain_maxk::Float64
end

"""
    VEVP_Zhao2021_ATState <: AbstractMaterialState

Wraps `VEVP_Zhao2021_ATStateInternal` for the `AbstractMaterialState` interface.
Holds `current` (trial) and `previous` (converged) states.
"""
mutable struct VEVP_Zhao2021_ATState <: AbstractMaterialState
    current::VEVP_Zhao2021_ATStateInternal
    previous::VEVP_Zhao2021_ATStateInternal
end

"""
    VEVP_Zhao2021_ATStiffnessDataGP

Temporary storage for stiffness calculation data at a quadrature point.
"""
struct VEVP_Zhao2021_ATStiffnessDataGP
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

# --- 3. API hooks ---

is_linear(::VEVP_Zhao2021_AT) = false

function create_state(mat::VEVP_Zhao2021_AT)
    NV = mat.vevp_NV
    muV = mat.vevp_muV

    function init_internal()
        I3 = one(SymT)
        Cik = [I3 for _ in 1:NV]
        Cik_inv = [I3 for _ in 1:NV]
        Cikp1_nd = [I3 for _ in 1:NV]
        muVk = muV
        strain_maxk = 0.0
        return VEVP_Zhao2021_ATStateInternal(Cik, Cik_inv, Cikp1_nd, muVk, strain_maxk)
    end

    return VEVP_Zhao2021_ATState(init_internal(), init_internal())
end

function update_state!(state::VEVP_Zhao2021_ATState)
    # Commit the current state to previous for the next load step
    copy_state!(state.previous, state.current)
end

function revert_state!(state::VEVP_Zhao2021_ATState)
    copy_state!(state.current, state.previous)
end

# --- 4. Stress helpers ---

# Round-trip through Matrix + (A+Aᵀ)/2 to guarantee a perfectly symmetric result before assigning to a SymT slot (Tensors.jl can otherwise leave 1-ULP off-diagonal asymmetry).
@inline function _symmetrize_to_SymT(A::AbstractMatrix)
    B = (A + A') ./ 2
    return SymT(B)
end

# Deviatoric operator: A - tr(A)/3 ⋅ I.
function get_deviate_zhao(A::SymT)
    return A - tr(A) / 3.0 * one(SymT)
end
function get_deviate_zhao(A::Tensor{2,3,Float64,9})
    return A - tr(A) / 3.0 * one(Tensor{2,3,Float64,9})
end

# Frobenius norm of the deviatoric: ‖dev(T)‖_F = sqrt(tr(dev(T) ⋅ dev(T)ᵀ)).
function get_matrix_norm_zhao(T)
    d = get_deviate_zhao(T)
    return sqrt(tr(d ⋅ d'))
end

get_ramp_value_zhao(x) = 0.5 * (x + abs(x))
get_trace_matrix_zhao(A) = tr(A)

# --- 5. Main driver ---

function set_vevp_zhao_stress_S_C(F::Tensor{2,3,Float64,9}, F_inv::Tensor{2,3,Float64,9},
    dt, prm_vevp::VEVP_Zhao2021_AT,
    state_eg::VEVP_Zhao2021_ATStateInternal, flag_implicit::Int)
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

    # Eigenvalue logic. `F ⋅ F'` is mathematically symmetric but Tensors.jl
    # returns a general `Tensor{2,3,9}`; use the general-matrix
    # `eigvals` (3×3 is tiny) to avoid a 1-ULP off-diagonal asymmetry
    # that would trip `SymmetricMatrix`'s strict constructor.
    F_Ft = F ⋅ F'
    eigs = sqrt.(eigvals(Matrix(F_Ft)))
    strain_maxk0 = abs(maximum(eigs) / minimum(eigs) - 1)
    strain_maxkp1 = max(strain_maxk0, strain_maxk)

    tauHat = zeros(NV)
    m = zeros(NV)
    muVi = zeros(NV)
    taucut = zeros(NV)
    # SV_trial is read downstream (summed into Sk_dev and the matrix
    # norm); keep it in SymT.
    SV_trial = Vector{SymT}(undef, NV)
    # SV (implicit-step stress) is computed by the original CAPRICCIO/Zhao
    # routine but never read downstream here; commented out to drop the dead
    # per-branch work while keeping the port structure visible.
    # SV = Vector{Tensor{2,3,Float64,9}}(undef, NV)
    gamma_D = zeros(NV)
    gamma_factor = zeros(NV)
    Cikp1 = Vector{SymT}(undef, NV)
    Cikp1_inv = Vector{SymT}(undef, NV)
    Cikp1_nd = Vector{SymT}(undef, NV)

    Jk = det(F)
    Jk_13 = Jk^(-1 / 3)
    Jk_23 = Jk_13^2

    # C, C_star, C_inv, SE_dev, SE_vol are routed through Matrix and recast to SymT via _symmetrize_to_SymT so per-branch arithmetic on the 6-component symmetric storage stays bit-symmetric (avoids InexactError from 1-ULP off-diagonal asymmetry in SymT ⋅ SymT promotions).
    C = F' ⋅ F
    C_star = SymT(C * Jk_23)
    C_inv = _symmetrize_to_SymT(Matrix(F_inv) * Matrix(F_inv)')

    C_inv_mat = Matrix(C_inv)
    C_star_mat = Matrix(C_star)
    # Note: deviatoric is written as `A - tr(A)/3 * I` (not `A .- tr(A)/3 .* I`):
    # `.*` (broadcast multiply) has no method for `UniformScaling{Bool}` (`I`),
    # but `Matrix - UniformScaling{Float64}` is defined and returns `Matrix`.
    C_star_dev_mat = C_star_mat - tr(C_star_mat) / 3.0 * I

    SE_dev = _symmetrize_to_SymT(muE * C_inv_mat * C_star_dev_mat)
    SE_vol = _symmetrize_to_SymT(kappa * Jk * (Jk - 1) * C_inv_mat)

    # Pre-allocate 3×3 buffers reused across branches to avoid per-branch allocation.
    prod_buf = zeros(3, 3)
    prod_sym_buf = zeros(3, 3)
    C_dev_buf = zeros(3, 3)
    SV_trial_buf = zeros(3, 3)

    for j = 1:NV
        tauHat[j] = tauHat1 + (tauHatN - tauHat1) / (NV - 1) * (j - 1)
        m[j] = m1 + (mN - m1) / (NV - 1) * (j - 1)
        if abs(muVN) < 1e-6
            muVi[j] = muVk / NV
        else
            muVi[j] = mu1 + (muVN - mu1) / (NV - 1) * (j - 1)
        end
        taucut[j] = taucut1 + (taucutN - taucut1) / (NV - 1) * (j - 1)

        # Compute SV_trial in matrix arithmetic and recast to SymT; the intermediate (prod_mat+prod_mat')/2 symmetrization is a deliberate 1-ULP change vs the CAPRICCIO/Zhao Matlab port (which feeds the unsymmetrized product to dev) and stays within the roundoff floor. The in-place mul!/.= path reuses the preallocated buffers; the deviatoric step subtracts tr/3 from the diagonal in a tight @inbounds loop.
        mul!(prod_buf, C_star_mat, Matrix(Cik_inv[j]))
        prod_sym_buf .= prod_buf
        prod_sym_buf .+= prod_buf'
        prod_sym_buf ./= 2
        scalar_dev = tr(prod_sym_buf) / 3.0
        C_dev_buf .= prod_sym_buf
        @inbounds for i in 1:3
            C_dev_buf[i, i] -= scalar_dev
        end
        SV_trial_buf .= muVi[j] * C_inv_mat * C_dev_buf
        SV_trial[j] = _symmetrize_to_SymT(SV_trial_buf)
        # `F ⋅ SV_trial[j] ⋅ F'` widens to a general `Tensor{2,3,9}`;
        # `get_matrix_norm_zhao` accepts both `SymT` and general
        # `Tensor{2,3,9}`.
        tau_trial_norm = get_matrix_norm_zhao(F ⋅ SV_trial[j] ⋅ F')

        if tau_trial_norm == 0
            gamma_factor[j] = 0
        else
            gamma_factor[j] = 1 / tau_trial_norm
        end

        gamma_D[j] = (get_ramp_value_zhao(tau_trial_norm / tauHat[j] - get_ramp_value_zhao(taucut[j] - beta * strain_maxkp1)))^m[j]

        if flag_implicit == 0
            # Explicit Euler update
            tmp = SymT(C ⋅ SV_trial[j] ⋅ Cik[j])
            dot_Cikj = SymT(gamma_factor[j] * gamma_D[j] * muVi[j] * tmp)
            Cikp1[j] = SymT(Cik[j] + dot_Cikj * dt)
            Cikp1_inv[j] = inv(Cikp1[j])
            # SV[j] = SV_trial[j]  # dead in this port (see SV allocation note above)
            Cikp1_nd[j] = Cikp1[j]
        elseif flag_implicit == 1
            # Semi-implicit update; round-trip through Matrix + (A+Aᵀ)/2 for the strict SymT cast.
            tmp_nd = Matrix(Cik[j]) .+ gamma_factor[j] .* gamma_D[j] .* muVi[j] .* dt .* Matrix(C_star)
            Cikp1_nd[j] = _symmetrize_to_SymT(tmp_nd)
            J_Cikp1 = det(Cikp1_nd[j])
            tmp1 = (J_Cikp1^(-1 / 3)) .* Matrix(Cikp1_nd[j])
            Cikp1[j] = _symmetrize_to_SymT(tmp1)
            Cikp1_inv[j] = inv(Cikp1[j])
            # SV[j] = muVi[j] * C_inv ⋅ get_deviate_zhao(C_star ⋅ Cikp1_inv[j])  # dead in this port (see SV allocation note above)
        end
    end

    muVkp1 = muVk + alpha * (muVe - muVk) * gamma_D[1] * dt
    # From here on `gamma_D` is the normalized flow prefactor used by the
    # stiffness block, not the raw power-flow measure.
    gamma_D = gamma_D .* gamma_factor

    # Both inputs are SymT, so + preserves the symmetric storage; Sk is consumed as P = F ⋅ Sk in _assemble_element!.
    Sk_dev = SE_dev
    for j = 1:NV
        Sk_dev = Sk_dev + SV_trial[j]
    end
    Sk = Sk_dev + SE_vol

    state_vevp_temp_eg = VEVP_Zhao2021_ATStateInternal(Cikp1, Cikp1_inv, Cikp1_nd, muVkp1, strain_maxkp1)

    return state_vevp_temp_eg, Sk, Sk_dev, gamma_D
end

function prepare_vevp_zhao_stiffness_data_gp(S_dev::SymT, S::SymT, gamma_D::Vector{Float64},
    state_vevp_temp_eg::VEVP_Zhao2021_ATStateInternal,
    F::Tensor{2,3,Float64,9}, Finv::Tensor{2,3,Float64,9},
    prm_vevp::VEVP_Zhao2021_AT,
    state_eg::VEVP_Zhao2021_ATStateInternal, dt, flag_implicit::Int)
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

    C = F' ⋅ F
    J = det(F)
    C_star = C * J^(-2 / 3)

    Ci_inv = (flag_implicit == 0) ? Cik_inv : Cikp1_inv

    # Convert at the boundary: the stiffness block operates on general
    # 3×3 matrices (outer products of shape function gradients are not
    # symmetric).
    return VEVP_Zhao2021_ATStiffnessDataGP(
        kappa, J, muE, NV,
        Matrix(C), Matrix(S), Matrix(S_dev), Matrix(C_star),
        Matrix(Finv), Matrix(Finv'),
        muVki, Matrix.(Ci_inv), gamma_D, Matrix.(Cikp1_nd), dt, flag_implicit,
    )
end

function calc_vevp_zhao_stiffness_block_jk(data::VEVP_Zhao2021_ATStiffnessDataGP, DN_j_vec, DN_k_vec)
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

"""
    _assemble_element!(ke, re, states, mp::VEVP_Zhao2021_AT, cellvalues, alphavalues, u, dt)

Assembles the stiffness matrix `ke` and residual vector `re` for a VEVP element.
`states` is a vector of `VEVP_Zhao2021_ATState`, one per quadrature point.
"""
function _assemble_element!(ke, re, states::Vector{<:AbstractMaterialState}, mp::VEVP_Zhao2021_AT, cellvalues, alphavalues, u, dt)
    n_basefuncs = getnbasefunctions(cellvalues)

    # Unit scaling: apply the user-supplied `dt_scale` kwarg (default 1.0).
    # Stress postprocessing intentionally uses dt = 0.0 to avoid evolving
    # history during output.
    dt_scaled = dt * mp.vevp_dt_scale

    for qp in 1:getnquadpoints(cellvalues)
        state = states[qp]
        state_old = state.previous
        α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)

        # Deformation gradient (Tensors.jl 3×3 representation; no
        # conversion to Matrix needed since the forward solve is in
        # `SymmetricTensor`).
        F = deformation_gradient(cellvalues, qp, u)
        Finv = inv(F)

        # Constitutive update (Semi-implicit = 1)
        new_internal_state, Sk, Sk_dev, gamma_D = set_vevp_zhao_stress_S_C(F, Finv, dt_scaled, mp, state_old, 1)

        # Store for next iteration
        state.current = new_internal_state

        # Internal Force Contribution: P = F ⋅ S
        P = F ⋅ Sk

        if re !== nothing
            for i in 1:n_basefuncs
                ∇δui = shape_gradient(cellvalues, qp, i)
                re[i] += (∇δui ⊡ P) * α_dΩ
            end
        end

        # Stiffness Matrix Contribution
        stiff_data = prepare_vevp_zhao_stiffness_data_gp(Sk_dev, Sk, gamma_D, new_internal_state, F, Finv, mp, state_old, dt_scaled, 1)

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
    _compute_stress_qp(mp::VEVP_Zhao2021_AT, cellvalues, alphavalues, qp, u_local, state, dt)

Computes the Cauchy stress at a quadrature point for output.
The `dt` argument is ignored by design: stress output evaluates the current state with a zero time increment, so postprocessing does not evolve history.
"""
function _compute_stress_qp(mp::VEVP_Zhao2021_AT, cellvalues, alphavalues, qp, u_local, state::VEVP_Zhao2021_ATState, dt=0.0)
    state_old = state.current

    F = deformation_gradient(cellvalues, qp, u_local)
    Finv = inv(F)
    J = det(F)

    _, Sk, _, _ = set_vevp_zhao_stress_S_C(F, Finv, 0.0, mp, state_old, 1)

    # Cauchy stress σ = (1/J) F ⋅ S ⋅ Fᵀ (push-forward of the PK2 stress)
    σ = (F ⋅ Sk ⋅ F') / J
    return σ
end

"""
    _assemble_element!(ke, re, states, ps::PlaneStrain{VEVP_Zhao2021_AT}, cellvalues, alphavalues, u, dt)

Specialized plane strain assembly for `VEVP_Zhao2021_AT`.
The in-plane deformation gradient is embedded in 3D with `F33 = 1`, and the analytical AT stiffness block is assembled from embedded shape gradients.
"""
function _assemble_element!(ke, re, states::Vector{<:AbstractMaterialState},
    ps::PlaneStrain{VEVP_Zhao2021_AT}, cellvalues, alphavalues, u, dt)
    mp = ps.model
    n_basefuncs = getnbasefunctions(cellvalues)
    n_nodes = n_basefuncs ÷ 2
    dt_scaled = dt * mp.vevp_dt_scale

    for qp in 1:getnquadpoints(cellvalues)
        state = states[qp]::VEVP_Zhao2021_ATState
        state_old = state.previous
        α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)

        F2D = deformation_gradient(cellvalues, qp, u)
        # Tensors.value on F2D, which the assembler passes as Float64; this path is not AD-transparent.
        F = embed_F_2D_to_3D(Tensors.value(F2D), 1.0)
        Finv = inv(F)

        new_internal_state, Sk, Sk_dev, gamma_D = set_vevp_zhao_stress_S_C(F, Finv, dt_scaled, mp, state_old, 1)
        state.current = new_internal_state

        P2D = extract_P_2D(F ⋅ Sk)

        if re !== nothing
            for i in 1:n_basefuncs
                re[i] += (shape_gradient(cellvalues, qp, i) ⊡ P2D) * α_dΩ
            end
        end

        if ke !== nothing
            stiff_data = prepare_vevp_zhao_stiffness_data_gp(Sk_dev, Sk, gamma_D, new_internal_state, F, Finv, mp, state_old, dt_scaled, 1)
            DNs = Vector{Vector{Float64}}(undef, n_nodes)
            for a in 1:n_nodes
                ∇Na = shape_gradient(cellvalues, qp, (a - 1) * 2 + 1)
                DNs[a] = [∇Na[1, 1], ∇Na[1, 2], 0.0]
            end

            for a in 1:n_nodes
                DN_a = DNs[a]
                for b in 1:n_nodes
                    DN_b = DNs[b]
                    K_ab = calc_vevp_zhao_stiffness_block_jk(stiff_data, DN_a, DN_b)
                    K_ab .*= α_dΩ

                    r_idx, c_idx = (a - 1) * 2, (b - 1) * 2
                    for r in 1:2, c in 1:2
                        ke[r_idx+r, c_idx+c] += K_ab[r, c]
                    end
                end
            end
        end
    end
end

function _compute_stress_qp(ps::PlaneStrain{VEVP_Zhao2021_AT}, cellvalues, alphavalues, qp, u_local,
    state::VEVP_Zhao2021_ATState, dt=0.0)
    F2D = deformation_gradient(cellvalues, qp, u_local)
    F = embed_F_2D_to_3D(Tensors.value(F2D), 1.0)
    Finv = inv(F)
    J = det(F)

    _, Sk, _, _ = set_vevp_zhao_stress_S_C(F, Finv, 0.0, ps.model, state.current, 1)
    σ = (F ⋅ Sk ⋅ F') / J
    return extract_P_2D(σ)
end

function _unsupported_zhao_at_wrapper()
    throw(ArgumentError("VEVP_Zhao2021_AT does not currently support the generic PlaneStrain/PlaneStress AD wrapper path. PlaneStrain has a specialized analytical-tangent assembly path; PlaneStress is not implemented. Use VEVP_Zhao2021_AD for PlaneStress or other generic wrapped 2D analyses."))
end

function compute_PK1_3D(mp::VEVP_Zhao2021_AT, F::Tensor{2,3,T}, dt, state::VEVP_Zhao2021_ATState) where T
    return _unsupported_zhao_at_wrapper()
end

function update_state_from_3D!(state::VEVP_Zhao2021_ATState, mp::VEVP_Zhao2021_AT, F::Tensor{2,3}, dt)
    return _unsupported_zhao_at_wrapper()
end
