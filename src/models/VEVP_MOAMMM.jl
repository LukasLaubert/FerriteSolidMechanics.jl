# --- 1. Material struct ---

"""
    VEVP_MOAMMM(order, KK_inf, GG_inf, alpha, nu_p, eta, p_exp,
                  sigmac0, hc1, hc2, hcexp,
                  sigmat0, ht1, ht2, htexp,
                  hb0, hb1, hb2,
                  KK, k, GG, g)

Viscoelastic-viscoplastic material model from the [MOAMMM project](https://www.moammm.eu/) with automatic differentiation.
Combines a logarithmic strain viscoelastic update, a Drucker–Prager yield surface, isotropic hardening through separate compression and tension yield stress evolution, and kinematic hardening.

### Parameters

- `order` – Polynomial order for the logarithm/exponential approximations
- `KK_inf` – Equilibrium bulk modulus; long-term volumetric stiffness
- `GG_inf` – Equilibrium shear modulus; long-term deviatoric stiffness
- `alpha` – Drucker–Prager yield exponent; 1 → linear Drucker–Prager, 2 → quadratic
- `nu_p` – Plastic Poisson's ratio; controls volumetric–deviatoric coupling under plastic flow
- `eta` – Viscoplastic viscosity; higher values delay rate-dependent flow
- `p_exp` – Viscoplastic exponent; controls stress sensitivity of the viscoplastic rate
- `sigmac0` – Initial compression yield limit; higher values delay compressive yield
- `hc1` – Linear compression yield stress growth coefficient
- `hc2` – Saturation compression yield stress increment
- `hcexp` – Compression yield stress saturation exponent; at γ = 1/h_{c,exp}, the factor (1 − exp(−h_{c,exp}·γ)) ≈ 0.63
- `sigmat0` – Initial tension yield limit; higher values delay tensile yield
- `ht1` – Linear tension yield stress growth coefficient
- `ht2` – Saturation tension yield stress increment
- `htexp` – Tension yield stress saturation exponent; at γ = 1/h_{t,exp}, the factor (1 − exp(−h_{t,exp}·γ)) ≈ 0.63
- `hb0` – Constant term of the kinematic hardening polynomial that scales the total backstress rate
- `hb1` – Linear term of the kinematic hardening polynomial; scales with plastic strain
- `hb2` – Quadratic term of the kinematic hardening polynomial; scales with the square of plastic strain
- `KK` – Vector of per-branch bulk moduli; raise the volumetric branch stiffness
- `k` – Vector of per-branch volumetric relaxation times; larger values delay volumetric equilibration
- `GG` – Vector of per-branch shear moduli; raise the deviatoric branch stiffness
- `g` – Vector of per-branch deviatoric relaxation times; larger values delay deviatoric equilibration

# References

- V.-D. Nguyen, F. Lani, T. Pardoen, X. P. Morelle, L. Noels.
  *A large strain hyperelastic viscoelastic-viscoplastic-damage constitutive model based on a multi-mechanism non-local damage continuum for amorphous glassy polymers.*
  International Journal of Solids and Structures **96** (2016) 192–216.
  <https://doi.org/10.1016/j.ijsolstr.2016.06.008>

The code-level reference for this implementation is the Fortran UMAT [`umat.f`](https://gitlab.uliege.be/moammm/moammmPublic/code/-/blob/main/MaterialModels/FiniteStrain/Finite_VEVP/umat.f?ref_type=heads) from the moammmPublic code repository.

!!! note
    The Nguyen et al. (2016) paper describes a full hyperelastic–viscoelastic–viscoplastic model with damage and softening.
    The translated Fortran `UMAT` (`umat.f` in the moammmPublic repository) implements only the damage-free, no-softening subset of that model.
    This Julia port implements the same subset.
"""
struct VEVP_MOAMMM <: AbstractMaterial
    order::Int
    KK_inf::Float64
    GG_inf::Float64
    alpha::Float64
    nu_p::Float64
    eta::Float64
    p_exp::Float64
    sigmac0::Float64
    hc1::Float64
    hc2::Float64
    hcexp::Float64
    sigmat0::Float64
    ht1::Float64
    ht2::Float64
    htexp::Float64
    hb0::Float64
    hb1::Float64
    hb2::Float64
    KK::Vector{Float64}
    k::Vector{Float64}
    GG::Vector{Float64}
    g::Vector{Float64}
    nbr::Int  # number of Maxwell branches (= length of KK = length of k = length of GG = length of g)

    function VEVP_MOAMMM(order, KK_inf, GG_inf, alpha, nu_p, eta, p_exp, sigmac0, hc1, hc2, hcexp, sigmat0, ht1, ht2, htexp, hb0, hb1, hb2, KK, k, GG, g)
        N = length(GG)
        length(KK) == N && length(k) == N && length(g) == N || throw(ArgumentError("VEVP_MOAMMM: branch vectors KK, k, GG, g must all have the same length (got KK=$(length(KK)), k=$(length(k)), GG=$(length(GG)), g=$(length(g))"))
        eta > 0.0 || throw(ArgumentError("VEVP_MOAMMM: eta must be positive (got eta=$eta)"))
        p_exp > 0.0 || throw(ArgumentError("VEVP_MOAMMM: p_exp must be positive (got p_exp=$p_exp)"))
        new(order, KK_inf, GG_inf, alpha, nu_p, eta, p_exp, sigmac0, hc1, hc2, hcexp, sigmat0, ht1, ht2, htexp, hb0, hb1, hb2, KK, k, GG, g, N)
    end
end

# --- 2. Convergence error and state struct ---

"""
    VEVP_MOAMMMConvergenceError

Recoverable failure raised when the local `VEVP_MOAMMM` plastic correction does not converge within its fixed Newton iteration budget.
"""
struct VEVP_MOAMMMConvergenceError <: LocalAssemblyFailure
    plastic_multiplier
    residual
    iterations::Int
end

function Base.showerror(io::IO, err::VEVP_MOAMMMConvergenceError)
    print(io, "VEVP_MOAMMM local plastic solve failed to converge")
    print(io, ": plastic_multiplier=", err.plastic_multiplier)
    print(io, ", residual=", err.residual)
    print(io, ", iterations=", err.iterations)
end

"""
    VEVP_MOAMMMState <: AbstractMaterialState

Quadrature point state for `VEVP_MOAMMM`.
Each variable is stored as a trial value and a committed start-of-step value.
The `current_*` fields hold the state written after each constitutive update.
`update_state!` copies them into the matching `previous_*` fields, and `revert_state!` copies the `previous_*` fields back into them.
The variables are the plastic deformation gradient `Fvp`, the elastic logarithmic strain `Eve`, the equivalent plastic strain `gma`, the corotational backstress tensor `b`, and one deviatoric `AA[j]` / volumetric scalar `BB[j]` pair per Maxwell branch.
"""
mutable struct VEVP_MOAMMMState <: AbstractMaterialState
    current_Fvp::Tensor{2,3,Float64,9}
    current_Eve::Tensor{2,3,Float64,9}
    current_gma::Float64
    current_b::Tensor{2,3,Float64,9}
    current_AA::Vector{Tensor{2,3,Float64,9}}
    current_BB::Vector{Float64}

    previous_Fvp::Tensor{2,3,Float64,9}
    previous_Eve::Tensor{2,3,Float64,9}
    previous_gma::Float64
    previous_b::Tensor{2,3,Float64,9}
    previous_AA::Vector{Tensor{2,3,Float64,9}}
    previous_BB::Vector{Float64}
end

# --- 3. API hooks ---

is_linear(::VEVP_MOAMMM) = false

function create_state(m::VEVP_MOAMMM)
    nbr = m.nbr
    return VEVP_MOAMMMState(
        one(Tensor{2,3,Float64,9}),
        zero(Tensor{2,3,Float64,9}),
        0.0,
        zero(Tensor{2,3,Float64,9}),
        [zero(Tensor{2,3,Float64,9}) for _ in 1:nbr],
        zeros(Float64, nbr),
        one(Tensor{2,3,Float64,9}),
        zero(Tensor{2,3,Float64,9}),
        0.0,
        zero(Tensor{2,3,Float64,9}),
        [zero(Tensor{2,3,Float64,9}) for _ in 1:nbr],
        zeros(Float64, nbr),
    )
end

function update_state!(state::VEVP_MOAMMMState)
    state.previous_Fvp = state.current_Fvp
    state.previous_Eve = state.current_Eve
    state.previous_gma = state.current_gma
    state.previous_b = state.current_b
    copy_state!(state.previous_AA, state.current_AA)
    copy_state!(state.previous_BB, state.current_BB)
    return nothing
end

function revert_state!(state::VEVP_MOAMMMState)
    state.current_Fvp = state.previous_Fvp
    state.current_Eve = state.previous_Eve
    state.current_gma = state.previous_gma
    state.current_b = state.previous_b
    copy_state!(state.current_AA, state.previous_AA)
    copy_state!(state.current_BB, state.previous_BB)
    return nothing
end

# --- 4. Logarithmic / exponential approximant helpers ---

function _approx_log_umat(AA::Tensor{2,3,T}, order::Int) where T
    I_mat = one(AA)
    coeffs = zeros(T, order + 1)
    for ii in 2:order+1
        idx = ii - 1
        coeffs[ii] = (ii % 2 == 0) ? (1.0 / idx) : (-1.0 / idx)
    end
    res = coeffs[order+1] * I_mat
    for ii in 1:order
        nn = order + 1 - ii
        res = coeffs[nn] * I_mat + res ⋅ AA
    end
    return res
end

function _d_approx_log_umat(AA::Tensor{2,3,T}, order::Int) where T
    I_mat = one(AA)
    II_sym = SymmetricTensor{4,3,T}((i, j, k, l) -> 0.5 * (I_mat[i, k] * I_mat[j, l] + I_mat[i, l] * I_mat[j, k]))

    coeffs = zeros(T, order + 1)
    for ii in 2:order+1
        idx = ii - 1
        coeffs[ii] = (ii % 2 == 0) ? (1.0 / idx) : (-1.0 / idx)
    end

    logAA = coeffs[order+1] * I_mat
    dlogAA = zero(Tensor{4,3,T})

    for ii in 1:order
        nn = order + 1 - ii

        # Horner derivative update:
        # D_new_ijrs = D_old_ilrs * A_lj + P_il * II_ljrs
        P_term = dot(logAA, II_sym)
        D_old = dlogAA
        dlogAA = Tensor{4,3,T}((i, j, r, s) -> D_old[i, 1, r, s] * AA[1, j] + D_old[i, 2, r, s] * AA[2, j] + D_old[i, 3, r, s] * AA[3, j] + P_term[i, j, r, s])
        logAA = coeffs[nn] * I_mat + logAA ⋅ AA
    end
    return dlogAA
end

function _approx_exp_umat(Q::Tensor{2,3,T}, order::Int) where T
    I_mat = one(Q)
    res = I_mat
    Q_pow = I_mat
    fact = 1.0
    for i in 1:order-1
        Q_pow = Q_pow ⋅ Q
        fact *= i
        res += Q_pow / fact
    end
    return res
end

function _hardn_umat(prm::VEVP_MOAMMM, gma_0::T, gma::T) where T
    sc = prm.sigmac0 + prm.hc1 * gma + prm.hc2 * (1.0 - exp(-prm.hcexp * gma))
    st = prm.sigmat0 + prm.ht1 * gma + prm.ht2 * (1.0 - exp(-prm.htexp * gma))
    hhc = prm.hc1 + prm.hc2 * prm.hcexp * exp(-prm.hcexp * gma)
    hht = prm.ht1 + prm.ht2 * prm.htexp * exp(-prm.htexp * gma)
    hhb = prm.hb0 + prm.hb1 * gma_0 + prm.hb2 * gma_0^2
    return sc, st, hhc, hht, hhb
end

function _DPcoeff_umat(alpha::Float64, sc::T, st::T) where T
    m = st / max(eps(), sc)
    a2 = 1.0 / (max(eps(), sc)^alpha)
    a1 = 3.0 * ((m^alpha - 1.0) / (m + 1.0)) * (1.0 / max(eps(), sc))
    a0 = (m^alpha + m) / (m + 1.0)
    return m, a0, a1, a2
end

# --- 5. Main driver ---

"""
    solve_local_vevp_moammm(F, dt, mat, prev::VEVP_MOAMMMState)

Run one local viscoelastic-plastic update of `VEVP_MOAMMM` for a 3D deformation gradient `F`, given the start-of-step state `prev` and the time step size `dt`.
The update reads the `prev.previous_*` fields, advances the viscoelastic predictors, and applies the Drucker–Prager / Perzyna plastic corrector where needed.
It returns the Cauchy stress `σ` and a `NamedTuple` of the new trial state values `Fvp`, `Eve`, `gma`, `b`, `AA` and `BB`.
The caller writes those into the `state.current_*` fields as trial values, and does not commit them.
"""
function solve_local_vevp_moammm(F::Tensor{2,3,T}, dt::Float64, mat::VEVP_MOAMMM, prev::VEVP_MOAMMMState) where T
    prm = mat
    I_mat = one(Tensor{2,3,T})

    # Start-of-step state
    Fvp_n = prev.previous_Fvp
    Fe_tr = F ⋅ inv(Fvp_n)
    Ce_tr = symmetric(Fe_tr' ⋅ Fe_tr)
    # Polynomial log for stress and tangent
    get_log_val(C) = 0.5 * _approx_log_umat(C - I_mat, prm.order)
    Eve_tr = get_log_val(Ce_tr)

    # Viscoelastic predictor
    nbr = mat.nbr
    AA_tr = Vector{Tensor{2,3,T}}(undef, nbr)
    BB_tr = Vector{T}(undef, nbr)
    GGe, KKe = T(prm.GG_inf), T(prm.KK_inf)
    Eve_n = prev.previous_Eve
    dE = Eve_tr - Eve_n
    dev_dE, tr_dE = dev(dE), tr(dE)
    for i in 1:nbr
        ztg, ztak = exp(-dt / (2 * prm.g[i])), exp(-dt / (2 * prm.k[i]))
        GGe += prm.GG[i] * ztg
        KKe += prm.KK[i] * ztak
        AA_tr[i] = exp(-dt / prm.g[i]) * prev.previous_AA[i] + ztg * dev_dE
        BB_tr[i] = exp(-dt / prm.k[i]) * T(prev.previous_BB[i]) + ztak * tr_dE
    end
    dk = dev(Eve_tr) * (2 * prm.GG_inf)
    pv = tr(Eve_tr) * prm.KK_inf
    for i in 1:nbr
        dk += 2 * prm.GG[i] * AA_tr[i]
        pv += BB_tr[i] * prm.KK[i]
    end
    kappa_tr = dk + pv * I_mat

    bn = prev.previous_b
    phi_tr = kappa_tr - bn
    pt_tr, PhiEq_tr = tr(phi_tr) / 3.0, sqrt(1.5 * (dev(phi_tr) ⊡ dev(phi_tr)))
    gn = T(prev.previous_gma)
    gma_0 = gn
    sc, st, _, _, _ = _hardn_umat(prm, gma_0, gn)
    m, a0, a1, a2 = _DPcoeff_umat(prm.alpha, sc, st)
    ft = a2 * max(eps(), PhiEq_tr)^prm.alpha - a1 * pt_tr - a0

    σ_f, Fvp_f, Eve_f, gma_f, b_f, AA_f, BB_f = zero(Tensor{2,3,T}), Fvp_n, Eve_tr, gn, bn, AA_tr, BB_tr

    function compute_cauchy(Fe, Ce, kappa, J)
        # S = 2.0 * kappa : dE/dC (Thermodynamic consistency)
        dEdC = 0.5 * _d_approx_log_umat(Ce - I_mat, prm.order)
        return (Fe ⋅ (2.0 * (kappa ⊡ dEdC)) ⋅ Fe') / J
    end

    if ft <= 1e-11
        σ_f = compute_cauchy(Fe_tr, Ce_tr, kappa_tr, det(F))
    else
        # Plastic corrector
        G, beta, kp = zero(T), 4.5 * (1.0 - 2.0 * prm.nu_p) / (prm.nu_p + 1.0), 1.0 / sqrt(1.0 + 2.0 * prm.nu_p^2)
        u, v, gi, pt_i, PhiEq_i = 1.0, 1.0, gn, pt_tr, PhiEq_tr
        maxiter = 500
        converged = false
        last_f = ft
        for _ in 1:maxiter
            sci, sti, hhci, hhti, hhbi = _hardn_umat(prm, gma_0, gi)
            mi, a0i, a1i, a2i = _DPcoeff_umat(prm.alpha, sci, sti)
            GGti, KKti = GGe + (kp / 2.0) * hhbi, KKe + (kp / 3.0) * hhbi
            # Guard dt=0 so stress output does not introduce viscous flow
            eta_over_dt = dt > 0.0 ? prm.eta / dt : 0.0
            f = a2i * max(eps(), PhiEq_i)^prm.alpha - a1i * pt_i - a0i
            G > 0 && (f -= (eta_over_dt * (G + 1e-20))^prm.p_exp)
            last_f = f
            if abs(f) < 1e-11
                converged = true
                break
            end
            A = sqrt(6 * PhiEq_i^2 + (4 / 3) * beta^2 * pt_i^2)
            dAdG = -(72 * GGti * PhiEq_i^2 / u + 16 * KKti * beta^3 * pt_i^2 / (3 * v)) / (2 * A)
            dDgG = kp * (A + G * dAdG)
            Dm = (hhti * sci - hhci * sti) / max(eps(), sci^2)
            Da1 = (3 / max(eps(), sci)) * (prm.alpha * mi^(prm.alpha - 1) / (mi + 1) - ((mi^prm.alpha - 1) / (mi + 1)) / (mi + 1))
            H2, H1 = -prm.alpha * (max(eps(), sci)^(-prm.alpha - 1)) * hhci, Da1 * Dm - 3 * ((mi^prm.alpha - 1) / (mi + 1) / max(eps(), sci^2)) * hhci
            H0 = ((prm.alpha * mi^(prm.alpha - 1) + 1.0) / (mi + 1) - ((prm.alpha * mi^(prm.alpha - 1) + mi) / (mi + 1)) / (mi + 1)) * Dm
            Df = (H2 * max(eps(), PhiEq_i)^prm.alpha - H1 * pt_i - H0) * dDgG - (prm.alpha * a2i * 6.0 * GGti * max(eps(), PhiEq_i)^prm.alpha) / u + a1i * pt_i * 2.0 * beta * KKti / v
            G > 0 && (Df -= prm.p_exp * eta_over_dt^prm.p_exp * (G + 1e-20)^(prm.p_exp - 1.0))
            dG = -f / Df
            if (G + dG) <= 0
                G /= 2.0
            else
                G += dG
            end
            u, v = 1 + 6 * GGti * G, 1 + 2 * beta * KKti * G
            PhiEq_i, pt_i = PhiEq_tr / u, pt_tr / v
            gi = gn + kp * G * sqrt(6.0 * PhiEq_i^2 + (4.0 / 3.0) * beta^2 * pt_i^2)
        end
        converged || throw(VEVP_MOAMMMConvergenceError(Tensors.value(G), Tensors.value(last_f), maxiter))
        # Corrected state
        Q = 3 * (dev(phi_tr) / u) + (2.0 * beta * pt_i / 3) * I_mat
        Fvp_f = _approx_exp_umat(G * Q, prm.order) ⋅ Fvp_n
        Fe_f = F ⋅ inv(Fvp_f)
        Ce_f = symmetric(Fe_f' ⋅ Fe_f)
        Eve_f = get_log_val(Ce_f)
        df_f = Eve_f - Eve_n
        for i in 1:nbr
            AA_f[i] = exp(-dt / prm.g[i]) * prev.previous_AA[i] + exp(-dt / (2.0 * prm.g[i])) * dev(df_f)
            BB_f[i] = exp(-dt / prm.k[i]) * T(prev.previous_BB[i]) + exp(-dt / (2.0 * prm.k[i])) * tr(df_f)
        end
        dk_f = dev(Eve_f) * (2 * prm.GG_inf)
        pv_f = tr(Eve_f) * prm.KK_inf
        for i in 1:nbr
            dk_f += 2 * prm.GG[i] * AA_f[i]
            pv_f += BB_f[i] * prm.KK[i]
        end
        σ_f = compute_cauchy(Fe_f, Ce_f, dk_f + pv_f * I_mat, det(F))
        gma_f = gi
        _, _, _, _, hhbf = _hardn_umat(prm, gma_0, gi)
        b_f = bn + hhbf * kp * (G * Q)
    end

    return σ_f, (Fvp=Fvp_f, Eve=Eve_f, gma=gma_f, b=b_f, AA=AA_f, BB=BB_f)
end

kinematics(::VEVP_MOAMMM) = FiniteStrain()

function material_response(mp::VEVP_MOAMMM, F::Tensor{2,3}, state::VEVP_MOAMMMState, dt, cache=nothing)
    σ, upd = solve_local_vevp_moammm(F, dt, mp, state)
    P = det(F) * σ ⋅ inv(F)'
    dP_dF = Tensors.gradient(F_ -> compute_PK1_3D(mp, F_, dt, state), F)
    return P, dP_dF, upd
end

function set_trial!(state::VEVP_MOAMMMState, new::NamedTuple)
    state.current_Fvp = new.Fvp
    state.current_Eve = new.Eve
    state.current_gma = new.gma
    state.current_b = new.b
    state.current_AA = new.AA
    state.current_BB = new.BB
    return nothing
end

function material_stress(mp::VEVP_MOAMMM, F::Tensor{2,3}, state::VEVP_MOAMMMState, dt, cache=nothing)
    old_Fvp = state.previous_Fvp
    old_Eve = state.previous_Eve
    old_gma = state.previous_gma
    old_b = state.previous_b
    old_AA = state.previous_AA
    old_BB = state.previous_BB

    state.previous_Fvp = state.current_Fvp
    state.previous_Eve = state.current_Eve
    state.previous_gma = state.current_gma
    state.previous_b = state.current_b
    state.previous_AA = state.current_AA
    state.previous_BB = state.current_BB
    return try
        solve_local_vevp_moammm(F, 0.0, mp, state)[1]  # pass dt=0.0 to not advance history in postprocessing (!)
    finally
        state.previous_Fvp = old_Fvp
        state.previous_Eve = old_Eve
        state.previous_gma = old_gma
        state.previous_b = old_b
        state.previous_AA = old_AA
        state.previous_BB = old_BB
    end
end

# 2D wrapper interface for VEVP_MOAMMM
function compute_PK1_3D(mp::VEVP_MOAMMM, F::Tensor{2,3,T}, dt, state::VEVP_MOAMMMState) where T
    σ, _ = solve_local_vevp_moammm(F, dt, mp, state)
    return det(F) * σ ⋅ inv(F)'
end

function update_state_from_3D!(state::VEVP_MOAMMMState, mp::VEVP_MOAMMM, F::Tensor{2,3}, dt)
    _, upd = solve_local_vevp_moammm(Tensors.value(F), dt, mp, state)
    state.current_Fvp = upd.Fvp
    state.current_Eve = upd.Eve
    state.current_gma = upd.gma
    state.current_b = upd.b
    state.current_AA = upd.AA
    state.current_BB = upd.BB
    return nothing
end