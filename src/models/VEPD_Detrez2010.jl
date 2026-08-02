# --- 1. Material struct ---

"""
    VEPD_Detrez2010(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ; plastic_update=:end_step, maxwell_update=:closed_form_cv)

Viscoelastic-plastic-damage material model after Detrez.
Combines crystalline elasticity, network hyperelasticity, selectable Maxwell-branch updates, and isotropic hardening with damage.

### Parameters

- `E` – Young's modulus of the crystalline phase; higher values stiffen the elastic stress response
- `ν` – Poisson's ratio of the crystalline phase; controls lateral contraction under stretch
- `R0` – Initial yield stress; higher values delay the onset of plasticity
- `Q` – Saturation modulus; higher values raise the asymptotic hardening limit
- `b` – Saturation exponent; larger values cause the yield stress to plateau at smaller plastic strains
- `α` – Damage scaling factor; sets the asymptotic damage level (value is capped at 1)
- `β` – Damage exponent; larger values accelerate damage accumulation with plastic strain
- `n_ab` – Number of Kuhn segments per Arruda–Boyce chain; larger values soften the network at moderate stretches and push locking to higher stretches
- `μ_ab` – Network shear modulus; scales the Arruda–Boyce network stress contribution
- `G` – Vector of per-Maxwell-branch shear moduli; higher values raise the branch elastic stiffness
- `τ` – Vector of per-Maxwell-branch relaxation times; larger values delay branch equilibration

### Keyword arguments

- `plastic_update` – `:end_step` applies one return map at the end of the global step; `:path_substepped` applies the same return map over 8 deformation-path subincrements
- `maxwell_update` – `:closed_form_cv` updates the viscous right Cauchy-Green branch state `Cv` in closed form; `:objective_rate` advances branch Cauchy stresses with a 16-substep objective rate update

# References

- F. Detrez, S. Cantournet, R. Séguéla.
  *A constitutive model for semi-crystalline polymer deformation
  involving lamellar fragmentation.*
  Comptes Rendus Mécanique **338**(12) (2010) 681–687.
  <https://doi.org/10.1016/j.crme.2010.10.008>

The code-level reference for this implementation is an unpublished C
routine that was kindly provided by Fabrice Detrez.
"""
struct VEPD_Detrez2010{PlasticUpdate,MaxwellUpdate} <: AbstractMaterial
    E::Float64; ν::Float64; R0::Float64; Q::Float64; b::Float64
    α::Float64; β::Float64; N_ab::Float64; μ_ab::Float64
    G::Vector{Float64}; τ::Vector{Float64}
    plastic_update::Symbol; maxwell_update::Symbol

    function VEPD_Detrez2010(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ; plastic_update=:end_step, maxwell_update=:closed_form_cv)
        length(G) == length(τ) || throw(ArgumentError("VEPD_Detrez2010: G and τ length mismatch"))
        all(>(0.0), τ) || throw(ArgumentError("VEPD_Detrez2010: τ must be positive"))
        plastic_update in _DETREZ_PLASTIC_UPDATE_OPTIONS ||
            throw(ArgumentError("VEPD_Detrez2010: plastic_update must be one of $(_DETREZ_PLASTIC_UPDATE_OPTIONS)"))
        maxwell_update in _DETREZ_MAXWELL_UPDATE_OPTIONS ||
            throw(ArgumentError("VEPD_Detrez2010: maxwell_update must be one of $(_DETREZ_MAXWELL_UPDATE_OPTIONS)"))
        new{plastic_update,maxwell_update}(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ, plastic_update, maxwell_update)
    end
end

const _DETREZ_PLASTIC_UPDATE_OPTIONS = (:end_step, :path_substepped)
const _DETREZ_MAXWELL_UPDATE_OPTIONS = (:closed_form_cv, :objective_rate)
const _DETREZ_PLASTIC_PATH_SUBSTEPS = 8
const _DETREZ_OBJECTIVE_MAXWELL_SUBSTEPS = 16

# --- 2. Convergence error and state struct ---

"""
    VEPD_Detrez2010ConvergenceError

Recoverable failure raised when the local VEPD Detrez plastic correction
does not converge within its fixed Newton iteration budget.
"""
struct VEPD_Detrez2010ConvergenceError <: LocalAssemblyFailure
    delta_p
    residual
    iterations::Int
end

function Base.showerror(io::IO, err::VEPD_Detrez2010ConvergenceError)
    print(io, "VEPD_Detrez2010 local plastic solve failed to converge")
    print(io, ": delta_p=", err.delta_p)
    print(io, ", residual=", err.residual)
    print(io, ", iterations=", err.iterations)
end

mutable struct VEPD_Detrez2010State <: AbstractMaterialState
    current_F::Tensor{2,3,Float64,9}
    current_Fp::Tensor{2,3,Float64,9}
    current_p::Float64
    current_Cv::Vector{SymmetricTensor{2,3,Float64,6}}
    current_σv::Vector{Tensor{2,3,Float64,9}}

    previous_F::Tensor{2,3,Float64,9}
    previous_Fp::Tensor{2,3,Float64,9}
    previous_p::Float64
    previous_Cv::Vector{SymmetricTensor{2,3,Float64,6}}
    previous_σv::Vector{Tensor{2,3,Float64,9}}

    track_σv::Bool
end

# --- 3. API hooks ---

function create_state(mat::VEPD_Detrez2010)
    F = one(Tensor{2,3}); Fp = one(Tensor{2,3}); p = 0.0
    Cv = [one(SymmetricTensor{2,3}) for _ in 1:length(mat.G)]
    track_σv = mat.maxwell_update == :objective_rate
    σv = track_σv ? [zero(Tensor{2,3}) for _ in 1:length(mat.G)] : Tensor{2,3,Float64,9}[]
    return VEPD_Detrez2010State(F, Fp, p, copy(Cv), copy(σv), F, Fp, p, copy(Cv), copy(σv), track_σv)
end

function update_state!(state::VEPD_Detrez2010State)
    state.previous_F = state.current_F
    state.previous_Fp = state.current_Fp
    state.previous_p = state.current_p
    copy_state!(state.previous_Cv, state.current_Cv)
    state.track_σv && copy_state!(state.previous_σv, state.current_σv)
end

function revert_state!(state::VEPD_Detrez2010State)
    state.current_F = state.previous_F
    state.current_Fp = state.previous_Fp; state.current_p = state.previous_p;
    copy_state!(state.current_Cv, state.previous_Cv)
    state.track_σv && copy_state!(state.current_σv, state.previous_σv)
end

function copy_state!(dest::VEPD_Detrez2010State, src::VEPD_Detrez2010State)
    dest.current_F = src.current_F
    dest.current_Fp = src.current_Fp
    dest.current_p = src.current_p
    copy_state!(dest.current_Cv, src.current_Cv)
    dest.track_σv && copy_state!(dest.current_σv, src.current_σv)
    dest.previous_F = src.previous_F
    dest.previous_Fp = src.previous_Fp
    dest.previous_p = src.previous_p
    copy_state!(dest.previous_Cv, src.previous_Cv)
    dest.track_σv && copy_state!(dest.previous_σv, src.previous_σv)
    return dest
end

is_linear(::VEPD_Detrez2010) = false

# --- 4. Phase stress helpers ---

@inline function _vepd_crystalline_stress(Ce::SymmetricTensor{2,3,T}, E, ν) where T
    μ, λ = lame_parameters(E, ν)
    Ee = 0.5 * (Ce - one(Ce))
    return λ * tr(Ee) * one(Ee) + 2.0 * μ * Ee
end

@inline function _vepd_network_stress(C::SymmetricTensor{2,3,T}, N, μ) where T
    J2 = max(det(C), 1e-12)
    I1 = J2^(-1 / 3) * tr(C)
    c1, c2, c3, c4, c5 = 1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750
    dW_dI1 = μ * (c1 + 2 * c2 / N * I1 + 3 * c3 / N^2 * I1^2 + 4 * c4 / N^3 * I1^3 + 5 * c5 / N^4 * I1^4)
    dI1_dC = J2^(-1 / 3) * (one(C) - (1 / 3) * tr(C) * inv(C))
    return 2.0 * dW_dI1 * dI1_dC
end

function _vepd_crystalline_S(Ce::SymmetricTensor{2,3,T}, mat) where T
    return _vepd_crystalline_stress(Ce, mat.E, mat.ν)
end

function _vepd_network_S(C::SymmetricTensor{2,3,T}, mat) where T
    return _vepd_network_stress(C, mat.N_ab, mat.μ_ab)
end

# --- 5. Main driver ---

function _vepd_plastic_end_step(F::Tensor{2,3,T}, mat::VEPD_Detrez2010, Fp_n, p_n) where T
    Δp = zero(T); p_new = p_n; Fp_new = Fp_n
    Fe_tr = F ⋅ inv(Fp_n); Ce_tr = symmetric(Fe_tr' ⋅ Fe_tr)
    Σ_tr = (Ce_tr ⋅ _vepd_crystalline_S(Ce_tr, mat)) / det(Fe_tr)
    Σ_eq_tr = sqrt(1.5 * (dev(Σ_tr) ⊡ dev(Σ_tr)))
    f_tr = Σ_eq_tr - (mat.R0 + mat.Q * (1.0 - exp(-mat.b * p_n)))

    if f_tr > 1e-9 * mat.E
        # Using linear update (1 + dP*N) which is stable for AD
        N = 1.5 * dev(Σ_tr) / Σ_eq_tr
        converged = false
        last_f = f_tr
        maxiter = 25
        for iter in 1:maxiter
            p_c = p_n + Δp; Fp_t = (one(F) + Δp * N) ⋅ Fp_n
            Fe_t = F ⋅ inv(Fp_t); Ce_t = symmetric(Fe_t' ⋅ Fe_t)
            Σ_t = (Ce_t ⋅ _vepd_crystalline_S(Ce_t, mat)) / det(Fe_t)
            Σ_eq_t = sqrt(1.5 * (dev(Σ_t) ⊡ dev(Σ_t)))
            f = Σ_eq_t - (mat.R0 + mat.Q * (1.0 - exp(-mat.b * p_c)))
            last_f = f
            if abs(f) < 1e-9 * mat.E
                converged = true
                break
            end
            μ_eff, _ = lame_parameters(mat.E, mat.ν)
            df = -3.0 * μ_eff - mat.Q * mat.b * exp(-mat.b * p_c)
            Δp -= f / df
        end
        converged || throw(VEPD_Detrez2010ConvergenceError(Δp, last_f, maxiter))
        p_new = p_n + Δp; Fp_new = (one(F) + Δp * N) ⋅ Fp_n
    end

    return Fp_new, p_new
end

function _vepd_plastic_update(F0, F::Tensor{2,3,T}, mat::VEPD_Detrez2010, Fp_n, p_n) where T
    mat.plastic_update == :end_step && return _vepd_plastic_end_step(F, mat, Fp_n, p_n)

    Fp = Fp_n
    p = p_n
    for i in 1:_DETREZ_PLASTIC_PATH_SUBSTEPS
        a = i / _DETREZ_PLASTIC_PATH_SUBSTEPS
        F_sub = (1.0 - a) * F0 + a * F
        Fp, p = _vepd_plastic_end_step(F_sub, mat, Fp, p)
    end
    return Fp, p
end

function _vepd_maxwell_closed_form_cv_update(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010, Cv_n) where T
    C_tot = symmetric(F' ⋅ F)
    Cv_new = Vector{SymmetricTensor{2,3,T}}(undef, length(mat.G))
    sigma_visco = zero(Tensor{2,3,T})
    for i in 1:length(mat.G)
        expt = exp(-dt / mat.τ[i])
        Cv_next = expt * Cv_n[i] + (1.0 - expt) * C_tot
        # Defensive reset if the branch state lost positive-definiteness.
        det(Cv_next) <= 0 && (Cv_next = one(Cv_next))
        σv_next = mat.G[i] * dev(F ⋅ inv(Cv_next) ⋅ F')
        sigma_visco += σv_next
        Cv_new[i] = Cv_next
    end
    return sigma_visco, Cv_new
end

function _vepd_objective_maxwell_rhs(σ, F, Fdot, G, τ)
    L = Fdot ⋅ inv(F)
    D = symmetric(L)
    return L ⋅ σ + σ ⋅ L' - σ / τ + G * dev(D)
end

function _vepd_objective_branch_update(F0, F1::Tensor{2,3,T}, dt, G, τ, σ0) where T
    σ = σ0 + zero(Tensor{2,3,T})
    h = dt / _DETREZ_OBJECTIVE_MAXWELL_SUBSTEPS
    Fdot = (F1 - F0) / dt
    for i in 0:(_DETREZ_OBJECTIVE_MAXWELL_SUBSTEPS - 1)
        a = i / _DETREZ_OBJECTIVE_MAXWELL_SUBSTEPS
        F = (1.0 - a) * F0 + a * F1
        k1 = _vepd_objective_maxwell_rhs(σ, F, Fdot, G, τ)
        k2 = _vepd_objective_maxwell_rhs(σ + 0.5 * h * k1, F + 0.5 * h * Fdot, Fdot, G, τ)
        k3 = _vepd_objective_maxwell_rhs(σ + 0.5 * h * k2, F + 0.5 * h * Fdot, Fdot, G, τ)
        k4 = _vepd_objective_maxwell_rhs(σ + h * k3, F + h * Fdot, Fdot, G, τ)
        σ += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    end
    return σ
end

function _vepd_maxwell_objective_update(F0, F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010, Cv_n, σv_n) where T
    _, Cv_new = _vepd_maxwell_closed_form_cv_update(F, dt, mat, Cv_n)
    σv_new = Vector{Tensor{2,3,T}}(undef, length(mat.G))
    sigma_visco = zero(Tensor{2,3,T})
    if iszero(dt)
        for i in 1:length(mat.G)
            σv_i = σv_n[i] + zero(Tensor{2,3,T})
            σv_new[i] = σv_i
            sigma_visco += σv_i
        end
        return sigma_visco, Cv_new, σv_new
    end

    for i in 1:length(mat.G)
        σv_i = _vepd_objective_branch_update(F0, F, dt, mat.G[i], mat.τ[i], σv_n[i])
        σv_new[i] = σv_i
        sigma_visco += σv_i
    end
    return sigma_visco, Cv_new, σv_new
end

function _vepd_maxwell_update(F0, F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010, Cv_n, σv_n) where T
    if mat.maxwell_update == :closed_form_cv
        sigma_visco, Cv_new = _vepd_maxwell_closed_form_cv_update(F, dt, mat, Cv_n)
        return sigma_visco, Cv_new, σv_n
    end
    return _vepd_maxwell_objective_update(F0, F, dt, mat, Cv_n, σv_n)
end

function _vepd_total_sigma(F::Tensor{2,3,T}, mat::VEPD_Detrez2010, Fp_new, p_new, sigma_visco) where T
    C_tot = symmetric(F' ⋅ F)
    J = det(F)
    sigma_net = dev((F ⋅ _vepd_network_S(C_tot, mat) ⋅ F') / J)
    D = min(1.0, mat.α * (1.0 - exp(-mat.β * p_new)))
    Fe_f = F ⋅ inv(Fp_new); Ce_f = symmetric(Fe_f' ⋅ Fe_f)
    sigma_cs = (1.0 - D) * (Fe_f ⋅ _vepd_crystalline_S(Ce_f, mat) ⋅ Fe_f') / det(Fe_f)
    return sigma_cs + sigma_net + sigma_visco
end

function _solve_local_vepd_default(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010, state::VEPD_Detrez2010State) where T
    Fp_new, p_new = _vepd_plastic_end_step(F, mat, state.previous_Fp, state.previous_p)
    sigma_visco, Cv_new = _vepd_maxwell_closed_form_cv_update(F, dt, mat, state.previous_Cv)
    return _vepd_total_sigma(F, mat, Fp_new, p_new, sigma_visco), F, Fp_new, p_new, Cv_new, state.previous_σv
end

function _solve_local_vepd_full(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010{:end_step,:closed_form_cv}, state::VEPD_Detrez2010State) where T
    return _solve_local_vepd_default(F, dt, mat, state)
end

function _solve_local_vepd_full(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010, state::VEPD_Detrez2010State) where T
    mat.plastic_update == :end_step && mat.maxwell_update == :closed_form_cv &&
        return _solve_local_vepd_default(F, dt, mat, state)

    F0 = state.previous_F
    Fp_n, p_n = state.previous_Fp, state.previous_p
    Cv_n, σv_n = state.previous_Cv, state.previous_σv

    Fp_new, p_new = _vepd_plastic_update(F0, F, mat, Fp_n, p_n)
    sigma_visco, Cv_new, σv_new = _vepd_maxwell_update(F0, F, dt, mat, Cv_n, σv_n)

    return _vepd_total_sigma(F, mat, Fp_new, p_new, sigma_visco), F, Fp_new, p_new, Cv_new, σv_new
end

function solve_local_vepd(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010, state::VEPD_Detrez2010State) where T
    sigma, _, Fp_new, p_new, Cv_new, _ = _solve_local_vepd_full(F, dt, mat, state)
    return sigma, Fp_new, p_new, Cv_new
end

function compute_vepd_PK1(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010, state::VEPD_Detrez2010State) where T
    sigma, _, _, _ = solve_local_vepd(F, dt, mat, state)
    return det(F) * sigma ⋅ inv(F)'
end

kinematics(::VEPD_Detrez2010) = FiniteStrain()

function material_response(mp::VEPD_Detrez2010, F::Tensor{2,3}, state::VEPD_Detrez2010State, dt, cache=nothing)
    sigma, F_v, Fp_v, p_v, Cv_v, σv_v = _solve_local_vepd_full(F, dt, mp, state)
    P = det(F) * sigma ⋅ inv(F)'
    dP_dF = Tensors.gradient(F_ -> compute_vepd_PK1(F_, dt, mp, state), F)
    return P, dP_dF, (F=F_v, Fp=Fp_v, p=p_v, Cv=Cv_v, σv=σv_v)
end

function set_trial!(state::VEPD_Detrez2010State, new::NamedTuple)
    state.current_F = new.F
    state.current_Fp = new.Fp
    state.current_p = new.p
    for i in eachindex(state.current_Cv)
        state.current_Cv[i] = new.Cv[i]
    end
    for i in eachindex(state.current_σv)
        state.current_σv[i] = new.σv[i]
    end
    return nothing
end

function material_stress(mp::VEPD_Detrez2010, F::Tensor{2,3}, state::VEPD_Detrez2010State, dt, cache=nothing)
    current = typeof(state)(
        state.current_F,
        state.current_Fp,
        state.current_p,
        state.current_Cv,
        state.current_σv,
        state.current_F,
        state.current_Fp,
        state.current_p,
        state.current_Cv,
        state.current_σv,
        state.track_σv,
    )
    sigma, _, _, _ = solve_local_vepd(F, 0.0, mp, current)  # pass dt=0.0 to not advance history in postprocessing (!)
    return sigma
end

# 2D wrapper interface for VEPD_Detrez2010
function compute_PK1_3D(mp::VEPD_Detrez2010, F::Tensor{2,3,T}, dt, state::VEPD_Detrez2010State) where T
    return compute_vepd_PK1(F, dt, mp, state)
end

function update_state_from_3D!(state::VEPD_Detrez2010State, mp::VEPD_Detrez2010, F::Tensor{2,3}, dt)
    _, F_v, Fp_v, p_v, Cv_v, σv_v = _solve_local_vepd_full(F, dt, mp, state)
    state.current_F = Tensors.value(F_v)
    state.current_Fp = Tensors.value(Fp_v)
    state.current_p = Tensors.value(p_v)
    for i in eachindex(state.current_Cv)
        state.current_Cv[i] = Tensors.value(Cv_v[i])
    end
    for i in eachindex(state.current_σv)
        state.current_σv[i] = Tensors.value(σv_v[i])
    end
    return nothing
end
