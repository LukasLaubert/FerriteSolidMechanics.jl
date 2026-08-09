# --- 1. Material struct ---

"""
    VEPD_Detrez2010_ClosedCVEndStep(; E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)

Viscoelastic-plastic-damage material model after Detrez.
Combines crystalline elasticity, network hyperelasticity, closed-form Maxwell-branch updates, and isotropic hardening with damage.

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

# References

- F. Detrez, S. Cantournet, R. Séguéla.
  *A constitutive model for semi-crystalline polymer deformation involving lamellar fragmentation.*
  Comptes Rendus Mécanique **338**(12) (2010) 681–687.
  <https://doi.org/10.1016/j.crme.2010.10.008>

The code-level reference for this implementation is an unpublished C routine that was kindly provided by Fabrice Detrez.
"""
struct VEPD_Detrez2010_ClosedCVEndStep <: AbstractMaterial
    E::Float64; ν::Float64; R0::Float64; Q::Float64; b::Float64
    α::Float64; β::Float64; N_ab::Float64; μ_ab::Float64
    G::Vector{Float64}; τ::Vector{Float64}

    function VEPD_Detrez2010_ClosedCVEndStep(;E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
        length(G) == length(τ) || throw(ArgumentError("VEPD_Detrez2010_ClosedCVEndStep: G and τ length mismatch"))
        all(>(0.0), τ) || throw(ArgumentError("VEPD_Detrez2010_ClosedCVEndStep: τ must be positive"))
        new(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
    end
end

VEPD_Detrez2010_ClosedCVEndStep(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ) =
    VEPD_Detrez2010_ClosedCVEndStep(; E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)

# --- 2. Convergence error and state struct ---

"""
    VEPD_Detrez2010_ClosedCVEndStepConvergenceError

Recoverable failure raised when the local VEPD Detrez plastic correction does not converge within its fixed Newton iteration budget.
"""
struct VEPD_Detrez2010_ClosedCVEndStepConvergenceError <: LocalAssemblyFailure
    delta_p
    residual
    iterations::Int
end

function Base.showerror(io::IO, err::VEPD_Detrez2010_ClosedCVEndStepConvergenceError)
    print(io, "VEPD_Detrez2010_ClosedCVEndStep local plastic solve failed to converge")
    print(io, ": delta_p=", err.delta_p)
    print(io, ", residual=", err.residual)
    print(io, ", iterations=", err.iterations)
end

mutable struct VEPD_Detrez2010_ClosedCVEndStepState <: AbstractMaterialState
    current_Fp::Tensor{2,3,Float64,9}
    current_p::Float64
    current_Cv::Vector{SymmetricTensor{2,3,Float64,6}}

    previous_Fp::Tensor{2,3,Float64,9}
    previous_p::Float64
    previous_Cv::Vector{SymmetricTensor{2,3,Float64,6}}
end

# --- 3. API hooks ---

function create_state(mat::VEPD_Detrez2010_ClosedCVEndStep)
    Fp = one(Tensor{2,3}); p = 0.0
    Cv = [one(SymmetricTensor{2,3}) for _ in 1:length(mat.G)]
    return VEPD_Detrez2010_ClosedCVEndStepState(Fp, p, copy(Cv), Fp, p, copy(Cv))
end

function update_state!(state::VEPD_Detrez2010_ClosedCVEndStepState)
    state.previous_Fp = state.current_Fp
    state.previous_p = state.current_p
    copy_state!(state.previous_Cv, state.current_Cv)
end

function revert_state!(state::VEPD_Detrez2010_ClosedCVEndStepState)
    state.current_Fp = state.previous_Fp; state.current_p = state.previous_p;
    copy_state!(state.current_Cv, state.previous_Cv)
end

function copy_state!(dest::VEPD_Detrez2010_ClosedCVEndStepState, src::VEPD_Detrez2010_ClosedCVEndStepState)
    dest.current_Fp = src.current_Fp
    dest.current_p = src.current_p
    copy_state!(dest.current_Cv, src.current_Cv)
    dest.previous_Fp = src.previous_Fp
    dest.previous_p = src.previous_p
    copy_state!(dest.previous_Cv, src.previous_Cv)
    return dest
end

is_linear(::VEPD_Detrez2010_ClosedCVEndStep) = false

# --- 4. Phase stress helpers ---

@inline function _vepd_crystalline_stress_closedcv_endstep(Ce::SymmetricTensor{2,3,T}, E, ν) where T
    μ, λ = lame_parameters(E, ν)
    Ee = 0.5 * (Ce - one(Ce))
    return λ * tr(Ee) * one(Ee) + 2.0 * μ * Ee
end

@inline function _vepd_network_stress_closedcv_endstep(C::SymmetricTensor{2,3,T}, N, μ) where T
    J2 = max(det(C), 1e-12)
    I1 = J2^(-1 / 3) * tr(C)
    c1, c2, c3, c4, c5 = 1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750
    dW_dI1 = μ * (c1 + 2 * c2 / N * I1 + 3 * c3 / N^2 * I1^2 + 4 * c4 / N^3 * I1^3 + 5 * c5 / N^4 * I1^4)
    dI1_dC = J2^(-1 / 3) * (one(C) - (1 / 3) * tr(C) * inv(C))
    return 2.0 * dW_dI1 * dI1_dC
end

function _vepd_crystalline_S(Ce::SymmetricTensor{2,3,T}, mat) where T
    return _vepd_crystalline_stress_closedcv_endstep(Ce, mat.E, mat.ν)
end

function _vepd_network_S(C::SymmetricTensor{2,3,T}, mat) where T
    return _vepd_network_stress_closedcv_endstep(C, mat.N_ab, mat.μ_ab)
end

# --- 5. Main driver ---

function solve_local_vepd(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010_ClosedCVEndStep, state::VEPD_Detrez2010_ClosedCVEndStepState) where T
    Fp_n, p_n, Cv_n = state.previous_Fp, state.previous_p, state.previous_Cv

    # 1. Plasticity (Exponential update for stability in AD)
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
        converged || throw(VEPD_Detrez2010_ClosedCVEndStepConvergenceError(Δp, last_f, maxiter))
        p_new = p_n + Δp; Fp_new = (one(F) + Δp * N) ⋅ Fp_n
    end

    # 2. Maxwell-branch update (closed-form end-step update)
    C_tot = symmetric(F' ⋅ F)
    Cv_new = Vector{SymmetricTensor{2,3,T}}(undef, length(mat.G))
    sigma_visco = zero(Tensor{2,3,T})
    for i in 1:length(mat.G)
        expt = exp(-dt / mat.τ[i])
        Cv_next = expt * Cv_n[i] + (1.0 - expt) * C_tot
        # Defensive reset if the branch state lost positive-definiteness.
        det(Cv_next) <= 0 && (Cv_next = one(Cv_next))
        sigma_visco += mat.G[i] * dev(F ⋅ inv(Cv_next) ⋅ F')
        Cv_new[i] = Cv_next
    end

    # 3. Network & Crystalline
    J = det(F)
    sigma_net = dev((F ⋅ _vepd_network_S(C_tot, mat) ⋅ F') / J)
    D = min(1.0, mat.α * (1.0 - exp(-mat.β * p_new)))
    Fe_f = F ⋅ inv(Fp_new); Ce_f = symmetric(Fe_f' ⋅ Fe_f)
    sigma_cs = (1.0 - D) * (Fe_f ⋅ _vepd_crystalline_S(Ce_f, mat) ⋅ Fe_f') / det(Fe_f)

    return sigma_cs + sigma_net + sigma_visco, Fp_new, p_new, Cv_new
end

function compute_vepd_PK1(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010_ClosedCVEndStep, state::VEPD_Detrez2010_ClosedCVEndStepState) where T
    sigma, _, _, _ = solve_local_vepd(F, dt, mat, state)
    return det(F) * sigma ⋅ inv(F)'
end

function _assemble_element!(ke, re, states::Vector{<:AbstractMaterialState}, mp::VEPD_Detrez2010_ClosedCVEndStep, cellvalues, alphavalues, u, dt)
    for qp in 1:getnquadpoints(cellvalues)
        state = states[qp]::VEPD_Detrez2010_ClosedCVEndStepState
        α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)
        F = deformation_gradient(cellvalues, qp, u)
        F_val = Tensors.value(F)

        sigma, Fp_v, p_v, Cv_v = solve_local_vepd(F_val, dt, mp, state)
        P = det(F_val) * sigma ⋅ inv(F_val)'
        dP_dF = Tensors.gradient(F_ -> compute_vepd_PK1(F_, dt, mp, state), F)

        state.current_Fp = Fp_v; state.current_p = p_v
        for i in eachindex(state.current_Cv)
            state.current_Cv[i] = Cv_v[i]
        end

        assemble_pk1_tangent!(ke, re, P, dP_dF, cellvalues, qp, α_dΩ)
    end
end

function _compute_stress_qp(mp::VEPD_Detrez2010_ClosedCVEndStep, cellvalues, alphavalues, qp, u_local, state::VEPD_Detrez2010_ClosedCVEndStepState, dt=0.0)
    F = deformation_gradient(cellvalues, qp, u_local)
    current = VEPD_Detrez2010_ClosedCVEndStepState(state.current_Fp, state.current_p, state.current_Cv, state.current_Fp, state.current_p, state.current_Cv)
    sigma, _, _, _ = solve_local_vepd(F, 0.0, mp, current)
    return sigma
end

# 2D wrapper interface for VEPD_Detrez2010_ClosedCVEndStep
function compute_PK1_3D(mp::VEPD_Detrez2010_ClosedCVEndStep, F::Tensor{2,3,T}, dt, state::VEPD_Detrez2010_ClosedCVEndStepState) where T
    return compute_vepd_PK1(F, dt, mp, state)
end

function update_state_from_3D!(state::VEPD_Detrez2010_ClosedCVEndStepState, mp::VEPD_Detrez2010_ClosedCVEndStep, F::Tensor{2,3}, dt)
    _, Fp_v, p_v, Cv_v = solve_local_vepd(F, dt, mp, state)
    state.current_Fp = Tensors.value(Fp_v)
    state.current_p = Tensors.value(p_v)
    for i in eachindex(state.current_Cv)
        state.current_Cv[i] = Tensors.value(Cv_v[i])
    end
    return nothing
end
