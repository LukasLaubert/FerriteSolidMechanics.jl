using Tensors
using LinearAlgebra

# --- Truly optimized exact model ---
# Same variable names as the production VEPD_Detrez2010, optimized for speed.
struct VEPD_Detrez2010_Optimized <: AbstractMaterial
    E::Float64; ν::Float64; R0::Float64; Q::Float64; b::Float64
    α::Float64; β::Float64; N_ab::Float64; μ_ab::Float64
    G::Vector{Float64}; τ::Vector{Float64}

    function VEPD_Detrez2010_Optimized(;E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
        new(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
    end
end

mutable struct VEPD_Detrez2010_OptimizedState <: AbstractMaterialState
    current_Fp::Tensor{2,3,Float64,9}
    current_p::Float64
    current_Cv::Vector{SymmetricTensor{2,3,Float64,6}}

    previous_Fp::Tensor{2,3,Float64,9}
    previous_p::Float64
    previous_Cv::Vector{SymmetricTensor{2,3,Float64,6}}
end

function create_state(mat::VEPD_Detrez2010_Optimized)
    Fp = one(Tensor{2,3}); p = 0.0
    Cv = [one(SymmetricTensor{2,3}) for _ in 1:length(mat.G)]
    return VEPD_Detrez2010_OptimizedState(Fp, p, Cv, Fp, p, Cv)
end

function update_state!(state::VEPD_Detrez2010_OptimizedState)
    state.previous_Fp = state.current_Fp
    state.previous_p = state.current_p
    copy_state!(state.previous_Cv, state.current_Cv)
end

function _vepd_crystalline_S_opt(Ce::SymmetricTensor{2,3,T}, mat::VEPD_Detrez2010_Optimized) where T
    return _vepd_crystalline_stress(Ce, mat.E, mat.ν)
end

function _vepd_network_S_opt(C::SymmetricTensor{2,3,T}, mat::VEPD_Detrez2010_Optimized) where T
    return _vepd_network_stress(C, mat.N_ab, mat.μ_ab)
end

function solve_local_vepd_opt(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010_Optimized, state::VEPD_Detrez2010_OptimizedState) where T
    Fp_n, p_n, Cv_n = state.previous_Fp, state.previous_p, state.previous_Cv

    # 1. Plasticity (Correct 3.0 Jacobian)
    Δp = zero(T); p_new = p_n; Fp_new = Fp_n
    Fe_tr = F ⋅ inv(Fp_n); Ce_tr = symmetric(Fe_tr' ⋅ Fe_tr)
    Σ_tr = (Ce_tr ⋅ _vepd_crystalline_S_opt(Ce_tr, mat)) / det(Fe_tr)
    Σ_eq_tr = sqrt(1.5 * (dev(Σ_tr) ⊡ dev(Σ_tr)))
    f_tr = Σ_eq_tr - (mat.R0 + mat.Q * (1.0 - exp(-mat.b * p_n)))

    if f_tr > 1e-9 * mat.E
        N = 1.5 * dev(Σ_tr) / Σ_eq_tr
        for _ in 1:25
            p_c = p_n + Δp; Fp_t = (one(F) + Δp * N) ⋅ Fp_n
            Fe_t = F ⋅ inv(Fp_t); Ce_t = symmetric(Fe_t' ⋅ Fe_t)
            Σ_t = (Ce_t ⋅ _vepd_crystalline_S_opt(Ce_t, mat)) / det(Fe_t)
            Σ_eq_t = sqrt(1.5 * (dev(Σ_t) ⊡ dev(Σ_t)))
            f = Σ_eq_t - (mat.R0 + mat.Q * (1.0 - exp(-mat.b * p_c)))
            if abs(f) < 1e-9 * mat.E break end
            μ_eff, _ = lame_parameters(mat.E, mat.ν)
            df = -3.0 * μ_eff - mat.Q * mat.b * exp(-mat.b * p_c)
            Δp -= f / df
        end
        p_new = p_n + Δp; Fp_new = (one(F) + Δp * N) ⋅ Fp_n
    end

    # 2. EXACT Viscoelasticity (Exponential update)
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
    sigma_net = dev((F ⋅ _vepd_network_S_opt(C_tot, mat) ⋅ F') / J)
    D = min(1.0, mat.α * (1.0 - exp(-mat.β * p_new)))
    Fe_f = F ⋅ inv(Fp_new); Ce_f = symmetric(Fe_f' ⋅ Fe_f)
    sigma_cs = (1.0 - D) * (Fe_f ⋅ _vepd_crystalline_S_opt(Ce_f, mat) ⋅ Fe_f') / det(Fe_f)

    return sigma_cs + sigma_net + sigma_visco, Fp_new, p_new, Cv_new
end

function compute_vepd_PK1_Opt(F, dt, mat::VEPD_Detrez2010_Optimized, state::VEPD_Detrez2010_OptimizedState)
    sigma, _, _, _ = solve_local_vepd_opt(F, dt, mat, state)
    return det(F) * sigma ⋅ inv(F)'
end

function _assemble_element!(ke, re, states::Vector{<:AbstractMaterialState}, mp::VEPD_Detrez2010_Optimized, cellvalues, alphavalues, u, dt)
    for qp in 1:getnquadpoints(cellvalues)
        state = states[qp]::VEPD_Detrez2010_OptimizedState
        α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)
        F = deformation_gradient(cellvalues, qp, u)

        P = compute_vepd_PK1_Opt(F, dt, mp, state)
        dP_dF = Tensors.gradient(F_ -> compute_vepd_PK1_Opt(F_, dt, mp, state), F)

        _, Fp_v, p_v, Cv_v = solve_local_vepd_opt(Tensors.value(F), dt, mp, state)
        state.current_Fp = Tensors.value(Fp_v); state.current_p = Tensors.value(p_v); state.current_Cv = Tensors.value.(Cv_v)

        assemble_pk1_tangent!(ke, re, P, dP_dF, cellvalues, qp, α_dΩ)
    end
end

function _compute_stress_qp(mp::VEPD_Detrez2010_Optimized, cellvalues, alphavalues, qp, u_local, state::VEPD_Detrez2010_OptimizedState, dt=0.0)
    F = deformation_gradient(cellvalues, qp, u_local)
    sigma, _, _, _ = solve_local_vepd_opt(F, dt, mp, state)
    return sigma
end

# 2D wrapper interface for VEPD_Detrez2010_Optimized
function compute_PK1_3D(mp::VEPD_Detrez2010_Optimized, F::Tensor{2,3,T}, dt, state::VEPD_Detrez2010_OptimizedState) where T
    return compute_vepd_PK1_Opt(F, dt, mp, state)
end

function update_state_from_3D!(state::VEPD_Detrez2010_OptimizedState, mp::VEPD_Detrez2010_Optimized, F::Tensor{2,3}, dt)
    _, Fp_v, p_v, Cv_v = solve_local_vepd_opt(F, dt, mp, state)
    state.current_Fp = Tensors.value(Fp_v)
    state.current_p = Tensors.value(p_v)
    state.current_Cv = Tensors.value.(Cv_v)
    return nothing
end
