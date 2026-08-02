using Tensors, LinearAlgebra

struct VEPD_Detrez2010_ExactVisco <: AbstractMaterial
    E::Float64; ν::Float64; R0::Float64; Q::Float64; b::Float64
    α::Float64; β::Float64; N_ab::Float64; μ_ab::Float64
    G::Vector{Float64}; τ::Vector{Float64}

    function VEPD_Detrez2010_ExactVisco(;E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
        new(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
    end
end

mutable struct VEPD_Detrez2010_ExactViscoState <: AbstractMaterialState
    current_Fp::Tensor{2,3,Float64,9}
    current_p::Float64
    current_Cv::Vector{SymmetricTensor{2,3,Float64,6}}
    previous_Fp::Tensor{2,3,Float64,9}
    previous_p::Float64
    previous_Cv::Vector{SymmetricTensor{2,3,Float64,6}}
end

function create_state(mat::VEPD_Detrez2010_ExactVisco)
    Fp = one(Tensor{2,3}); p = 0.0
    Cv = [one(SymmetricTensor{2,3}) for _ in 1:length(mat.G)]
    return VEPD_Detrez2010_ExactViscoState(Fp, p, Cv, Fp, p, Cv)
end

function update_state!(state::VEPD_Detrez2010_ExactViscoState)
    state.previous_Fp = state.current_Fp
    state.previous_p = state.current_p
    copy_state!(state.previous_Cv, state.current_Cv)
end

is_linear(::VEPD_Detrez2010_ExactVisco) = false

function _vepd_crystalline_S_exact(Ce::SymmetricTensor{2,3,T}, mat::VEPD_Detrez2010_ExactVisco) where T
    return _vepd_crystalline_stress(Ce, mat.E, mat.ν)
end

function solve_local_vepd_exact_visco(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010_ExactVisco, state::VEPD_Detrez2010_ExactViscoState) where T
    Fp_n = state.previous_Fp; p_n = state.previous_p; Cv_n = state.previous_Cv

    # 1. Plasticity Update (Implicit Return Mapping - Unchanged structure)
    Δp = zero(T); p_new = p_n; Fp_new = Fp_n
    Fe_trial = F ⋅ inv(Fp_n); Ce_trial = symmetric(Fe_trial' ⋅ Fe_trial)
    Σ_trial = (Ce_trial ⋅ _vepd_crystalline_S_exact(Ce_trial, mat)) / det(Fe_trial)
    Σ_eq_trial = sqrt(1.5 * (dev(Σ_trial) ⊡ dev(Σ_trial)))
    f_trial = Σ_eq_trial - (mat.R0 + mat.Q * (1.0 - exp(-mat.b * p_n)))

    if f_trial > 1e-9 * mat.E
        N = 1.5 * dev(Σ_trial) / Σ_eq_trial
        for _ in 1:25
            p_curr = p_n + Δp
            Fp_tmp = (one(F) + Δp * N) ⋅ Fp_n
            Fe_tmp = F ⋅ inv(Fp_tmp); Ce_tmp = symmetric(Fe_tmp' ⋅ Fe_tmp)
            Σ_inner = (Ce_tmp ⋅ _vepd_crystalline_S_exact(Ce_tmp, mat)) / det(Fe_tmp)
            Σ_eq_tmp = sqrt(1.5 * (dev(Σ_inner) ⊡ dev(Σ_inner)))
            f = Σ_eq_tmp - (mat.R0 + mat.Q * (1.0 - exp(-mat.b * p_curr)))
            if abs(f) < 1e-9 * mat.E break end
            μ_eff, _ = lame_parameters(mat.E, mat.ν)
            df = -3.0 * μ_eff - mat.Q * mat.b * exp(-mat.b * p_curr)
            Δp -= f / df
        end
        p_new = p_n + Δp; Fp_new = (one(F) + Δp * N) ⋅ Fp_n
    end

    # 2. EXACT Viscoelastic Update (These.c-like exponential update on Cᵥ)
    C_tot = symmetric(F' ⋅ F)
    Cv_new = Vector{SymmetricTensor{2,3,T}}(undef, length(mat.G))
    for i in 1:length(mat.G)
        expt = exp(-dt / mat.τ[i])
        Cv_next = expt * Cv_n[i] + (1.0 - expt) * C_tot
        # Defensive reset if the branch state lost positive-definiteness.
        det(Cv_next) <= 0 && (Cv_next = one(Cv_next))
        Cv_new[i] = Cv_next
    end

    # 3. Stress
    D_new = mat.α * (1.0 - exp(-mat.β * p_new))
    Fe_fin = F ⋅ inv(Fp_new)
    sigma_cs = (1.0 - D_new) * (Fe_fin ⋅ _vepd_crystalline_S_exact(symmetric(Fe_fin' ⋅ Fe_fin), mat) ⋅ Fe_fin') / det(Fe_fin)

    sigma_visco = zero(Tensor{2,3,T})
    for i in 1:length(mat.G)
        sigma_visco += mat.G[i] * dev(F ⋅ inv(Cv_new[i]) ⋅ F')
    end

    return sigma_cs + sigma_visco, Fp_new, p_new, Cv_new
end

function compute_vepd_PK1_Exact(F, dt, mat::VEPD_Detrez2010_ExactVisco, state::VEPD_Detrez2010_ExactViscoState)
    sigma, _, _, _ = solve_local_vepd_exact_visco(F, dt, mat, state)
    return det(F) * sigma ⋅ inv(F)'
end

function _assemble_element!(ke, re, states::Vector{<:AbstractMaterialState}, mp::VEPD_Detrez2010_ExactVisco, cellvalues, alphavalues, u, dt)
    for qp in 1:getnquadpoints(cellvalues)
        state = states[qp]::VEPD_Detrez2010_ExactViscoState
        α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)
        F = deformation_gradient(cellvalues, qp, u)

        P = compute_vepd_PK1_Exact(F, dt, mp, state)
        dP_dF = Tensors.gradient(F_ -> compute_vepd_PK1_Exact(F_, dt, mp, state), F)

        _, Fp_v, p_v, Cv_v = solve_local_vepd_exact_visco(Tensors.value(F), dt, mp, state)
        state.current_Fp = Tensors.value(Fp_v); state.current_p = Tensors.value(p_v); state.current_Cv = Tensors.value.(Cv_v)

        assemble_pk1_tangent!(ke, re, P, dP_dF, cellvalues, qp, α_dΩ)
    end
end

function _compute_stress_qp(mp::VEPD_Detrez2010_ExactVisco, cellvalues, alphavalues, qp, u_local, state::VEPD_Detrez2010_ExactViscoState, dt=0.0)
    F = deformation_gradient(cellvalues, qp, u_local)
    sigma, _, _, _ = solve_local_vepd_exact_visco(F, dt, mp, state)
    return sigma
end

# 2D wrapper interface for VEPD_Detrez2010_ExactVisco
function compute_PK1_3D(mp::VEPD_Detrez2010_ExactVisco, F::Tensor{2,3,T}, dt, state::VEPD_Detrez2010_ExactViscoState) where T
    return compute_vepd_PK1_Exact(F, dt, mp, state)
end

function update_state_from_3D!(state::VEPD_Detrez2010_ExactViscoState, mp::VEPD_Detrez2010_ExactVisco, F::Tensor{2,3}, dt)
    _, Fp_v, p_v, Cv_v = solve_local_vepd_exact_visco(F, dt, mp, state)
    state.current_Fp = Tensors.value(Fp_v)
    state.current_p = Tensors.value(p_v)
    state.current_Cv = Tensors.value.(Cv_v)
    return nothing
end
