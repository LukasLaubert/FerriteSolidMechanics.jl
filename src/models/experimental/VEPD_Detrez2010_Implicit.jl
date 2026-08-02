using Tensors
using LinearAlgebra

struct VEPD_Detrez2010_Implicit <: AbstractMaterial
    # Crystalline Phase (St. Venant–Kirchhoff)
    E::Float64
    ν::Float64
    R0::Float64
    Q::Float64
    b::Float64
    # Damage
    α::Float64
    β::Float64
    # Network (Arruda–Boyce)
    N_ab::Float64
    μ_ab::Float64
    # Viscous Branches (Maxwell)
    G::Vector{Float64}
    τ::Vector{Float64}

    function VEPD_Detrez2010_Implicit(;E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
        new(E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
    end
end

mutable struct VEPD_Detrez2010_ImplicitState <: AbstractMaterialState
    # Converged and Trial state variables
    current_Fp::Tensor{2,3,Float64,9}
    current_p::Float64
    current_Cv::Vector{SymmetricTensor{2,3,Float64,6}}

    previous_Fp::Tensor{2,3,Float64,9}
    previous_p::Float64
    previous_Cv::Vector{SymmetricTensor{2,3,Float64,6}}
end

function create_state(mat::VEPD_Detrez2010_Implicit)
    Fp = one(Tensor{2,3})
    p = 0.0
    Cv = [one(SymmetricTensor{2,3}) for _ in 1:length(mat.G)]
    return VEPD_Detrez2010_ImplicitState(Fp, p, Cv, Fp, p, Cv)
end

function update_state!(state::VEPD_Detrez2010_ImplicitState)
    state.previous_Fp = state.current_Fp
    state.previous_p = state.current_p
    copy_state!(state.previous_Cv, state.current_Cv)
end

is_linear(::VEPD_Detrez2010_Implicit) = false

# --- Potentials ---

function _vepd_crystalline_S_implicit(Ce::SymmetricTensor{2,3,T}, mat::VEPD_Detrez2010_Implicit) where T
    return _vepd_crystalline_stress(Ce, mat.E, mat.ν)
end

function _vepd_network_S_implicit(C::SymmetricTensor{2,3,T}, mat::VEPD_Detrez2010_Implicit) where T
    return _vepd_network_stress(C, mat.N_ab, mat.μ_ab)
end

# --- Local constitutive driver ---

function solve_local_vepd_implicit(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010_Implicit, state::VEPD_Detrez2010_ImplicitState) where T
    Fp_n = state.previous_Fp
    p_n = state.previous_p
    Cv_n = state.previous_Cv

    # 1. Plasticity Update
    Δp = zero(T)
    p_new = p_n
    Fp_new = Fp_n

    Fe_trial = F ⋅ inv(Fp_n)
    Ce_trial = symmetric(Fe_trial' ⋅ Fe_trial)
    Σ_trial = (Ce_trial ⋅ _vepd_crystalline_S_implicit(Ce_trial, mat)) / det(Fe_trial)
    Σ_eq_trial = sqrt(1.5 * (dev(Σ_trial) ⊡ dev(Σ_trial)))

    f_trial = Σ_eq_trial - (mat.R0 + mat.Q * (1.0 - exp(-mat.b * p_n)))

    if f_trial > 1e-9 * mat.E
        N = 1.5 * dev(Σ_trial) / Σ_eq_trial
        for _ in 1:25
            p_curr = p_n + Δp
            Fp_tmp = (one(F) + Δp * N) ⋅ Fp_n
            Fe_tmp = F ⋅ inv(Fp_tmp)
            Ce_tmp = symmetric(Fe_tmp' ⋅ Fe_tmp)
            Σ_inner = (Ce_tmp ⋅ _vepd_crystalline_S_implicit(Ce_tmp, mat)) / det(Fe_tmp)
            Σ_eq_tmp = sqrt(1.5 * (dev(Σ_inner) ⊡ dev(Σ_inner)))
            R = mat.Q * (1.0 - exp(-mat.b * p_curr))
            f = Σ_eq_tmp - (mat.R0 + R)
            if abs(f) < 1e-9 * mat.E break end
            μ_eff, _ = lame_parameters(mat.E, mat.ν)
            df = -3.0 * μ_eff - mat.Q * mat.b * exp(-mat.b * p_curr)
            Δp -= f / df
        end
        p_new = p_n + Δp
        Fp_new = (one(F) + Δp * N) ⋅ Fp_n
    end

    # 2. Network Hyperelastic & Viscous
    J = det(F)
    C_tot = symmetric(F' ⋅ F)
    sigma_net0 = dev((F ⋅ _vepd_network_S_implicit(C_tot, mat) ⋅ F') / J)

    sigma_visco = zero(Tensor{2,3,T})
    Cv_new = Vector{SymmetricTensor{2,3,T}}(undef, length(mat.G))
    for i in 1:length(mat.G)
        coeff = 2.0 * dt / mat.τ[i]
        Cv_next = (Cv_n[i] + coeff * C_tot) / (1.0 + coeff)
        # Defensive reset if the branch state lost positive-definiteness.
        det(Cv_next) <= 0 && (Cv_next = one(Cv_next))
        sigma_visco += mat.G[i] * dev(F ⋅ inv(Cv_next) ⋅ F')
        Cv_new[i] = Cv_next
    end

    # 3. Damaged Crystalline Stress
    D_new = mat.α * (1.0 - exp(-mat.β * p_new))
    Fe_fin = F ⋅ inv(Fp_new)
    sigma_cs = (1.0 - D_new) * (Fe_fin ⋅ _vepd_crystalline_S_implicit(symmetric(Fe_fin' ⋅ Fe_fin), mat) ⋅ Fe_fin') / det(Fe_fin)

    return sigma_cs + sigma_net0 + sigma_visco, Fp_new, p_new, Cv_new
end

# --- PK1 and AD helpers ---

function compute_vepd_PK1_implicit(F::Tensor{2,3,T}, dt, mat::VEPD_Detrez2010_Implicit, state::VEPD_Detrez2010_ImplicitState) where T
    sigma, _, _, _ = solve_local_vepd_implicit(F, dt, mat, state)
    return det(F) * sigma ⋅ inv(F)'
end

function _assemble_element!(ke, re, states::Vector{<:AbstractMaterialState}, mp::VEPD_Detrez2010_Implicit, cellvalues, alphavalues, u, dt)
    for qp in 1:getnquadpoints(cellvalues)
        state = states[qp]::VEPD_Detrez2010_ImplicitState
        α_dΩ = alpha_value(alphavalues, qp) * getdetJdV(cellvalues, qp)
        F = deformation_gradient(cellvalues, qp, u)

        P = compute_vepd_PK1_implicit(F, dt, mp, state)
        dP_dF = Tensors.gradient(F_ -> compute_vepd_PK1_implicit(F_, dt, mp, state), F)

        _, Fp_v, p_v, Cv_v = solve_local_vepd_implicit(Tensors.value(F), dt, mp, state)
        state.current_Fp = Tensors.value(Fp_v); state.current_p = Tensors.value(p_v); state.current_Cv = Tensors.value.(Cv_v)

        assemble_pk1_tangent!(ke, re, P, dP_dF, cellvalues, qp, α_dΩ)
    end
end

function _compute_stress_qp(mp::VEPD_Detrez2010_Implicit, cellvalues, alphavalues, qp, u_local, state::VEPD_Detrez2010_ImplicitState, dt=0.0)
    F = deformation_gradient(cellvalues, qp, u_local)
    sigma, _, _, _ = solve_local_vepd_implicit(F, dt, mp, state)
    return sigma
end

# 2D wrapper interface for VEPD_Detrez2010_Implicit
function compute_PK1_3D(mp::VEPD_Detrez2010_Implicit, F::Tensor{2,3,T}, dt, state::VEPD_Detrez2010_ImplicitState) where T
    return compute_vepd_PK1_implicit(F, dt, mp, state)
end

function update_state_from_3D!(state::VEPD_Detrez2010_ImplicitState, mp::VEPD_Detrez2010_Implicit, F::Tensor{2,3}, dt)
    _, Fp_v, p_v, Cv_v = solve_local_vepd_implicit(F, dt, mp, state)
    state.current_Fp = Tensors.value(Fp_v)
    state.current_p = Tensors.value(p_v)
    state.current_Cv = Tensors.value.(Cv_v)
    return nothing
end
