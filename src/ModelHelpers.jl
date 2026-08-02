@inline function lame_parameters(E, nu)
    mu = 0.5 * E / (1 + nu)
    lambda = E * nu / ((1 + nu) * (1 - 2nu))
    return mu, lambda
end

@inline function assemble_pk1_tangent!(ke, re, P, dP_dF, cellvalues, qp, alpha_dOmega)
    n_basefuncs = getnbasefunctions(cellvalues)

    if re !== nothing && ke !== nothing
        @inbounds for i in 1:n_basefuncs
            grad_i = shape_gradient(cellvalues, qp, i)
            re[i] += (grad_i ⊡ P) * alpha_dOmega
            grad_i_dP_dF = grad_i ⊡ dP_dF
            for j in 1:n_basefuncs
                ke[i, j] += (grad_i_dP_dF ⊡ shape_gradient(cellvalues, qp, j)) * alpha_dOmega
            end
        end
    elseif re !== nothing
        @inbounds for i in 1:n_basefuncs
            re[i] += (shape_gradient(cellvalues, qp, i) ⊡ P) * alpha_dOmega
        end
    elseif ke !== nothing
        @inbounds for i in 1:n_basefuncs
            grad_i_dP_dF = shape_gradient(cellvalues, qp, i) ⊡ dP_dF
            for j in 1:n_basefuncs
                ke[i, j] += (grad_i_dP_dF ⊡ shape_gradient(cellvalues, qp, j)) * alpha_dOmega
            end
        end
    end

    return nothing
end

function _finite_strain_pk1_tangent end
function _second_piola end

@inline function _pk1_from_second_piola(F::Tensor{2,dim,T}, mp) where {dim,T}
    C = tdot(F)
    S = _second_piola(C, mp)
    return F ⋅ S
end

@inline function _pk1_tangent_from_second_piola(F::Tensor{2,dim,T}, mp) where {dim,T}
    C = tdot(F)
    S, dS_dC = constitutive_driver(C, mp)
    return _pk1_tangent_from_second_piola(F, S, dS_dC)
end

@inline function _pk1_tangent_from_second_piola(F::Tensor{2,dim,T}, S, dS_dC) where {dim,T}
    I = one(S)
    P = F ⋅ S
    dP_dF = otimesu(I, S) + 2 * otimesu(F, I) ⊡ dS_dC ⊡ otimesu(F', I)
    return P, dP_dF
end

# Tangent integration is selected by tangent_symmetry(mp): MajorSymmetric fills the
# lower triangle and mirrors it, Unsymmetric integrates every entry. The residual
# never touches D, so it shares one method.
@inline function assemble_small_strain_tangent!(::MajorSymmetric, ke::AbstractMatrix, re::AbstractVector, sigma, D, cellvalues, qp, alpha_dOmega, n_basefuncs::Int)
    @inbounds for i in 1:n_basefuncs
        delta_epsilon = shape_symmetric_gradient(cellvalues, qp, i)
        re[i] += (delta_epsilon ⊡ sigma) * alpha_dOmega
        delta_epsilon_D = delta_epsilon ⊡ D
        for j in 1:i
            ke[i, j] += (delta_epsilon_D ⊡ shape_symmetric_gradient(cellvalues, qp, j)) * alpha_dOmega
        end
    end
    _mirror_lower_triangle!(ke, n_basefuncs)
    return nothing
end

@inline function assemble_small_strain_tangent!(::Unsymmetric, ke::AbstractMatrix, re::AbstractVector, sigma, D, cellvalues, qp, alpha_dOmega, n_basefuncs::Int)
    @inbounds for i in 1:n_basefuncs
        delta_epsilon = shape_symmetric_gradient(cellvalues, qp, i)
        re[i] += (delta_epsilon ⊡ sigma) * alpha_dOmega
        delta_epsilon_D = delta_epsilon ⊡ D
        for j in 1:n_basefuncs
            ke[i, j] += (delta_epsilon_D ⊡ shape_symmetric_gradient(cellvalues, qp, j)) * alpha_dOmega
        end
    end
    return nothing
end

@inline function assemble_small_strain_tangent!(::AbstractTangentSymmetry, ::Nothing, re::AbstractVector, sigma, D, cellvalues, qp, alpha_dOmega, n_basefuncs::Int)
    @inbounds for i in 1:n_basefuncs
        delta_epsilon = shape_symmetric_gradient(cellvalues, qp, i)
        re[i] += (delta_epsilon ⊡ sigma) * alpha_dOmega
    end
    return nothing
end

@inline function assemble_small_strain_tangent!(::MajorSymmetric, ke::AbstractMatrix, ::Nothing, sigma, D, cellvalues, qp, alpha_dOmega, n_basefuncs::Int)
    @inbounds for i in 1:n_basefuncs
        delta_epsilon_D = shape_symmetric_gradient(cellvalues, qp, i) ⊡ D
        for j in 1:i
            ke[i, j] += (delta_epsilon_D ⊡ shape_symmetric_gradient(cellvalues, qp, j)) * alpha_dOmega
        end
    end
    _mirror_lower_triangle!(ke, n_basefuncs)
    return nothing
end

@inline function assemble_small_strain_tangent!(::Unsymmetric, ke::AbstractMatrix, ::Nothing, sigma, D, cellvalues, qp, alpha_dOmega, n_basefuncs::Int)
    @inbounds for i in 1:n_basefuncs
        delta_epsilon_D = shape_symmetric_gradient(cellvalues, qp, i) ⊡ D
        for j in 1:n_basefuncs
            ke[i, j] += (delta_epsilon_D ⊡ shape_symmetric_gradient(cellvalues, qp, j)) * alpha_dOmega
        end
    end
    return nothing
end

@inline function _mirror_lower_triangle!(ke, n_basefuncs::Int)
    @inbounds for i in 1:n_basefuncs, j in (i + 1):n_basefuncs
        ke[i, j] = ke[j, i]
    end
    return nothing
end

@inline function _compute_stress_qp(mp::AbstractMaterial, cellvalues, alphavalues, qp, u_local, state::Nothing, dt=0.0)
    return _compute_stress_qp(mp, cellvalues, alphavalues, qp, u_local, NoState(), dt)
end

#################################
# AbstractHyperelastic defaults #
#################################

kinematics(::AbstractHyperelastic) = FiniteStrain()

# AD defaults from the strain energy density Ψ(C, mp), following Ferrite.jl's
# hyperelasticity tutorial; models override with analytic expressions where available.
function constitutive_driver(C, mp::AbstractHyperelastic)
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    return 2∂Ψ∂C, 2∂²Ψ∂C²
end

@inline _second_piola(C, mp::AbstractHyperelastic) = 2 * Tensors.gradient(y -> Ψ(y, mp), C)

@inline _finite_strain_pk1_tangent(F::Tensor{2,3}, mp::AbstractHyperelastic, dt) = _pk1_tangent_from_second_piola(F, mp)

function material_response(mp::AbstractHyperelastic, F::Tensor{2,3}, state, dt, cache=nothing)
    P, dP_dF = _finite_strain_pk1_tangent(F, mp, dt)
    return P, dP_dF, state
end

function _compute_stress_qp(mp::AbstractHyperelastic, cellvalues, alphavalues, qp, u_local, state::NoState=NoState(), dt=0.0)
    F = deformation_gradient(cellvalues, qp, u_local)
    C = tdot(F)
    S = _second_piola(C, mp)
    return (F ⋅ S ⋅ F') / det(F)
end

# 2D wrapper interface shared by all hyperelastic materials
compute_PK1_3D(mp::AbstractHyperelastic, F::Tensor{2,3}, dt, state) = _pk1_from_second_piola(F, mp)
update_state_from_3D!(state, mp::AbstractHyperelastic, F, dt) = nothing

############################################################
# Generic element routine (fallback for material_response) #
############################################################

# Per-QP state access: the assembler passes a Vector of QP states for
# nonlinear cells and a bare NoState() during linear preassembly. The witness
# pins the concrete state type so the per-QP loop dispatches statically (the
# assembler stores states as Vector{AbstractMaterialState}).
@inline _state_witness(states::AbstractVector) = isempty(states) ? NoState() : @inbounds(states[1])
@inline _state_witness(state) = state
@inline _qp_state(states::AbstractVector, qp::Int, ::Type{W}) where {W} = @inbounds states[qp]::W
@inline _qp_state(state, qp::Int, ::Type{W}) where {W} = state

# Store the trial result of material_response according to its form (see the
# material_response docstring): state itself -> nothing to do; replacement
# AbstractMaterialState -> replace the QP slot; snapshot -> set_trial!.
@inline function _store_trial!(states, state, new, qp::Int)
    new === state && return nothing
    if new isa AbstractMaterialState
        states isa AbstractVector && (@inbounds states[qp] = new)
    else
        set_trial!(state, new)
    end
    return nothing
end

function _assemble_element!(ke, re, states, mp::AbstractMaterial, cellvalues, alphavalues, u, dt)
    return _generic_element!(kinematics(mp), ke, re, states, _state_witness(states), mp, cellvalues, alphavalues, u, dt)
end

function _generic_element!(::SmallStrain, ke, re, states, witness::W, mp, cv, av, u, dt) where {W}
    n_basefuncs = getnbasefunctions(cv)
    cache = allocate_material_cache(mp)
    sym = tangent_symmetry(mp)
    for qp in 1:getnquadpoints(cv)
        α_dΩ = alpha_value(av, qp) * getdetJdV(cv, qp)
        state = _qp_state(states, qp, W)
        ε = (u === nothing) ? zero(shape_symmetric_gradient(cv, qp, 1)) : function_symmetric_gradient(cv, qp, u)
        σ, D, new = material_response(mp, ε, state, dt, cache)
        _store_trial!(states, state, new, qp)
        assemble_small_strain_tangent!(sym, ke, re, σ, D, cv, qp, α_dΩ, n_basefuncs)
    end
    return nothing
end

function _generic_element!(::FiniteStrain, ke, re, states, witness::W, mp, cv, av, u, dt) where {W}
    cache = allocate_material_cache(mp)
    for qp in 1:getnquadpoints(cv)
        α_dΩ = alpha_value(av, qp) * getdetJdV(cv, qp)
        state = _qp_state(states, qp, W)
        F = (u === nothing) ? one(shape_gradient(cv, qp, 1)) : deformation_gradient(cv, qp, u)
        P, dP_dF, new = material_response(mp, F, state, dt, cache)
        _store_trial!(states, state, new, qp)
        assemble_pk1_tangent!(ke, re, P, dP_dF, cv, qp, α_dΩ)
    end
    return nothing
end

# Default material_stress: derive the output stress from material_response
# evaluated at the QP state as-is and discard the tangent/trial. Correct for
# any material, but it also computes the tangent – history models with
# expensive tangents can/should override material_stress with a stress-only path.
material_stress(mp::AbstractMaterial, strain, state, dt, cache=nothing) =
    _material_stress(kinematics(mp), mp, strain, state, dt, cache)
_material_stress(::SmallStrain, mp, ε, state, dt, cache) = material_response(mp, ε, state, dt, cache)[1]
function _material_stress(::FiniteStrain, mp, F, state, dt, cache)
    P = material_response(mp, F, state, dt, cache)[1]
    return (P ⋅ F') / det(F)
end

# Last-resort stress fallback: extract the QP strain measure and hand it to
# material_stress. Models can get a cheap stress path by overriding
# material_stress rather than this cellvalues-level method.
function _compute_stress_qp(mp::AbstractMaterial, cellvalues, alphavalues, qp, u_local, state, dt=0.0)
    return _generic_stress_qp(kinematics(mp), mp, cellvalues, alphavalues, qp, u_local, state, dt)
end

function _generic_stress_qp(::SmallStrain, mp, cv, av, qp, u_local, state, dt)
    ε = function_symmetric_gradient(cv, qp, u_local)
    return material_stress(mp, ε, state, dt, allocate_material_cache(mp))
end

function _generic_stress_qp(::FiniteStrain, mp, cv, av, qp, u_local, state, dt)
    F = deformation_gradient(cv, qp, u_local)
    return material_stress(mp, F, state, dt, allocate_material_cache(mp))
end

###################################################################
# Generic PlaneStrain/PlaneStress hooks (fallback via kinematics) #
###################################################################

# A SmallStrain material is wrappable straight from material_response: the
# wrappers pass the embedded 3D F, we take ε = sym(F - I), evaluate the
# constitutive update, and map σ to PK1. FiniteStrain materials have no such
# universal conversion, so the FiniteStrain branch below errors. Both are
# reached only when a material has no hand-written hook.
compute_PK1_3D(mp::AbstractMaterial, F::Tensor{2,3}, dt, state) =
    _compute_PK1_3D(kinematics(mp), mp, F, dt, state)

function _compute_PK1_3D(::SmallStrain, mp, F::Tensor{2,3}, dt, state)
    ε = symmetric(F - one(F))
    σ, _, _ = material_response(mp, ε, state, dt, allocate_material_cache(mp))
    return det(F) * σ ⋅ inv(F)'
end

_compute_PK1_3D(::FiniteStrain, mp, F, dt, state) = throw(ArgumentError(
    "$(typeof(mp)) uses FiniteStrain kinematics and provides no compute_PK1_3D method; " *
    "finite-strain materials must implement compute_PK1_3D to be wrapped by PlaneStrain/PlaneStress."))

update_state_from_3D!(state, mp::AbstractMaterial, F, dt) =
    _update_state_from_3D!(kinematics(mp), state, mp, F, dt)

# Stateless materials have no trial to store – skip the material_response call.
_update_state_from_3D!(::SmallStrain, state::NoState, mp, F, dt) = nothing

function _update_state_from_3D!(::SmallStrain, state, mp, F, dt)
    ε = symmetric(F - one(F))
    _, _, new = material_response(mp, ε, state, dt, allocate_material_cache(mp))
    new === state && return nothing
    new isa AbstractMaterialState && return new
    set_trial!(state, new)
    return nothing
end

_update_state_from_3D!(::FiniteStrain, state, mp, F, dt) = throw(ArgumentError(
    "$(typeof(mp)) uses FiniteStrain kinematics and provides no update_state_from_3D! method; " *
    "finite-strain materials must implement update_state_from_3D! to be wrapped by PlaneStrain/PlaneStress."))