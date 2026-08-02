"""
    Ogden(μ, α, κ)

Compressible Ogden hyperelastic material with an arbitrary number of isochoric terms.
The stress and tangent are analytic, in spectral form.
The vectors `μ` and `α` hold one coefficient and one exponent per term, in the `μ / α` Ogden convention.
Scalar `μ` and `α` values are accepted as a one-term shorthand.
The bulk modulus `κ` parameterizes the volumetric response.
The combined small-strain shear modulus `0.5 * sum(μ .* α)` must be positive.

# References

- R. W. Ogden.
  *Large deformation isotropic elasticity – on the correlation of theory and
  experiment for incompressible rubberlike solids.*
  Proceedings of the Royal Society of London. Series A, Mathematical and
  Physical Sciences **326**(1567) (1972) 565-584.
  <https://doi.org/10.1098/rspa.1972.0026>
- R. W. Ogden.
  *Non-Linear Elastic Deformations.*
  Ellis Horwood, 1984.
"""
struct Ogden{N} <: AbstractHyperelastic
    μ::NTuple{N,Float64}
    α::NTuple{N,Float64}
    κ::Float64

    function Ogden(μ, α, κ)
        μ_tuple = _ogden_parameter_tuple(μ, "μ")
        α_tuple = _ogden_parameter_tuple(α, "α")
        length(μ_tuple) == length(α_tuple) ||
            throw(ArgumentError("Ogden: μ and α must have the same length (got $(length(μ_tuple)) and $(length(α_tuple)))"))
        if any(iszero, α_tuple)
            throw(ArgumentError("Ogden: every α entry must be nonzero"))
        end
        if !isfinite(κ)
            throw(ArgumentError("Ogden: κ must be finite (got κ=$κ)"))
        end
        if !isless(0.0, κ)
            throw(ArgumentError("Ogden: κ must be positive (got κ=$κ)"))
        end
        μ0 = 0.5 * sum(μi * αi for (μi, αi) in zip(μ_tuple, α_tuple))
        if !isless(0.0, μ0)
            throw(ArgumentError("Ogden: 0.5 * sum(μ .* α) must be positive (got $μ0)"))
        end
        return new{length(μ_tuple)}(μ_tuple, α_tuple, Float64(κ))
    end
end

function _ogden_parameter_tuple(x, name::String)
    if x isa Real
        isfinite(x) || throw(ArgumentError("Ogden: $name must be finite (got $name=$x)"))
        return (Float64(x),)
    elseif x isa Tuple || x isa AbstractVector
        values = Tuple(Float64(v) for v in x)
        isempty(values) && throw(ArgumentError("Ogden: $name must contain at least one entry"))
        all(isfinite, values) || throw(ArgumentError("Ogden: every $name entry must be finite"))
        return values
    else
        throw(ArgumentError("Ogden: $name must be a real scalar, tuple, or vector"))
    end
end

# Generic API methods

@inline function _ogden_divided_difference(λa, λb, q)
    scale = max(one(λa), abs(λa), abs(λb))
    if abs(λa - λb) <= sqrt(eps(Float64)) * scale
        λ = (λa + λb) / 2
        return q * λ^(q - 1)
    end
    return (λa^q - λb^q) / (λa - λb)
end

function _ogden_power_from_eigen(ev, q, ::Type{T}) where T
    λ = ev.values
    Q = ev.vectors
    return SymmetricTensor{2,3,T}((i, j) -> begin
        value = zero(T)
        @inbounds for a in 1:3
            value += λ[a]^q * Q[i, a] * Q[j, a]
        end
        value
    end)
end

function _ogden_power_derivative_from_eigen(ev, q, ::Type{T}) where T
    λ = ev.values
    Q = ev.vectors
    return Tensor{4,3,T}((i, j, k, l) -> begin
        value = zero(T)
        @inbounds for a in 1:3, b in 1:3
            L = _ogden_divided_difference(λ[a], λ[b], q)
            value += L * Q[i, a] * Q[j, b] *
                     (Q[k, a] * Q[l, b] + Q[l, a] * Q[k, b]) / 2
        end
        value
    end)
end

@inline function _ogden_term_trace(ev, p)
    value = zero(eltype(ev.values))
    @inbounds for λ in ev.values
        value += λ^p
    end
    return value
end

function Ψ(C::SymmetricTensor{2,3,T}, mp::Ogden) where T
    J2 = det(C)
    J = sqrt(J2)
    ev = eigen(C)
    Ψiso = zero(T)
    @inbounds for n in eachindex(mp.μ)
        αn = mp.α[n]
        p = αn / 2
        Ψiso += mp.μ[n] / αn * (J2^(-αn / 6) * _ogden_term_trace(ev, p) - 3)
    end
    return Ψiso + 0.5 * mp.κ * (J - 1)^2
end

function _ogden_second_piola(C::SymmetricTensor{2,3,T}, mp::Ogden) where T
    Cinv = inv(C)
    J2 = det(C)
    J = sqrt(J2)
    ev = eigen(C)

    S = mp.κ * J * (J - 1) * Cinv
    @inbounds for n in eachindex(mp.μ)
        αn = mp.α[n]
        p = αn / 2
        g = J2^(-αn / 6)
        trCp = _ogden_term_trace(ev, p)
        A = _ogden_power_from_eigen(ev, p - 1, T)
        S += mp.μ[n] * g * (A - (trCp / 3) * Cinv)
    end
    return S
end

@inline _second_piola(C::SymmetricTensor{2,3,T}, mp::Ogden) where T = _ogden_second_piola(C, mp)

function constitutive_driver(C::SymmetricTensor{2,3,T}, mp::Ogden) where T
    Cinv = inv(C)
    J2 = det(C)
    J = sqrt(J2)
    ev = eigen(C)

    S = mp.κ * J * (J - 1) * Cinv
    dpvol_dC = 0.5 * mp.κ * J * (2 * J - 1) * Cinv
    dS_dC = Tensor{4,3,T}((i, j, k, l) -> begin
        inv_sym = 0.5 * (Cinv[i, k] * Cinv[l, j] + Cinv[i, l] * Cinv[k, j])
        Cinv[i, j] * dpvol_dC[k, l] - mp.κ * J * (J - 1) * inv_sym
    end)

    @inbounds for n in eachindex(mp.μ)
        μn = mp.μ[n]
        αn = mp.α[n]
        p = αn / 2
        g = J2^(-αn / 6)
        trCp = _ogden_term_trace(ev, p)
        A = _ogden_power_from_eigen(ev, p - 1, T)
        dA_dC = _ogden_power_derivative_from_eigen(ev, p - 1, T)
        B = A - (trCp / 3) * Cinv
        dg_dC = -(αn / 6) * g * Cinv
        dtrCp_dC = p * A

        S += μn * g * B
        dS_dC += Tensor{4,3,T}((i, j, k, l) -> begin
            inv_sym = 0.5 * (Cinv[i, k] * Cinv[l, j] + Cinv[i, l] * Cinv[k, j])
            μn * (B[i, j] * dg_dC[k, l] +
                  g * (dA_dC[i, j, k, l] - (dtrCp_dC[k, l] / 3) * Cinv[i, j] + (trCp / 3) * inv_sym))
        end)
    end

    return S, dS_dC
end

function _ogden_pk1_tangent(F::Tensor{2,3,T}, mp::Ogden) where T
    return _pk1_tangent_from_second_piola(F, mp)
end

@inline _finite_strain_pk1_tangent(F::Tensor{2,3,T}, mp::Ogden, dt) where T = _ogden_pk1_tangent(F, mp)

@inline function _ogden_extract_plane_strain_tangent(dP3D::AbstractTensor{4,3,T}) where T
    return Tensor{4,2,T}((i, j, k, l) -> dP3D[i, j, k, l])
end

# Wrapper specializations: Ogden provides the analytic (P, ∂P∂F) pair, so the
# 2D reductions use it directly instead of AD through the 2D->3D embedding.
# Both use Tensors.value on F2D, which the assembler passes as Float64; these paths are not AD-transparent.
function material_response(ps::PlaneStrain{<:Ogden}, F2D::Tensor{2,2}, state, dt, cache=nothing)
    F3D = embed_F_2D_to_3D(Tensors.value(F2D), 1.0)
    P3D, dP3D_dF3D = _ogden_pk1_tangent(F3D, ps.model)
    update_state_from_3D!(state, ps.model, F3D, dt)
    return extract_P_2D(P3D), _ogden_extract_plane_strain_tangent(dP3D_dF3D), state
end

function material_response(ps::PlaneStress{<:Ogden}, F2D::Tensor{2,2}, state::PlaneStressStateWrapper, dt, cache=nothing)
    F2D_val = Tensors.value(F2D)
    F33 = state.F33_previous
    converged = false
    last_res = NaN

    for iter in 1:ps.maxiter
        F3D = embed_F_2D_to_3D(F2D_val, F33)
        P3D, dP3D_dF3D = _ogden_pk1_tangent(F3D, ps.model)
        res = P3D[3, 3]
        last_res = res
        # Relative to the stress scale, so tol means the same for a soft and a stiff material
        if abs(res) <= ps.tol * max(norm(P3D), 1.0)
            converged = true
            break
        end
        dP33_dF33 = dP3D_dF3D[3, 3, 3, 3]
        if abs(dP33_dF33) <= sqrt(eps(Float64))
            throw(PlaneStressConvergenceError(:small_newton_derivative, dP33_dF33; residual=res, iterations=iter))
        end
        F33 -= res / dP33_dF33
    end

    converged || throw(PlaneStressConvergenceError(:local_newton_nonconvergence, F33; residual=last_res, iterations=ps.maxiter))
    state.F33_current = F33

    F3D = embed_F_2D_to_3D(F2D_val, F33)
    P3D, dP3D_dF3D = _ogden_pk1_tangent(F3D, ps.model)
    P2D = extract_P_2D(P3D)
    dP2D_dF2D = condense_tangent_2D(dP3D_dF3D)

    update_state_from_3D!(state.inner, ps.model, F3D, dt)
    return P2D, dP2D_dF2D, state
end