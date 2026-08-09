# Core logic for generic plane strain and plane stress wrappers

"""
    PlaneStressConvergenceError

Recoverable failure raised by [`PlaneStress`](@ref) when the local out-of-plane Newton solve or tangent condensation cannot produce a valid plane stress state.
Catch this directly, or call [`try_stiffness_matrix`](@ref) to receive it as `result.error`.
"""
struct PlaneStressConvergenceError <: LocalAssemblyFailure
    reason::Symbol
    value
    residual
    iterations::Int
end

function PlaneStressConvergenceError(reason::Symbol, value; residual=NaN, iterations=0)
    return PlaneStressConvergenceError(reason, value, residual, Int(iterations))
end

function Base.showerror(io::IO, err::PlaneStressConvergenceError)
    print(io, "PlaneStress local solve failed (", err.reason, ")")
    print(io, ": value=", err.value)
    print(io, ", residual=", err.residual)
    print(io, ", iterations=", err.iterations)
end

# --- 1. Helper utilities for dimension conversion ---

# Embed 2D deformation gradient into 3D (column-major storage in Tensors.jl)
function embed_F_2D_to_3D(F2D::Tensor{2,2,T}, F33::S=one(T)) where {T,S<:Number}
    T2 = promote_type(T, S)
    return Tensor{2,3,T2}((
        T2(F2D[1, 1]), T2(F2D[2, 1]), zero(T2),
        T2(F2D[1, 2]), T2(F2D[2, 2]), zero(T2),
        zero(T2), zero(T2), T2(F33)
    ))
end

# Extract in-plane 2D PK1 from 3D PK1
function extract_P_2D(P3D::AbstractTensor{2,3,T}) where T
    return Tensor{2,2,T}((P3D[1, 1], P3D[2, 1], P3D[1, 2], P3D[2, 2]))
end

# Extract in-plane 2D tangent via static condensation from 3D tangent
# dP2D_dF2D[i,j,k,l] = dP3D_dF3D[i,j,k,l] - dP3D_dF3D[i,j,3,3]*dP3D_dF3D[3,3,k,l] / dP3D_dF3D[3,3,3,3]
function condense_tangent_2D(dP::AbstractTensor{4,3,T}) where T
    C3333 = dP[3, 3, 3, 3]
    # Relative check: an absolute floor misses C3333 when it is tiny compared to norm(dP)
    R = real(float(T))
    if abs(C3333) <= sqrt(eps(R)) * max(norm(dP), one(R))
        throw(PlaneStressConvergenceError(:singular_condensation, C3333))
    end
    return Tensor{4,2,T}((i, j, k, l) -> dP[i, j, k, l] - dP[i, j, 3, 3] * dP[3, 3, k, l] / C3333)
end

# --- 2. PlaneStrain wrapper ---

"""
    PlaneStrain(model)

Wrap a 3D material model for a 2D plane strain analysis.
"""
struct PlaneStrain{M<:AbstractMaterial} <: AbstractMaterial
    model::M
end

is_linear(ps::PlaneStrain) = false
create_state(ps::PlaneStrain) = create_state(ps.model)
kinematics(::PlaneStrain) = FiniteStrain()

# Two calls on purpose: `gradient(..., :all)` is no faster and shifts results ~1e-13
function material_response(ps::PlaneStrain, F2D::Tensor{2,2}, state, dt, cache=nothing)
    # Exact consistent tangent via AD through the 2D->3D embedding
    F3D = embed_F_2D_to_3D(F2D, 1.0)
    P2D = extract_P_2D(Tensor{2,3}(compute_PK1_3D(ps.model, F3D, dt, state)))
    dP2D_dF2D = Tensors.gradient(F2D_ -> extract_P_2D(Tensor{2,3}(compute_PK1_3D(ps.model, embed_F_2D_to_3D(F2D_, 1.0), dt, state))), F2D)

    # Update internal 3D state variables at converged F
    updated_state = update_state_from_3D!(state, ps.model, F3D, dt)
    return P2D, dP2D_dF2D, (updated_state isa AbstractMaterialState ? updated_state : state)
end

function _compute_stress_qp(ps::PlaneStrain, cv, av, qp, ul, state::AbstractMaterialState, dt=0.0)
    F2D = deformation_gradient(cv, qp, ul)
    F3D = embed_F_2D_to_3D(F2D, 1.0)
    P3D = Tensor{2,3}(compute_PK1_3D(ps.model, F3D, dt, state))
    σ3D = (P3D ⋅ F3D') / det(F3D)
    return extract_P_2D(σ3D) # Cauchy 2D is just the in-plane part of Cauchy 3D
end

# --- 3. PlaneStress wrapper ---

"""
    PlaneStress(model; tol=1e-10, maxiter=20)

Wrap a 3D material model for a 2D plane stress analysis.
A scalar Newton iteration solves the out-of-plane stretch locally at each quadrature point.

### Keyword arguments

- `tol` – Residual tolerance on `P[3,3]`, relative to the stress scale `max(norm(P3D), 1)`
- `maxiter` – Maximum number of local Newton iterations
"""
struct PlaneStress{M<:AbstractMaterial} <: AbstractMaterial
    model::M
    tol::Float64
    maxiter::Int

    function PlaneStress(model::M; tol::Real=1e-10, maxiter::Integer=20) where {M<:AbstractMaterial}
        tol > 0 || throw(ArgumentError("PlaneStress tolerance must be positive, got $tol"))
        maxiter > 0 || throw(ArgumentError("PlaneStress maxiter must be positive, got $maxiter"))
        return new{M}(model, Float64(tol), Int(maxiter))
    end
end

is_linear(::PlaneStress) = false # Always nonlinear due to constraint iteration

"""
    PlaneStressStateWrapper

State wrapper for `PlaneStress`.
It holds the wrapped material state together with the current and previous out-of-plane stretch.
"""
mutable struct PlaneStressStateWrapper{S} <: AbstractMaterialState
    inner::S
    F33_current::Float64
    F33_previous::Float64
end

function create_state(ps::PlaneStress)
    inner = create_state(ps.model)
    return PlaneStressStateWrapper(inner, 1.0, 1.0)
end

function update_state!(state::PlaneStressStateWrapper{S}) where S
    updated_inner = update_state!(state.inner)
    if updated_inner isa S
        state.inner = updated_inner
    end
    state.F33_previous = state.F33_current
    return nothing
end

function revert_state!(state::PlaneStressStateWrapper{S}) where S
    reverted_inner = revert_state!(state.inner)
    if reverted_inner isa S
        state.inner = reverted_inner
    end
    state.F33_current = state.F33_previous
    return nothing
end

function copy_state!(dest::PlaneStressStateWrapper{S}, src::PlaneStressStateWrapper{S}) where S
    copied_inner = copy_state!(dest.inner, src.inner)
    if copied_inner isa S
        dest.inner = copied_inner
    end
    dest.F33_current = src.F33_current
    dest.F33_previous = src.F33_previous
    return dest
end

kinematics(::PlaneStress) = FiniteStrain()

function material_response(ps::PlaneStress, F2D::Tensor{2,2}, state::PlaneStressStateWrapper, dt, cache=nothing)
    # Local Newton loop to find F33 (scalar) such that P3D[3,3] = 0
    # F2D is already Float64 from the assembler; F33 is a plain scalar.
    F33 = state.F33_previous  # warm start from previous converged value
    tol = ps.tol
    maxiter = ps.maxiter
    converged = false
    last_res = NaN
    F3D = embed_F_2D_to_3D(F2D, F33)
    P3D = zero(F3D)
    for iter in 1:maxiter
        P3D = Tensor{2,3}(compute_PK1_3D(ps.model, F3D, dt, state.inner))
        res = P3D[3, 3]
        last_res = res
        # Relative to the stress scale, so tol means the same for a soft and a stiff material
        if abs(res) <= tol * max(norm(P3D), 1.0)
            converged = true
            break
        end

        dP33_dF33 = Tensors.gradient(F33_ -> begin
                F3D_ = embed_F_2D_to_3D(F2D, F33_)
                Tensor{2,3}(compute_PK1_3D(ps.model, F3D_, dt, state.inner))[3, 3]
            end, F33)
        if abs(dP33_dF33) <= sqrt(eps(Float64))
            throw(PlaneStressConvergenceError(:small_newton_derivative, dP33_dF33; residual=res, iterations=iter))
        end
        F33 -= res / dP33_dF33
        F3D = embed_F_2D_to_3D(F2D, F33)
    end
    converged || throw(PlaneStressConvergenceError(:local_newton_nonconvergence, F33; residual=last_res, iterations=maxiter))
    state.F33_current = F33  # already Float64

    # Tangent at the converged F33
    dP3D = Tensors.gradient(F_ -> Tensor{2,3}(compute_PK1_3D(ps.model, F_, dt, state.inner)), F3D)
    P2D = extract_P_2D(P3D)
    dP2D_dF2D = condense_tangent_2D(dP3D)

    # Update internal 3D state variables at the converged F33
    updated_inner = update_state_from_3D!(state.inner, ps.model, F3D, dt)
    if updated_inner isa typeof(state.inner)
        state.inner = updated_inner
    end

    return P2D, dP2D_dF2D, state
end

function _compute_stress_qp(ps::PlaneStress, cv, av, qp, ul, state::PlaneStressStateWrapper, dt=0.0)
    F2D = deformation_gradient(cv, qp, ul)
    # We use F33_current as the best estimate for stress output
    F3D = embed_F_2D_to_3D(F2D, state.F33_current)
    P3D = Tensor{2,3}(compute_PK1_3D(ps.model, F3D, dt, state.inner))
    σ3D = (P3D ⋅ F3D') / det(F3D)
    return extract_P_2D(σ3D)
end