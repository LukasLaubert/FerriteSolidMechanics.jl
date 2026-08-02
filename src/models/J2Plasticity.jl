# The J2 return mapping follows Ferrite.jl's von Mises plasticity tutorial.

"""
    J2Plasticity(E, nu, sigma0, H)

Small-strain von Mises plasticity with isotropic hardening.

### Parameters

- `E` – Young's modulus; higher values stiffen the elastic response
- `nu` – Poisson's ratio; controls lateral contraction under stretch
- `sigma0` – Initial yield limit; higher values delay the onset of plasticity
- `H` – Linear hardening modulus; larger values raise the post-yield stiffness

# References

- E. A. de Souza Neto, D. Perić, D. R. J. Owen. *Computational Methods for Plasticity: Theory and Applications.* Wiley, 2008.
- J. C. Simo, T. J. R. Hughes. *Computational Inelasticity.* Springer, 1998.
- Ferrite.jl documentation. *Von Mises plasticity tutorial.* <https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/plasticity/>
"""
struct J2Plasticity{T, S <: SymmetricTensor{4, 3, T}} <: AbstractMaterial
    G::T   # Shear modulus
    K::T   # Bulk modulus
    σ₀::T  # Initial yield limit
    H::T   # Hardening modulus
    Dᵉ::S  # Elastic stiffness tensor
end

function J2Plasticity(E, ν, σ₀, H)
    δ(i, j) = i == j ? 1.0 : 0.0
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)

    temp(i, j, k, l) = 2.0G * (0.5 * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)) + ν / (1.0 - 2.0ν) * δ(i, j) * δ(k, l))
    Dᵉ = SymmetricTensor{4, 3}(temp)
    return J2Plasticity(G, K, σ₀, H, Dᵉ)
end

# --- State ---

struct J2PlasticityStateInternal{T}
    ϵᵖ::SymmetricTensor{2, 3, T, 6}  # plastic strain
    σ::SymmetricTensor{2, 3, T, 6}   # stress
    k::T                             # hardening variable
end

mutable struct J2PlasticityState <: AbstractMaterialState
    current::J2PlasticityStateInternal{Float64}
    previous::J2PlasticityStateInternal{Float64}
end

# --- API hooks ---

is_linear(::J2Plasticity) = false

function create_state(::J2Plasticity)
    return J2PlasticityState(
        J2PlasticityStateInternal(zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), 0.0),
        J2PlasticityStateInternal(zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), 0.0),
    )
end

function update_state!(state::J2PlasticityState)
    state.previous = state.current
    return nothing
end

function revert_state!(state::J2PlasticityState)
    state.current = state.previous
    return nothing
end

# --- Constitutive driver ---

function vonMises(σ)
    s = dev(σ)
    return sqrt(3.0 / 2.0 * s ⊡ s)
end

function compute_stress_tangent(ϵ::SymmetricTensor{2, 3}, material::J2Plasticity, state::J2PlasticityStateInternal)
    # unpack some material parameters
    G = material.G
    H = material.H

    # Trial-values
    σᵗ = material.Dᵉ ⊡ (ϵ - state.ϵᵖ)  # trial-stress
    sᵗ = dev(σᵗ)                        # deviatoric part of trial-stress
    J₂ = 0.5 * sᵗ ⊡ sᵗ                 # second invariant of sᵗ
    σᵗₑ = sqrt(3.0 * J₂)                # effective trial-stress (von Mises stress)
    σʸ = material.σ₀ + H * state.k      # Previous yield limit

    φᵗ = σᵗₑ - σʸ                       # Trial-value of the yield surface

    if φᵗ < 0.0                         # elastic loading
        # Under the plane wrappers' ForwardDiff pass σᵗ is Dual while the
        # history stays Float64; convert so the state's eltype stays uniform
        T = eltype(σᵗ)
        ϵᵖ = convert(SymmetricTensor{2, 3, T, 6}, state.ϵᵖ)
        return σᵗ, material.Dᵉ, J2PlasticityStateInternal(ϵᵖ, σᵗ, convert(T, state.k))
    else                                # plastic loading
        h = H + 3G
        μ = φᵗ / h                      # plastic multiplier

        c1 = 1 - 3G * μ / σᵗₑ
        s = c1 * sᵗ                     # updated deviatoric stress
        σ = s + vol(σᵗ)                 # updated stress

        # Algorithmic tangent stiffness D = Δσ / Δϵ
        κ = H * (state.k + μ)           # drag stress
        σₑ = material.σ₀ + κ            # updated yield surface

        δ(i, j) = i == j ? 1.0 : 0.0
        Isymdev(i, j, k, l) = 0.5 * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)) - 1.0 / 3.0 * δ(i, j) * δ(k, l)
        Q(i, j, k, l) = Isymdev(i, j, k, l) - 3.0 / (2.0 * σₑ^2) * s[i, j] * s[k, l]
        b = (3G * μ / σₑ) / (1.0 + 3G * μ / σₑ)

        Dtemp(i, j, k, l) = -2G * b * Q(i, j, k, l) - 9G^2 / (h * σₑ^2) * s[i, j] * s[k, l]
        D = material.Dᵉ + SymmetricTensor{4, 3}(Dtemp)

        # Return new state
        Δϵᵖ = 3 / 2 * μ / σₑ * s          # plastic strain increment
        ϵᵖ = state.ϵᵖ + Δϵᵖ               # plastic strain
        k = state.k + μ                   # hardening variable
        return σ, D, J2PlasticityStateInternal(ϵᵖ, σ, k)
    end
end

# --- Constitutive interface (small-strain) ---

kinematics(::J2Plasticity) = SmallStrain()

# Associated flow with isotropic hardening: both Dᵉ and the plastic-branch tangent
# are major-symmetric, so the symmetric tangent loop applies
tangent_symmetry(::J2Plasticity) = MajorSymmetric()

function material_response(mp::J2Plasticity, ε::SymmetricTensor{2,3}, state::J2PlasticityState, dt, cache=nothing)
    return compute_stress_tangent(ε, mp, state.previous)
end

# Stress output and the 2D wrapper hooks (compute_PK1_3D / update_state_from_3D!)
# are derived generically for SmallStrain materials from material_response – see
# ModelHelpers.jl. A small-strain history model needs only material_response.