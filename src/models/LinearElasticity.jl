# The 3D Hooke routine follows Ferrite.jl's linear elasticity tutorial.

########################
# 3D linear elasticity #
########################

"""
    Hooke(E, nu)

Three-dimensional isotropic linear elastic material.
The stress is `σ = C ⊡ ε` with a constant tangent `C`, so the assembler treats `Hooke` cells as linear.

### Parameters

- `E` – Young's modulus; higher values stiffen the elastic response
- `nu` – Poisson's ratio; controls lateral contraction under stretch

# References

- Ferrite.jl documentation. *Linear elasticity tutorial.* <https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/linear_elasticity/>
"""
struct Hooke <: AbstractMaterial
    C::SymmetricTensor{4,3,Float64,36}
end

function Hooke(E, ν)
    μ, λ = lame_parameters(E, ν)

    I = one(SymmetricTensor{2,3})
    Isym = one(SymmetricTensor{4,3})

    return Hooke(λ * I ⊗ I + 2μ * Isym)
end

########################
# 2D linear elasticity #
########################

"""
    Hooke2D(E, nu; plane_stress=false)

Two-dimensional isotropic linear elastic material.
The stress is `σ = C ⊡ ε` with a constant tangent `C`, so the assembler treats `Hooke2D` cells as linear.

### Parameters

- `E` – Young's modulus; higher values stiffen the elastic response
- `nu` – Poisson's ratio; controls lateral contraction under stretch

### Keyword arguments

- `plane_stress` – selects the plane stress constitutive matrix (`σ_zz = 0`); the default `false` selects plane strain (`ε_zz = 0`)
"""
struct Hooke2D <: AbstractMaterial
    C::SymmetricTensor{4,2,Float64,9}
end

function Hooke2D(E, ν; plane_stress::Bool=false)
    if plane_stress
        # Plane Stress assumption: sigma_zz = 0
        C_voigt = E / (1 - ν^2) * [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1-ν)/2]
        return Hooke2D(fromvoigt(SymmetricTensor{4,2}, C_voigt))
    else
        # Plane Strain assumption: epsilon_zz = 0
        μ, λ = lame_parameters(E, ν)
        I = one(SymmetricTensor{2,2})
        Isym = one(SymmetricTensor{4,2})
        return Hooke2D(λ * I ⊗ I + 2μ * Isym)
    end
end

####################
# Common interface #
####################

# Generic API methods
is_linear(::Union{Hooke,Hooke2D}) = true

# Unified constitutive response for both 2D and 3D; element assembly and
# stress output come from the generic routines.
kinematics(::Union{Hooke,Hooke2D}) = SmallStrain()

# Isotropic elasticity: C is major-symmetric, so the symmetric tangent loop applies
tangent_symmetry(::Union{Hooke,Hooke2D}) = MajorSymmetric()

function material_response(mp::Union{Hooke,Hooke2D}, ε::SymmetricTensor{2}, state, dt, cache=nothing)
    return mp.C ⊡ ε, mp.C, state
end