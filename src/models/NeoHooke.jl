# The NeoHooke strain energy density follows Ferrite.jl's hyperelasticity tutorial.

"""
    NeoHooke(E, nu)

Compressible Neo–Hookean hyperelastic material parameterized by Young's modulus `E` and Poisson's ratio `nu`.
The constructor stores the equivalent Lamé parameters `μ` (shear modulus) and `λ`.

# References

- Ferrite.jl documentation.
  *Hyperelasticity tutorial.*
  <https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/hyperelasticity/>
"""
struct NeoHooke <: AbstractHyperelastic
    μ::Float64
    λ::Float64

    function NeoHooke(E, ν)
        μ, λ = lame_parameters(E, ν)
        return new(μ, λ)
    end
end

# Only the energy is model-specific; stress, tangent, element assembly, and the
# 2D wrapper interface all come from the AbstractHyperelastic AD defaults.

function Ψ(C, mp::NeoHooke)
    Ic = tr(C)
    J = sqrt(det(C))
    return 0.5mp.μ * (Ic - 3 - 2log(J)) + 0.5mp.λ * (J - 1)^2
end