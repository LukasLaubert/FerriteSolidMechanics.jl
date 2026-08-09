"""
    MooneyRivlin(C10, C01, κ; tangent=:AD)

Compressible Mooney–Rivlin hyperelastic material.
The isochoric response is linear in the first two modified invariants.

### Parameters

- `C10` – Coefficient of the first modified invariant
- `C01` – Coefficient of the second modified invariant
- `κ` – Bulk modulus; parameterizes the volumetric response

### Keyword arguments

- `tangent` – chooses how the model forms `∂P/∂F`: `:AD` (default) obtains it by automatic differentiation of the PK1 map, `:AT` builds it from the analytic second Piola–Kirchhoff stress and its derivative

# References

- M. Mooney.
  *A theory of large elastic deformation.*
  Journal of Applied Physics **11**(9) (1940) 582-592.
  <https://doi.org/10.1063/1.1712836>
- R. S. Rivlin.
  *Large elastic deformations of isotropic materials. IV. further developments of the general theory.*
  Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences **241**(835) (1948) 379-397.
  <https://doi.org/10.1098/rsta.1948.0024>
"""
struct MooneyRivlin{Tangent} <: AbstractHyperelastic
    C10::Float64
    C01::Float64
    κ::Float64
    tangent::Symbol

    function MooneyRivlin(C10, C01, κ; tangent=:AD)
        if !isfinite(C10)
            throw(ArgumentError("MooneyRivlin: C10 must be finite (got C10=$C10)"))
        end
        if !isfinite(C01)
            throw(ArgumentError("MooneyRivlin: C01 must be finite (got C01=$C01)"))
        end
        if !isfinite(κ)
            throw(ArgumentError("MooneyRivlin: κ must be finite (got κ=$κ)"))
        end
        if !isless(0.0, C10 + C01)
            throw(ArgumentError("MooneyRivlin: C10 + C01 must be positive (got C10 + C01=$(C10 + C01))"))
        end
        if !isless(0.0, κ)
            throw(ArgumentError("MooneyRivlin: κ must be positive (got κ=$κ)"))
        end
        if !(tangent in _MOONEY_RIVLIN_TANGENT_OPTIONS)
            throw(ArgumentError("MooneyRivlin: tangent must be one of " * repr(_MOONEY_RIVLIN_TANGENT_OPTIONS)))
        end
        return new{tangent}(C10, C01, κ, tangent)
    end
end

const _MOONEY_RIVLIN_TANGENT_OPTIONS = (:AT, :AD)

# Generic API methods

@inline function _mooney_rivlin_I2(C)
    I1 = tr(C)
    return 0.5 * (I1^2 - C ⊡ C)
end

function Ψ(C, mp::MooneyRivlin)
    J2 = det(C)
    I1 = tr(C)
    I2 = _mooney_rivlin_I2(C)
    I1bar = J2^(-1 / 3) * I1
    I2bar = J2^(-2 / 3) * I2
    J = sqrt(J2)
    return mp.C10 * (I1bar - 3) + mp.C01 * (I2bar - 3) + 0.5 * mp.κ * (J - 1)^2
end

function _mooney_rivlin_second_piola(C::SymmetricTensor{2,3,T}, mp::MooneyRivlin) where T
    I = one(C)
    Cinv = inv(C)
    I1 = tr(C)
    I2 = _mooney_rivlin_I2(C)
    J2 = det(C)
    J = sqrt(J2)
    J2m13 = J2^(-1 / 3)
    J2m23 = J2^(-2 / 3)

    B1 = I - (I1 / 3) * Cinv
    B2 = I1 * I - C - (2 * I2 / 3) * Cinv
    p = mp.κ * (J - 1) * J

    return 2 * mp.C10 * J2m13 * B1 + 2 * mp.C01 * J2m23 * B2 + p * Cinv
end

@inline _second_piola(C::SymmetricTensor{2,3,T}, mp::MooneyRivlin) where T = _mooney_rivlin_second_piola(C, mp)

function constitutive_driver(C::SymmetricTensor{2,3,T}, mp::MooneyRivlin) where T
    I = one(C)
    Cinv = inv(C)
    I1 = tr(C)
    I2 = _mooney_rivlin_I2(C)
    J2 = det(C)
    J = sqrt(J2)
    J2m13 = J2^(-1 / 3)
    J2m23 = J2^(-2 / 3)

    B1 = I - (I1 / 3) * Cinv
    B2 = I1 * I - C - (2 * I2 / 3) * Cinv

    q1 = 2 * mp.C10 * J2m13
    q2 = 2 * mp.C01 * J2m23
    dq1_dC = -(q1 / 3) * Cinv
    dq2_dC = -(2 * q2 / 3) * Cinv

    p = mp.κ * (J - 1) * J
    dp_dC = 0.5 * mp.κ * J * (2 * J - 1) * Cinv

    S = q1 * B1 + q2 * B2 + p * Cinv
    dS_dC = Tensor{4,3,T}((i, j, k, l) -> begin
        sym_id = 0.5 * (I[i, k] * I[j, l] + I[i, l] * I[j, k])
        inv_sym = 0.5 * (Cinv[i, k] * Cinv[l, j] + Cinv[i, l] * Cinv[k, j])
        dI2_dC = I1 * I[k, l] - C[k, l]
        dB1 = -(1 / 3) * I[k, l] * Cinv[i, j] + (I1 / 3) * inv_sym
        dB2 = I[k, l] * I[i, j] - sym_id - (2 / 3) * dI2_dC * Cinv[i, j] + (2 * I2 / 3) * inv_sym
        dCinv = -inv_sym
        B1[i, j] * dq1_dC[k, l] + q1 * dB1 +
        B2[i, j] * dq2_dC[k, l] + q2 * dB2 +
        Cinv[i, j] * dp_dC[k, l] + p * dCinv
    end)

    return S, dS_dC
end

function _mooney_rivlin_pk1_tangent(F::Tensor{2,3,T}, mp::MooneyRivlin{:AT}) where T
    return _pk1_tangent_from_second_piola(F, mp)
end

function _mooney_rivlin_pk1_tangent(F::Tensor{2,3,T}, mp::MooneyRivlin{:AD}) where T
    dP_dF, P = Tensors.gradient(F_ -> _pk1_from_second_piola(F_, mp), F, :all)
    return P, dP_dF
end

@inline _finite_strain_pk1_tangent(F::Tensor{2,3,T}, mp::MooneyRivlin, dt) where T = _mooney_rivlin_pk1_tangent(F, mp)
