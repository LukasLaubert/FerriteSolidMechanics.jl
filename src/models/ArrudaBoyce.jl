# ArrudaBoyce has a model-specific second-Piola stress and material tangent,
# but shares the generic second-Piola-to-PK1 mapping with other hyperelastic models.

"""
    ArrudaBoyce(μ, κ, N; tangent=:AT)

Compressible Arruda–Boyce eight-chain hyperelastic material.
The network shear modulus `μ` and the limiting chain extensibility `N` parameterize the isochoric network response.
The bulk modulus `κ` parameterizes the volumetric response.

### Keyword arguments

- `tangent` – chooses how the model forms `∂P/∂F`: `:AT` (default) builds it from the analytic second Piola–Kirchhoff stress and its derivative, `:AD` obtains it by automatic differentiation of the PK1 map

# References

- E. M. Arruda, M. C. Boyce.
  *A three-dimensional constitutive model for the large stretch behavior
  of rubber elastic materials.*
  Journal of the Mechanics and Physics of Solids **41**(2) (1993) 389-412.
  <https://doi.org/10.1016/0022-5096(93)90013-6>
"""
struct ArrudaBoyce{Tangent} <: AbstractHyperelastic
    μ::Float64
    κ::Float64
    N::Float64
    tangent::Symbol

    function ArrudaBoyce(μ, κ, N; tangent=:AT)
        μ > 0.0 || throw(ArgumentError("ArrudaBoyce: μ must be positive (got μ=$μ)"))
        κ > 0.0 || throw(ArgumentError("ArrudaBoyce: κ must be positive (got κ=$κ)"))
        N > 0.0 || throw(ArgumentError("ArrudaBoyce: N must be positive (got N=$N)"))
        tangent in _ARRUDA_BOYCE_TANGENT_OPTIONS ||
            throw(ArgumentError("ArrudaBoyce: tangent must be one of " * repr(_ARRUDA_BOYCE_TANGENT_OPTIONS)))
        return new{tangent}(μ, κ, N, tangent)
    end
end

const _ARRUDA_BOYCE_TANGENT_OPTIONS = (:AT, :AD)

# Generic API methods

@inline function _arruda_boyce_dWdI1bar(I1bar, mp::ArrudaBoyce)
    c1, c2, c3, c4, c5 = 1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750
    N = mp.N
    return mp.μ * (
        c1 +
        2 * c2 / N * I1bar +
        3 * c3 / N^2 * I1bar^2 +
        4 * c4 / N^3 * I1bar^3 +
        5 * c5 / N^4 * I1bar^4
    )
end

@inline function _arruda_boyce_d2WdI1bar2(I1bar, mp::ArrudaBoyce)
    _, c2, c3, c4, c5 = 1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750
    N = mp.N
    return mp.μ * (
        2 * c2 / N +
        6 * c3 / N^2 * I1bar +
        12 * c4 / N^3 * I1bar^2 +
        20 * c5 / N^4 * I1bar^3
    )
end

function Ψ(C, mp::ArrudaBoyce)
    c1, c2, c3, c4, c5 = 1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750
    I1bar = det(C)^(-1 / 3) * tr(C)
    N = mp.N
    Ψiso = mp.μ * (
        c1 * (I1bar - 3) +
        c2 / N * (I1bar^2 - 3^2) +
        c3 / N^2 * (I1bar^3 - 3^3) +
        c4 / N^3 * (I1bar^4 - 3^4) +
        c5 / N^4 * (I1bar^5 - 3^5)
    )
    J = sqrt(det(C))
    Ψvol = 0.5 * mp.κ * (J - 1)^2
    return Ψiso + Ψvol
end

function _arruda_boyce_second_piola(C::SymmetricTensor{2,3,T}, mp::ArrudaBoyce) where T
    I = one(C)
    Cinv = inv(C)
    I1 = tr(C)
    J2 = det(C)
    J = sqrt(J2)
    J2m13 = J2^(-1 / 3)
    I1bar = J2m13 * I1

    dWdI = _arruda_boyce_dWdI1bar(I1bar, mp)
    B = I - (I1 / 3) * Cinv
    q = 2 * dWdI * J2m13
    p = mp.κ * (J - 1) * J
    return q * B + p * Cinv
end

@inline _second_piola(C::SymmetricTensor{2,3,T}, mp::ArrudaBoyce) where T = _arruda_boyce_second_piola(C, mp)

function constitutive_driver(C::SymmetricTensor{2,3,T}, mp::ArrudaBoyce) where T
    I = one(C)
    Cinv = inv(C)
    I1 = tr(C)
    J2 = det(C)
    J = sqrt(J2)
    J2m13 = J2^(-1 / 3)
    I1bar = J2m13 * I1

    dWdI = _arruda_boyce_dWdI1bar(I1bar, mp)
    d2WdI2 = _arruda_boyce_d2WdI1bar2(I1bar, mp)

    B = I - (I1 / 3) * Cinv
    q = 2 * dWdI * J2m13
    dq_dC = 2 * J2m13 * (d2WdI2 * J2m13 * B - (dWdI / 3) * Cinv)

    p = mp.κ * (J - 1) * J
    dp_dC = 0.5 * mp.κ * J * (2 * J - 1) * Cinv

    S = q * B + p * Cinv
    dS_dC = Tensor{4,3,T}((i, j, k, l) -> begin
        inv_sym = 0.5 * (Cinv[i, k] * Cinv[l, j] + Cinv[i, l] * Cinv[k, j])
        dB = -(1 / 3) * I[k, l] * Cinv[i, j] + (I1 / 3) * inv_sym
        dCinv = -inv_sym
        B[i, j] * dq_dC[k, l] + q * dB + Cinv[i, j] * dp_dC[k, l] + p * dCinv
    end)

    return S, dS_dC
end

function _arruda_boyce_pk1_tangent(F::Tensor{2,3,T}, mp::ArrudaBoyce{:AT}) where T
    return _pk1_tangent_from_second_piola(F, mp)
end

function _arruda_boyce_pk1_tangent(F::Tensor{2,3,T}, mp::ArrudaBoyce{:AD}) where T
    dP_dF, P = Tensors.gradient(F_ -> _pk1_from_second_piola(F_, mp), F, :all)
    return P, dP_dF
end

@inline _finite_strain_pk1_tangent(F::Tensor{2,3,T}, mp::ArrudaBoyce, dt) where T = _arruda_boyce_pk1_tangent(F, mp)
