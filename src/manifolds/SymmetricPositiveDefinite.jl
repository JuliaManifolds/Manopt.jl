#
# Manopt.jl – The manifold of symmetric positive definite matrices
# with affine metric.
#
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import Base: exp, log, show
import LinearAlgebra: norm, dot

using Base: eps
using LinearAlgebra: svd, Diagonal, diag, diagm, eigen, eigvals, eigvecs, tr, triu, qr, cholesky, Hermitian

export SymmetricPositiveDefinite, SPDPoint, SPDTVector, show
export distance, dot, exp, norm, manifoldDimension, parallelTransport
export validateMPoint, validateTVector
export randomMPoint, randomTVector
export typeofMPoint, typeofTVector
export validateMPoint, validateTVector
export zeroTVector
# Types
# ---
@doc doc"""
    SymmetricPositiveDefinite <: Manifold

The manifold $\mathcal M = \mathcal P(n)$ of $n\times n$ symmetric positive
definite matrices.

# Fields
* `name` – representative String representing the current manifold
* `n` – size of the matrices of this manifold, i.e. $n\times n$ matrices
* `abbreviation` – short descriptor for the manifold.

# Constructor
    SymmetricPositiveDefinite(n)

construct the manifold of `n`-by-`n` matrices with affine metric.
"""
struct SymmetricPositiveDefinite <: Manifold
  name::String
  n::Int
  abbreviation::String
  SymmetricPositiveDefinite(n::Int) = new(
      "$n-by-$n symmetric positive definite matrices",n,"SPD($n) affine")
end
@doc doc"""
    SPDPoint <: MPoint

A point $x$ on the [`SymmetricPositiveDefinite`](@ref) manifold
$\mathcal M = \mathcal P(n)$ of $n\times n$,
represented in the redundant way of a symmetric positive definite matrix.
"""
struct SPDPoint{ T <: AbstractFloat} <: MPoint
	value::Matrix{T}
	SPDPoint{T}(v::Matrix{T}) where T<:AbstractFloat = new(v);
end
SPDPoint(v::Matrix{T}) where {T <: AbstractFloat} = SPDPoint{T}(v)
getValue(x::SPDPoint) = x.value
@doc doc"""
    SPDTVector <: TVector

A tangent vector

$\xi \in T_x\mathcal M = \{ x^{\frac{1}{2}}\nu x^{\frac{1}{2}}
\big| \nu\in\mathbb R^{n,n}\text{ with }\nu=\nu^{\mathrm{T}}\}$

to the [`SymmetricPositiveDefinite`](@ref) manifold
$\mathcal M = \mathcal P(n)$ at the [`SPDPoint`](@ref) `x` represented in the
redundant way of a skew symmetric matrix $\nu$, i.e. in the Lie algebra
$T_I\mathcal P(n)$, where $I\in\mathbb R^{n\times n}$ denotes the identity
matrix.
"""
struct SPDTVector{T <: AbstractFloat} <: TVector
	value::Matrix{T}
  	SPDTVector{T}(value::Matrix{T}) where {T <: AbstractFloat} = new(value);
end
SPDTVector(value::Matrix{T}) where {T<:AbstractFloat} = SPDTVector{T}(value)
getValue(ξ::SPDTVector) = ξ.value
# Traits
# ---
# (a) P(n) is a matrix manidolf
@traitimpl IsMatrixM{SymmetricPositiveDefinite}
@traitimpl IsMatrixP{SPDPoint}
@traitimpl IsMatrixTV{SPDTVector}
# Functions
# ---
@doc doc"""
    distance(M,x,y)
    
compute the Riemannian distance on the [`SymmetricPositiveDefinite`](@ref)
manifold $\mathcal M=\mathcal P(n)$, given by

```math
d_{\mathcal P(n)}(x,y) = \lVert \operatorname{Log}(x^{-\frac{1}{2}}yx^{-\frac{1}{2}})\rVert,
```

where $\operatorname{Log}$ denotes the matrix logarithm and the Norm is the Frobenius norm
in matrices
"""
function distance(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint)
    S = log(getValue(x)\getValue(y))
    return real(sqrt(tr(transpose(S)*S)))
end
@doc doc"""
    dot(M,x,ξ,ν)

compute the innter product of the two [`SPDTVector`](@ref)s `ξ,ν` from the tangent
space $T_x\mathcal M$ of the [`SPDPoint`](@ref) `x` on the
[`SymmetricPositiveDefinite`](@ref) manifold `M` given by the formula

$ \langle \xi, \nu \rangle_x = \operatorname{tr}(x^{-1}\xi x^{-1}\nu ),$

where $\operatorname{tr}(y)$ denotes the trace of the matrix $y$.
"""
dot(M::SymmetricPositiveDefinite, x::SPDPoint, ξ::SPDTVector, ν::SPDTVector) = tr( (getValue(x)\getValue(ξ)) * (getValue(x)\getValue(ν)) )

@doc doc"""
    exp(M,x,ξ,[t=1.0])

compute the exponential map on the [`SymmetricPositiveDefinite`](@ref) manifold `M`
 `M`$=\mathcal P(n)$ with respect to the [`SPDPoint`](@ref) `x` and the
[`SPDTVector`](@ref)` ξ`, which can be shortened with `t` to `tξ`.
The formula reads

$\exp_x\xi = x^{\frac{1}{2}}\operatorname{Exp}(x^{-\frac{1}{2}}\xi x^{-\frac{1}{2}})x^{\frac{1}{2}},$

where $\operatorname{Exp}$ denotes the matrix exponential
"""
function exp(M::SymmetricPositiveDefinite, x::SPDPoint, ξ::SPDTVector, t::Float64=1.0)
    S = getValue(x)*real(exp(getValue(x)\(t*getValue(ξ))))
    return SPDPoint( 0.5*(transpose(S)+S) )
end
@doc doc"""
    log(M,x,y)

compute the logarithmic map on the [`SymmetricPositiveDefinite`](@ref) manifold
$\mathcal M=\mathcal P(n)$, i.e. the [`SPDTVector`](@ref) whose corresponding
[`geodesic`](@ref) starting from [`SPDPoint`](@ref) `x` reaches the
[`SPDPoint`](@ref)`y` after time 1.
The formula reads

$\log_x y = x^{\frac{1}{2}}\operatorname{Log}(x^{-\frac{1}{2}} y x^{-\frac{1}{2}})x^{\frac{1}{2}},$

where $\operatorname{Log}$ denotes the matrix logarithm.
"""#
function log(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint)
    S = getValue(x)*real(log(getValue(x)\getValue(y)))
    return SPDTVector( 0.5*(transpose(S) + S) )
end
@doc doc"""
    manifoldDimension(M)

return the manifold dimension of the [`SymmetricPositiveDefinite`](@ref)
[`Manifold`](@ref) `M`, i.e. for $n\times n$ matrices the dimension
is $\frac{n(n+1)}{2}$.
"""
manifoldDimension(M::SymmetricPositiveDefinite) = (M.n+1)*M.n/2
@doc doc"""
    manifoldDimension(x)

returns the manifold dimension of the [`SymmetricPositiveDefinite`](@ref) manifold `M`
manifold the [`SPDPoint`](@ref) `x` belongs to,
i.e. for $n\times n$ matrices the dimension is
$\frac{n(n+1)}{2}$.
"""
manifoldDimension(x::SPDPoint) = Int( size( getValue(x), 1)*(size( getValue(x), 1)+1)/2 )
@doc doc"""
    norm(M,x,ξ)

compute the norm of the [`SPDTVector`](@ref)` ξ` from the tangent space $T_x\mathcal M$
at the [`SPDPoint`](@ref) `x` on the [`SymmetricPositiveDefinite`](@ref) manifold `M`
induced by the inner product [`dot`](@ref) as $\lVert\xi\rVert_x = \sqrt{\langle\xi,\xi\rangle_x}$.
"""
function norm(M::SymmetricPositiveDefinite,x::SPDPoint,ξ::SPDTVector)
    S = getValue(x)\getValue(ξ)
    return real(sqrt(tr( transpose(S)*S )))
end
@doc doc"""
    parallelTransport(M,x,y,ξ)

compute the paralllel transport of the [`SPDTVector`](@ref)` ξ` from
the tangent space $T_x\mathcal M$ at [`SPDPoint`](@ref) `x` to
$T_y\mathcal M$ at [`SPDPoint`](@ref)`y` on the [`SymmetricPositiveDefinite`](@ref) manifold `M`
along the [`geodesic`](@ref) $g(\cdot;x,y)$.
The formula reads

$P_{x\to y}(\xi) = x^{\frac{1}{2}}
\operatorname{Exp}\bigl(
\frac{1}{2}x^{-\frac{1}{2}}\log_x(y)x^{-\frac{1}{2}}
\bigr)
x^{-\frac{1}{2}}\xi x^{-\frac{1}{2}}
\operatorname{Exp}\bigl(
\frac{1}{2}x^{-\frac{1}{2}}\log_x(y)x^{-\frac{1}{2}}
\bigr)
x^{\frac{1}{2}},$

where $\operatorname{Exp}$ denotes the matrix exponential
and `log` the logarithmic map.
"""
function parallelTransport(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint,ξ::SPDTVector)
  svd1 = svd( getValue(x) )
  if norm(getValue(x)-getValue(y))<1e-13
    return ξ
  end
  U = svd1.U
  S = svd1.S
  Ssqrt = sqrt.(S)
  SsqrtInv = Matrix(  Diagonal( 1 ./ Ssqrt )  )
  Ssqrt = Matrix(  Diagonal( Ssqrt )  )
  xSqrt = U*Ssqrt*transpose(U)
  xSqrtInv = U*SsqrtInv*transpose(U)
  tξ = xSqrtInv * getValue(ξ) * xSqrtInv
  tY = xSqrtInv * getValue(y) * xSqrtInv
  svd2 = svd(tY)
  Se = Matrix(  Diagonal( log.(svd2.S) )  )
  Ue = svd2.U
  tY2 = Ue*Se*transpose(Ue)
  eig1 = eigen(0.5*tY2)
  Sf = Matrix(  Diagonal( exp.(eig1.values) )  )
  Uf = eig1.vectors
  ν = xSqrt*Uf*Sf*transpose(Uf)*(0.5*(tξ+transpose(tξ)))*Uf*Sf*transpose(Uf)*xSqrt
  return SPDTVector(ν)
end

@doc doc"""
    randomMPoint(M,:Gaussian[, σ=1.0])

gerenate a random symmetric positive definite matrix on the
[`SymmetricPositiveDefinite`](@ref) manifold `M`.
"""
function randomMPoint(M::SymmetricPositiveDefinite,::Val{:Gaussian},σ::Float64=1.0)
    D = Diagonal( 1 .+ randn(M.n) ) # random diagonal matrix
    s = qr(σ * randn(M.n,M.n)) # random q
    return SPDPoint(s.Q*D* transpose(s.Q))
end
@doc doc"""
    randomTVector(M,x [,:Gaussian,σ = 1.0])

generate a random tangent vector in the tangent space of the [`SPDPoint`](@ref)
`x` on the [`SymmetricPositiveDefinite`](@ref) manifold `M` by using a Gaussian distribution
with standard deviation `σ` on an ONB of the tangent space.
"""
function randomTVector(M::SymmetricPositiveDefinite, x::SPDPoint, ::Val{:Gaussian}, σ::Float64 = 0.01)
    # generate ONB in TxM
    I = SPDPoint(one(zeros(M.n,M.n)))
    Ξ,κ = tangentONB(M,I,SPDTVector(one(zeros(M.n,M.n))))
    Ξx = parallelTransport.(Ref(M), Ref(I), Ref(x), Ξ)
    return sum( randn(length(Ξx)) .* Ξx )
end
@doc doc"""
    randomTVector(M,x [,:Gaussian,σ = 1.0])

generate a random tangent vector in the tangent space of the [`SPDPoint`](@ref)
`x` on the [`SymmetricPositiveDefinite`](@ref) manifold `M` by using a Rician distribution
with standard deviation `σ`.
"""
function randomTVector(M::SymmetricPositiveDefinite, x::SPDPoint, ::Val{:Rician}, σ::Real = 0.01)
    # Rician
    C = cholesky( Hermitian(getValue(x)) )
    R = sqrt(σ) * triu( randn(M.n,M.n),0)
    T = C.L * transpose(R)*R*C.U
    return log(M,x, SPDPoint(T))
end

@doc doc"""
    Ξ,κ = tangentONB(M,x,y)

compute a ONB in the tangent space of the [`SPDPoint`](@ref) `x` on the
[`SymmetricPositiveDefinite`](@ref) manifold `M` where the first vector is given by the
normed `log(M,x,y)`, i.e. the direction to the [`SPDPoint`](@ref) `y`.
"""
tangentONB(M::SymmetricPositiveDefinite, x::SPDPoint, y::SPDPoint) = tangentONB(M,x,log(M,x,y))
@doc doc"""
    Ξ,κ = tangentONB(M,x,y)

compute a ONB in the tangent space of the [`SPDPoint`](@ref) `x` on the
[`SymmetricPositiveDefinite`](@ref) manifold `M` where the first
vector is the normed tangent vector of the [`SPDTVector`](@ref) `ξ`.

The basis is computed using the eigenvectors $v_i$, $i=1,\ldots,n$, of `ξ` and
define

```math
\xi_{ij} := \begin{cases}
    \frac{1}{2} (v_i v_j^{\mathrm{T}} + v_j v_i^{\mathrm{T}})
    & \mathrm{ if } i=j	,\\
    \frac{1}{\sqrt{2}} (v_i v_j^{\mathrm{T}} + v_j v_i^{\mathrm{T}}),
    & \mathrm{ if } i \neq j,
\end{cases} \qquad\text{where } i=1,\ldots,n, k=i,\ldots,n,
```

and the correspoinding eigenvalues of the curvature tensor are given using the
eigenvalues $\lambda_i$, $i=1,\ldots,n$ of `ξ` as

```math
\kappa_{i,j} = -\frac{1}{4}(\lambda_i-\lambda_j)^2,\qquad
\text{where } i=1,\ldots,n, k=i,\ldots,n,
```
"""
function tangentONB(M::SymmetricPositiveDefinite,x::SPDPoint,ξ::SPDTVector)
    xSqrt = sqrt(getValue(x)) 
    v = eigvecs(getValue(ξ))
    Ξ = [
        SPDTVector( (i==j ? 1/2 : 1/sqrt(2))*
            ( v[:,i] * transpose(v[:,j])  +  v[:,j] * transpose(v[:,i]) )
        )
        for i=1:M.n for j= i:M.n
    ]
    λ = eigvals(getValue(ξ))
    κ = [ -1/4 * (λ[i]-λ[j])^2 for i=1:M.n for j= i:M.n ]
  return Ξ,κ
end

typeofTVector(::Type{SPDPoint{T}}) where T = SPDTVector{T}
typeofMPoint(::Type{SPDTVector{T}}) where T = SPDPoint{T} 

@doc doc"""
    typicalDistance(M)

returns the typical distance on the [`SymmetricPositiveDefinite`](@ref)
manifold `M` $\sqrt{\frac{n(n+1)}{2}}$.
"""
typicalDistance(M::SymmetricPositiveDefinite) = sqrt(manifoldDimension(M));

@doc doc"""
    validateMPoint(M,x)

validate that the [`SPDPoint`](@ref) `x` is a valid point on the manifold
[`SymmetricPositiveDefinite`](@ref) manifold `M`, i.e. the matrix is symmetric and
positive definite.
"""
function validateMPoint(M::SymmetricPositiveDefinite, x::SPDPoint)
      if manifoldDimension(M) ≠ manifoldDimension(x)
        throw(DomainError(
            "The point $x does not lie on $M,, since the manifold dimension of $M ($(manifoldDimension(M)))does not fit the manifold dimension of $x ($(manifoldDimension(x)))."
        ))
    end
    if norm(getValue(x) - transpose(getValue(x))) > 10^(-14)
        throw(DomainError(
            "The point $x does not lie on $M, since the matrix of $x is not symmetric."
        ))
    end
    e = eigvals(getValue(x))
    if  !( (all(isreal.(e))) && (all( e .> 0)) ) # not spd
        throw(DomainError(
            "The point $x does not lie on $M, since the matrix of $x is not symmetric positive definite."
        ))
    end
    return true
end

@doc doc"""
    validateTVector(M,x,ξ)

validate, that the [`SPDTVector`](@ref)` ξ` is a tangent vector at the
[`SPDPoint`](@ref) `x` on the [`SymmetricPositiveDefinite`](@ref) `M`,
i.e. all dimensions are corrrect and the matrix is skew symmetric since
we only store the corresponding value in the Lie algebra.
"""
function validateTVector(M::SymmetricPositiveDefinite,x::SPDPoint,ξ::SPDTVector)
    if !validateMPoint(M,x)
        return false
    end
    if size(getValue(x)) != size(getValue(ξ))
        throw(DomainError(
            "The tangent vector $ξ can not be a tangent vector to $x (on $M), since the size of the matrix of ξ ($(size(getValue(ξ))) does not match the size of its base point matrix ($(size(getValue(x))))."
        ))
    end
    if norm(getValue(ξ) - transpose(getValue(ξ))) > 10^(-14)
        throw(DomainError(
            "The tangent vector $ξ does not represent a tangent vector (to $x) on $M, since the matrix of $ξ is not symmetric."
        ))
    end
    return true
end

@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`SPDPoint`](@ref) $x\in\mathcal P(n)$ on the
[`SymmetricPositiveDefinite`](@ref) manifold `M`.
"""
zeroTVector(M::SymmetricPositiveDefinite, x::SPDPoint) = SPDTVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, M::SymmetricPositiveDefinite) = print(io, "The Manifold of $(M.name)")
show(io::IO, x::SPDPoint) = print(io, "SPD($(x.value))")
show(io::IO, ξ::SPDTVector) = print(io, "SPDT($(ξ.value))")
