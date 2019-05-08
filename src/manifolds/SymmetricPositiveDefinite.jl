#
# Manopt.jl – The manifold of symmetric positive definite matrices
# with affine metric.
#
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import LinearAlgebra: svd, norm, dot, Diagonal, diag, eigen, eigvals, tr, qr
import Base: exp, log, show, eps

export SymmetricPositiveDefinite, SPDPoint, SPDTVector, show
export distance, dot, exp, norm, dot, manifoldDimension, parallelTransport
export randomMPoint, randomTVector
export validateMPoint, validateTVector
export zeroTVector
# Types
# ---
@doc doc"""
    SymmetricPositiveDefinite <: Manifold
The manifold $\mathcal M = \mathcal P(n)$ of $n\times n$ symmetric positive
definite matrices.
"""
struct SymmetricPositiveDefinite <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  SymmetricPositiveDefinite(dimension::Int) = new("$dimension-by-$dimension symmetric positive definite matrices",(dimension*(dimension+1)/2),"SPD($dimension) affine")
end
@doc doc"""
    SPDPoint <: MPoint
A point $x$ on the manifold $\mathcal M = \mathcal P(n)$ of $n\times n$
symmetric positive definite matrices, represented in the redundant way of a
symmetric positive definite matrix.
"""
struct SPDPoint <: MPoint
	value::Matrix{Float64}
	SPDPoint(v::Matrix{Float64}) = new(v);
end
getValue(x::SPDPoint) = x.value
@doc doc"""
    SPDTVector <: TVector
A tangent vector

$\xi \in T_x\mathcal M = \{ x^{\frac{1}{2}}\nu x^{\frac{1}{2}}
\big| \nu\in\mathbb R^{n,n}\text{ with }\nu=\nu^{\mathrm{T}}\}$

to the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref)
$\mathcal M = \mathcal P(n)$ at the [`SPDPoint`](@ref) `x` represented in the
redundant way of a skew symmetric matrix $\nu$, i.e. in the Lie Algebra
$T_I\mathcal P(n)$, where $I\in\mathbb R^{n\times n}$ denotes the identity
matrix.
"""
struct SPDTVector <: TVector
	value::Matrix{Float64}
  	SPDTVector(value::Matrix{Float64}) = new(value);
end
getValue(ξ::SPDTVector) = ξ.value
# Traits
# ---
# (a) P(n) is a matrix manidolf
@traitimpl IsMatrixM{SymmetricPositiveDefinite}
@traitimpl IsMatrixP{SPDPoint}
@traitimpl IsMatrixV{SPDTVector}
# Functions
# ---
@doc doc"""
	distance(M,x,y)
Compute the Riemannian distance on the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) $\mathcal M=\mathcal P(n)$, can be computed as

$ d_{\mathcal P(n)}(x,y) = \lVert \operatorname{Log}(x^{-\frac{1}{2}}yx^{-\frac{1}{2}})\rVert, $

where $\operatorname{Log}$ denotes the matrix logarithm and the Norm is the Frobenius norm
in matrices
"""
distance(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint) = sqrt(sum(log.(abs.(eigen(getValue(x), getValue(y) ).values)).^2))
function dot(M::SymmetricPositiveDefinite, x::SPDPoint, ξ::SPDTVector, ν::SPDTVector)
	svd1 = svd( getValue(x) )
	U = svd1.U
	S = svd1.S
	SInv = Matrix(  Diagonal( 1 ./ diag(S) )  )
	return tr(getValue(ξ) * U*SInv*transpose(U)*getValue(ν)*U*SInv*transpose(U) )
end
@doc doc"""
    dot(M,x,ξ,ν)
compute the innter product of the two [`SPDTVector`](@ref)`s ξ,ν` from the tangent
space $T_x\mathcal M$ of the [`SPDPoint`](@ref) `x` on the
[`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M` given by the formula

$ \langle \xi, \nu \rangle_x = \operatorname{tr}(x^{-1}\xi x^{-1}\nu ),$

where $\operatorname{tr}(y)$ denotes the trace of the matrix $y$.
"""
dot(M,x,ξ,ν) = tr( (x\ξ)*(x\ν) ) #use \ instead of inversion

@doc doc"""
    exp(M,x,ξ[, t=1.0])
Compute the exponential map on the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`
 `M`$=\mathcal P(n)$ with respect to the [`SPDPoint`](@ref) `x` and the
[`SPDTVector`](@ref) `ξ`, which can be shortened with `t` to `tξ`.
The formula reads

$\exp_x\xi = x^{\frac{1}{2}}\operatorname{Exp}(x^{-\frac{1}{2}}\xi x^{-\frac{1}{2}})x^{\frac{1}{2}},$

where $\operatorname{Exp}$ denotes the matrix exponential
"""
function exp(M::SymmetricPositiveDefinite, x::SPDPoint, ξ::SPDTVector, t::Float64=1.0)
	svd1 = svd( getValue(x) );
	U = svd1.U;
	S = copy(svd1.S);
	Ssqrt = sqrt.(S);
	SsqrtInv = Matrix(  Diagonal( 1 ./ Ssqrt ));
	pSqrt = U*Matrix(  Diagonal( Ssqrt )  )*transpose(U);
  	T = U*SsqrtInv*transpose(U)*(t.*ξ.value)*U*SsqrtInv*transpose(U);
    svd2 = svd(T);
   	Se = Matrix(  Diagonal( exp.(svd2.S) )  )
  	Ue = svd2.U
	return SPDPoint(pSqrt*Ue*Se*transpose(Ue)*pSqrt)
end
@doc doc"""
    log(M,x,y)
Compute the logarithmic map on the [`SymmetricPositiveDefinite`](@ref)
$\mathcal M=\mathcal P(n)$, i.e. the [`SPDTVector`](@ref) whose corresponding
[`geodesic`](@ref) starting from [`SPDPoint`](@ref) `x` reaches the
[`SPDPoint`](@ref)` y` after time 1 on the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`.
The formula reads for

$\log_x y = x^{\frac{1}{2}}\operatorname{Log}(x^{-\frac{1}{2}} y x^{-\frac{1}{2}})x^{\frac{1}{2}},$

where $\operatorname{Log}$ denotes the matrix logarithm.
"""#
function log(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint)
	svd1 = svd( getValue(x) )
	U = svd1.U
	S = svd1.S
	Ssqrt = sqrt.(S)
	SsqrtInv = Matrix(  Diagonal( 1 ./ Ssqrt )  )
	Ssqrt = Matrix(  Diagonal( Ssqrt )  )
  	pSqrt = U*Ssqrt*transpose(U)
	T = U * SsqrtInv * transpose(U) * getValue(y) * U * SsqrtInv * transpose(U)
	svd2 = svd(T)
	Se = Matrix(  Diagonal( log.(svd2.S) )  )
	Ue = svd2.U
	ξ = pSqrt*Ue*Se*transpose(Ue)*pSqrt
	return SPDTVector(ξ)
end
@doc doc"""
    manifoldDimension(M)
returns the manifold dimension of the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`, i.e. for $n\times n$ matrices the dimension
is $d_{\mathcal P(n)} = \frac{n(n+1)}{2}$.
"""
manifoldDimension(M::SymmetricPositiveDefinite) = M.dimension
@doc doc"""
    manifoldDimension(x)
returns the manifold dimension of the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`
manifold the [`SPDPoint`](@ref) `x` belongs to,
i.e. for $n\times n$ matrices the dimension is
$d_{\mathcal P(n)} = \frac{n(n+1)}{2}$.
"""
manifoldDimension(x::SPDPoint) = size( getValue(x), 1)*(size( getValue(x), 1)+1)/2
@doc doc"""
    norm(M,x,ξ)
Computes the norm of the [`SPDTVector`](@ref) `ξ` from the tangent space $T_x\mathcal M$
at the [`SPDPoint`](@ref) `x` on the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`
induced by the inner product [`dot`](@ref) as $\lVert\xi\rVert_x = \sqrt{\langle\xi,\xi\rangle_x}$.
"""
norm(M::SymmetricPositiveDefinite,x::SPDPoint,ξ::SPDTVector) = sqrt(dot(M,x,ξ,ξ) )
@doc doc"""
    parallelTransport(M,x,y,ξ)
Compute the paralllel transport of the [`SPDTVector`](@ref) `ξ` from
the tangent space $T_x\mathcal M$ at [`SPDPoint`](@ref) `x` to
$T_y\mathcal M$ at [`SPDPoint`](@ref)` y` on the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`
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
    randomMPoint(M,n)

gerenate a random symmetric positive definite matrix on the
[`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`.
"""
function randomMPoint(M::SymmetricPositiveDefinite)
    D = Diagonal( 1 .+ randn(M.dimension) ) # random diagonal matrix
    s = qr(randn(M.dimension,M.dimension)) # random q
    return SPDPoint(s.Q*D* transpose(s.Q))
end
@doc doc"""
    randomTVector(M,x)

generate a random [`SPDTVector`](@ref) in the tangent space of the
[`SPDPoint`](@ref) `x` on the
[`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`.
"""
function randomTVector(M::SymmetricPositiveDefinite, x::SPDPoint)
    ξ = randn(M.dimension,M.dimension)
    return SPDTVector( 0.5 * ( ξ + transpose(ξ) ) )
end

tangentONB(M::SymmetricPositiveDefinite, x::SPDPoint, y::SPDPoint) = tangentONB(M,x,log(M,x,y))
function tangentONB(M::SymmetricPositiveDefinite,x::SPDPoint,ξ::SPDTVector)
  svd1 = svd( getValue(x) )
  S = sqrt.(svd1.S)
  U = svd1.U
  SInv = Matrix(  Diagonal( 1 ./ S )  )
  S = Matrix(  Diagonal( S )  )
  xSqrt = U * S * transpose(U)
  xSqrtInv = U * SInv * transpose(U)
  Ξ = Array{SPDTVector}(
    fill( SPDTVector(zero(getValue(ξ))) , manifoldDimension(x))
  )
  if !all(  vec( S .== 0 )  )
    T = xSqrtInv * getValue(ξ) * xSqrtInv
    eig = eigen(1/2 * (T + transpose(T) ) );
    U = real( eig.vectors ) # cpointV
	λ = eig.values
    s = 1;
    κ = zeros( manifoldDimension(x) )
    for i = 1 : size(S,1)
      for j = i : size(S,2)
        κ[s] = -1/4 * abs( λ[i] - λ[j] ).^2; # norm(T)
        t = (i!=j) ? sqrt(0.5) : 0.5;
        V = t * ( U[:,i] * transpose(U[:,j]) + U[:,j] * transpose(U[:,i]) )
        Ξ[s] = SPDTVector(  xSqrt * V * xSqrt )
        s = s+1
      end
    end
  end
  return Ξ,κ
end
@doc doc"""
    typicalDistance(M)
returns the typical distance on the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M` $\sqrt{\frac{n(n+1)}{2}}$.
"""
typicalDistance(M::SymmetricPositiveDefinite) = sqrt(M.dimension);
@doc doc"""
    validateMPoint(M,x)

validate that the [`SPDPoint`](@ref) `x` is a valid point on the manifold
[`SymmetricPositiveDefinite`](@ref) `M`, i.e. the matrix is symmetric and
positive definite.
"""
function validateMPoint(M::SymmetricPositiveDefinite, x::SPDPoint)
      if manifoldDimension(M) ≠ manifoldDimension(x)
        throw(ErrorException(
            "The point $x does not lie on $M,, since the manifold dimension of $M ($(manifoldDimension(M)))does not fit the manifold dimension of $x ($(manifoldDimension(x)))."
        ))
    end
    if norm(getValue(x) - transpose(getValue(x))) > 10^(-14)
        throw(ErrorException(
            "The point $x does not lie on $M, since the matrix of $x is not symmetric."
        ))
    end
    e = eigvals(getValue(x))
    if  !( (all(isreal.(e))) && (all( e .> 0)) ) # not spd
        throw(ErrorException(
            "The point $x does not lie on $M, since the matrix of $x is not symmetric positive definite."
        ))
    end
    return true
end
@doc doc"""
    validateTVector(M,x,ξ)

validate, that the [`SPDTVector`](@ref) `ξ` is a tangent vector at the
[`SPDPoint`](@ref) `x` on the [`SymmetricPositiveDefinite`](@ref) `M`,
i.e. all dimensions are corrrect and the matrix is skew symmetric since
we only store the corresponding value in the Lie Algebra 
"""

@doc doc"""
    ξ = zeroTVector(M,x)
returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`SPDPoint`](@ref) $x\in\mathcal P(n)$ on the [`SymmetricPositiveDefinite`](@ref)` `[`Manifold`](@ref) `M`.
"""
zeroTVector(M::SPDPoint, x::SPDPoint) = SPDTVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, M::SymmetricPositiveDefinite) = print(io, "The Manifold $(M.name).")
show(io::IO, p::SPDPoint) = print(io, "SPD($(p.value))")
show(io::IO, ξ::SPDTVector) = print(io, "SPDT($(ξ.value))")
