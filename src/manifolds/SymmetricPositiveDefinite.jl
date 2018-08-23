#
# Manopt.jl – The manifold of symmetric positive definite matrices
# with affine metric.
#
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import LinearAlgebra: svd, norm, dot, Diagonal, eigen
import Base: exp, log, show

export SymmetricPositiveDefinite, SPDPoint, SPDTVector, show
export distance, dot, exp, norm, dot, manifoldDimension, parallelTransport
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
A tangent vector $\xi$ in
$T_x\mathcal M = \{ x^{\frac{1}{2}}\nu x^{\frac{1}{2}}
\big| \nu\in\mathbb R^{n,n}\text{ with }\nu=\nu^{\mathrm{T}}\}$
to the manifold $\mathcal M = \mathcal P(n)$ of $n\times n$ symmetric positive
definite matrices, represented in the redundant way of a skew symmetric
positive definite matrix.
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
distance(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint) = sqrt(sum(log.(abs.(eigen(getValue(x), getValue(y) ).values)).^2))
function dot(M::SymmetricPositiveDefinite, x::SPDPoint, ξ::SPDTVector, ν::SPDTVector)
	svd1 = svd( getValue(x) )
	U = svd1.U
	S = svd1.S
	SInv = Matrix(  Diagonal( 1 ./ diag(S) )  )
	return trace(getValue(ξ) * U*SInv*transpose(U)*getValue(ν)*U*SInv*transpose(U) )
end
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
manifoldDimension(M::SymmetricPositiveDefinite) = M.dimension
manifoldDimension(x::SPDPoint) = size( getValue(x), 1)*(size( getValue(x), 1)+1)/2
norm(M::SymmetricPositiveDefinite,x::SPDPoint,ξ::SPDTVector) = sqrt(dot(M,x,ξ,ξ) )
function parallelTransport(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint,ξ::SPDTVector)
	svd1 = svd( getValue(x) )
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
	return SPDTVector(xSqrt*Uf*Sf*transpose(Uf)*(0.5*(tξ+transpose(tξ)))*Uf*Sf*transpose(Uf)*xSqrt)
end
@doc doc"""
    typicalDistance(M)
returns the typical distance on the
[`SymmetricPositiveDefinite`](@ref) manifold
: $\sqrt{\frac{n(n+1)}{2}}$.
"""
typicalDistance(M::SymmetricPositiveDefinite) = sqrt(M.dimension);
@doc doc"""
    ξ = zeroTVector(M,x)
returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`SPDPoint`](@ref) $x\in\mathcal P(n)$ on the [`SymmetricPositiveDefinite`](@ref) manifold.
"""
zeroTVector(M::SPDPoint, x::SPDPoint) = SPDTVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, M::SymmetricPositiveDefinite) = print(io, "The Manifold $(M.name).")
show(io::IO, p::SPDPoint) = print(io, "SPD($(p.value))")
show(io::IO, ξ::SPDTVector) = print(io, "SPDT($(ξ.value))")
