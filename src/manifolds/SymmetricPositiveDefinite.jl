#
# Manopt.jl – The manifold of symmetric positive definite matrices
# with affine metric.
#
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import Base.LinAlg: svd, norm, dot
import Base: exp, log, show

export SymmetricPositiveDefinite, SPDPoint, SPDTVector, show
export distance, dot, exp, norm, dot, manifoldDimension, parallelTransport
# Types
# ---
doc"""
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
doc"""
    SPDPoint <: SPDPoint
A point $x$ on the manifold $\mathcal M = \mathcal P(n)$ of $n\times n$
symmetric positive definite matrices, represented in the redundant way of a
symmetric positive definite matrix.
"""
struct SPDPoint <: MPoint
	value::Matrix{Float64}
	SPDPoint(v::Matrix{Float64}) = new(v);
end
getValue(x::SPDPoint) = x.value
doc"""
    SPDTVector <: TVector
A tangent vector $\xi$ in
$T_x\mathcal M = \{ x^{\frac{1}{2}\nu x^{\frac{1}{2}} \big| \nu\in\mathbb R^{n,n}\text{ with }\nu=\nu^\tT\}$
to the manifold $\mathcal M = \mathcal P(n)$ of $n\times n$ symmetric positive
definite matrices, represented in the redundant way of a skew symmetrymmetric positive definite matrix.
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
distance(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint) = sqrt(sum(log.(abs.(eig(getValue(x), getValue(y) )[1])).^2))
function dot(M::SymmetricPositiveDefinite, x::SPDPoint, ξ::SPDTVector, ν::SPDTVector)
	svd1 = svd( getValue(x) )
	U = svd1[1]
	S = svd1[2]
	SInv = diagm(1./diag(S))
	return trace(getValue(ξ) * U*SInv*U.'*getValue(ν)*U*SInv*U.' )
end
function exp(M::SymmetricPositiveDefinite,x::SPDPoint,ξ::SPDTVector,t::Float64=1.0)
	svd1 = svd(p.value);
	U = svd1[1];
	S = copy(svd1[2]);
	Ssqrt = sqrt.(diag(S));
	SsqrtInv = diagm(1./Ssqrt);
	pSqrt = U*diagm(Ssqrt)*U.';
  	T = U*SsqrtInv*U.'*(t.*ξ.value)*U*SsqrtInv*U.';
    svd2 = svd(T);
   	Se = diagm(exp.(svd2[2]))
  	Ue = svd2[1]
	return SPDPoint(pSqrt*Ue*Se*Ue.'*pSqrt)
end
function log(M::SymmetricPositiveDefinite,p::SPDPoint,q::SPDPoint)
	svd1 = svd(p.value)
	U = svd1[1]
	S = svd1[2]
	Ssqrt = sqrt.(diag(S))
	SsqrtInv = diagm(1./Ssqrt)
	Ssqrt = diagm(Ssqrt)
  	pSqrt = U*Ssqrt*U.'
	T = U*SsqrtInv*U.'*q.value*U*SsqrtInv*U.'
	svd2 = svd(T)
	Se = diagm(log.(svd2[2]))
	Ue = svd2[1]
	ξ = pSqrt*Ue*Se*Ue.'*pSqrt
	return SPDTVector(ξ)
end
manifoldDimension(M::SymmetricPositiveDefinite) = M.dimension
manifoldDimension(x::SPDPoint) = size( getValue(x), 1)*(size( getValue(x), 1)+1)/2
norm(M::SymmetricPositiveDefinite,x::SPDPoint,ξ::SPDTVector) = sqrt(dot(M,x,ξ,ξ) )
function parallelTransport(M::SymmetricPositiveDefinite,x::SPDPoint,y::SPDPoint,ξ::SPDTVector)
	svd1 = svd( getValue(x) )
	U = svd1[1]
	S = svd1[2]
	Ssqrt = sqrt.(diag(S))
	SsqrtInv = diagm(1./Ssqrt)
	Ssqrt = diagm(Ssqrt)
	xSqrt = U*Ssqrt*U.'
  	xSqrtInv = U*SsqrtInv*U.'
	tξ = xSqrtInv * getValue(ξ) * xSqrtInv
	tY = xSqrtInv * getValue(q) * xSqrtInv
	svd2 = svd(tY)
	Se = diagm(log.(svd2[2]))
	Ue = svd2[1]
	tY2 = Ue*Se*Ue.'
	eig1 = eig(0.5*tY2)
	Sf = diagm(exp.(eig1[1]))
	Uf = eig1[2]
	return SPDTVector(xSqrt*Uf*Sf*Uf.'*(0.5*(tξ+tξ.'))*Uf*Sf*Uf.'*xSqrt)
end
# Display
# ---
show(io::IO, M::SymmetricPositiveDefinite) = print(io, "The Manifold $(M.name).")
show(io::IO, p::SPDPoint) = print(io, "SPD($(p.value))")
show(io::IO, ξ::SPDTVector) = print(io, "SPDT($(ξ.value))")
