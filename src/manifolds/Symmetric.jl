#
# Symetric.jl – The manifold of symmetric matrices
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import Base.LinAlg: vecnorm, norm, dot
import Base: exp, log, show

export SymmetricMatrices, SymPoint, SymTVector, show
# also indicates which functions are available (already) for Sym
export distance, dot, exp, norm, dot, manifoldDimension, parallelTransport
# Types
# ---
doc"""
    Symmetric <: Manifold
The manifold $\mathcal M = \mathcal S(n)$, where $\mathcal S(n) = \{
x \in \mathbb R^{n\times n} | x = x^\tT
\}$, $n\in\mathbb N$ denotes the manifold of symmetric matrices
equipped with the trace inner product and its induced Forbenius norm.
Abbreviation: `Sym`
"""
struct Symmetric <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Symmetric(dimension::Int) = new("$dimension-by-$dimension symmetric matrices",(dimension*(dimension+1)/2),"Sym($dimension)")
end
doc"""
    SymPoint <: MPoint
A point $x$ on the manifold $\mathcal M = \mathcal S(n)$ of $n\times n$
symmetric positive definite matrices, represented in the redundant way of a
symmetric positive definite matrix.
"""
struct SymPoint <: MPoint
	value::Matrix{Float64}
	SymPoint(v::Matrix{Float64}) = new(v);
end
getValue(x::SymPoint) = x.value
doc"""
    SymTVector <: TVector
A tangent vector $\xi$ in $T_x\mathcal M$ of a symmetric matrix $x\in\mathcal M$.
"""
struct SymTVector <: TVector
	value::Matrix{Float64}
  	SymTVector(value::Matrix{Float64}) = new(value);
end
getValue(ξ::SymTVector) = ξ.value
# Traits
# ---
# (a) P(n) is a matrix manidolf
@traitimpl IsMatrixM{Symmetric}
@traitimpl IsMatrixP{SymPoint}
@traitimpl IsMatrixV{SymTVector}
# Functions
# ---
distance(M::Symmetric,x::SymPoint,y::SymPoint) = vecnorm( getValue(x) - getValue(y) )
function dot(M::Symmetric, x::SymPoint, ξ::SymTVector, ν::SymTVector) = vec( getValue(ξ) )'*vec( getValue(ν) )
function exp(M::Symmetric, x::SymPoint, ξ::SymTVector, t::Float64=1.0)
	svd1 = svd( getValue(x) );
	U = svd1[1];
	S = copy(svd1[2]);
	Ssqrt = sqrt.(S);
	SsqrtInv = diagm(1./Ssqrt);
	pSqrt = U*diagm(Ssqrt)*U.';
  	T = U*SsqrtInv*U.'*(t.*ξ.value)*U*SsqrtInv*U.';
    svd2 = svd(T);
   	Se = diagm(exp.(svd2[2]))
  	Ue = svd2[1]
	return SymPoint(pSqrt*Ue*Se*Ue.'*pSqrt)
end
function log(M::Symmetric,x::SymPoint,y::SymPoint)
	svd1 = svd( getValue(x) )
	U = svd1[1]
	S = svd1[2]
	Ssqrt = sqrt.(S)
	SsqrtInv = diagm(1./Ssqrt)
	Ssqrt = diagm(Ssqrt)
  	pSqrt = U*Ssqrt*U.'
	T = U * SsqrtInv * U.' * getValue(y) * U * SsqrtInv * U.'
	svd2 = svd(T)
	Se = diagm(log.(svd2[2]))
	Ue = svd2[1]
	ξ = pSqrt*Ue*Se*Ue.'*pSqrt
	return SymTVector(ξ)
end
manifoldDimension(M::Symmetric) = M.dimension
manifoldDimension(x::SymPoint) = size( getValue(x), 1)*(size( getValue(x), 1)+1)/2
norm(M::Symmetric,x::SymPoint,ξ::SymTVector) = vecnorm( getValue(ξ) )
function parallelTransport(M::Symmetric,x::SymPoint,y::SymPoint,ξ::SymTVector)
	svd1 = svd( getValue(x) )
	U = svd1[1]
	S = svd1[2]
	Ssqrt = sqrt.(S)
	SsqrtInv = diagm(1./Ssqrt)
	Ssqrt = diagm(Ssqrt)
	xSqrt = U*Ssqrt*U.'
  	xSqrtInv = U*SsqrtInv*U.'
	tξ = xSqrtInv * getValue(ξ) * xSqrtInv
	tY = xSqrtInv * getValue(y) * xSqrtInv
	svd2 = svd(tY)
	Se = diagm(log.(svd2[2]))
	Ue = svd2[1]
	tY2 = Ue*Se*Ue.'
	eig1 = eig(0.5*tY2)
	Sf = diagm(exp.(eig1[1]))
	Uf = eig1[2]
	return SymTVector(xSqrt*Uf*Sf*Uf.'*(0.5*(tξ+tξ.'))*Uf*Sf*Uf.'*xSqrt)
end
# Display
# ---
show(io::IO, M::Symmetric) = print(io, "The Manifold $(M.name).")
show(io::IO, p::SymPoint) = print(io, "Sym($(p.value))")
show(io::IO, ξ::SymTVector) = print(io, "SymT($(ξ.value))")
