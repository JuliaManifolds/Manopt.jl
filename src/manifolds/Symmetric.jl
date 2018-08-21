#
# Symetric.jl – The manifold of symmetric matrices
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import LinearAlgebra: vecnorm, norm, dot
import Base: exp, log, show

export SymmetricMatrices, SymPoint, SymTVector, show
# also indicates which functions are available (already) for Sym
export distance, dot, exp, norm, dot, manifoldDimension, parallelTransport
export zeroTVector
# Types
# ---
"""
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
"""
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
"""
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
function exp(M::Symmetric, x::SymPoint, ξ::SymTVector, t::Float64=1.0) = SymPoint( getValue(x) + t*getValue(ξ) )
function log(M::Symmetric,x::SymPoint,y::SymPoint) = SymTVector( getValue(y) - getValue(x) )
manifoldDimension(M::Symmetric) = M.dimension
manifoldDimension(x::SymPoint) = size( getValue(x), 1)*(size( getValue(x), 1)+1)/2
norm(M::Symmetric,x::SymPoint,ξ::SymTVector) = vecnorm( getValue(ξ) )
function parallelTransport(M::Symmetric,x::SymPoint,y::SymPoint,ξ::SymTVector) = ξ
end
"""
    typicalDistance(M)
returns the typical distance on the [`Symmetric`](@ref)` Sym`: $n$.
"""
typicalDistance(M::Symmetric) = manifoldDimension(M);
"""
    ξ = zeroTVector(M,x)
returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`SymPoint`](@ref) $x\in\mathcal S(n)$ on the [`Symmetric`](@ref)` Sym`.
"""
zeroTVector(M::Symmetric, x::SymPoint) = SymTVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, M::Symmetric) = print(io, "The Manifold $(M.name).")
show(io::IO, p::SymPoint) = print(io, "Sym($(p.value))")
show(io::IO, ξ::SymTVector) = print(io, "SymT($(ξ.value))")
