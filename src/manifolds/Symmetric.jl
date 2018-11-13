#
# Symetric.jl – The manifold of symmetric matrices
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import LinearAlgebra: norm, dot
import Base: exp, log, show

export SymmetricMatrices, SymPoint, SymTVector, show
# also indicates which functions are available (already) for Sym
export distance, dot, exp, norm, dot, manifoldDimension, parallelTransport
export zeroTVector
# Types
# ---
@doc doc"""
    Symmetric <: Manifold
The manifold $\mathcal M = \mathcal S(n)$, where $\mathcal S(n) = \{
x \in \mathbb R^{n\times n} | x = x^\tT
\}$, $n\in\mathbb N$ denotes the manifold of symmetric matrices
equipped with the trace inner product and its induced Forbenius norm.

Abbreviation: `Sym` or `Sym(n)`, respectively.
"""
struct Symmetric <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Symmetric(dimension::Int) = new("$dimension-by-$dimension symmetric matrices",(dimension*(dimension+1)/2),"Sym($dimension)")
end
@doc doc"""
    SymPoint <: MPoint
A point $x$ on the manifold $\mathcal M = \mathcal S(n)$ of $n\times n$
symmetric matrices, represented in the redundant way of a
symmetric matrix (instead of storing just the upper half).
"""
struct SymPoint <: MPoint
	value::Matrix{Float64}
	SymPoint(v::Matrix{Float64}) = new(v);
end
getValue(x::SymPoint) = x.value
@doc doc"""
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
# (a) P(n) is a matrix manifold
@traitimpl IsMatrixM{Symmetric}
@traitimpl IsMatrixP{SymPoint}
@traitimpl IsMatrixV{SymTVector}
# Functions
# ---
@doc doc"""
    distance(M,x,y)
distance of two symmetric manifolds inherited from embedding them in
$\mathbb R^{n\times n}$, i.e. use the Frobenious norm
"""
distance(M::Symmetric,x::SymPoint,y::SymPoint) = vecnorm( getValue(x) - getValue(y) )
@doc doc"""
   dot(M,x,ξ,ν)
inner product of two tangent vectors `ξ,ν::SymTVector` lying in the tangent
space of `x::SymPoint` of the manifold `M::Symmetric`.

"""
dot(M::Symmetric, x::SymPoint, ξ::SymTVector, ν::SymTVector) = dot( getValue(ξ), getValue(ν) )
@doc doc"""
    exp(M,x,ξ,[t=1.0])
computes the exponential map on the manifold of symmetric matrices `M::Symmetric`,
which is given by $\exp_{x}ξ = x+ξ$, where the additional parameter `t` can be
used to scale the tangent vector to $t\xi$.
"""
exp(M::Symmetric, x::SymPoint, ξ::SymTVector, t::Float64=1.0) = SymPoint( getValue(x) + t*getValue(ξ) )
@doc doc"""
   log(M,x,y)
computes the logarithmic map on the manifold of symmetric matrices `M::Symmetric`,
which is given by $\log_xy = y-x$.
"""
log(M::Symmetric,x::SymPoint,y::SymPoint) = SymTVector( getValue(y) - getValue(x) )
"""
    manifoldDimension(M)
returns the manifold dimension of the manifold of symmetric matrices `M`.
"""
manifoldDimension(M::Symmetric) = M.dimension
"""
    manifoldDimension(x)
returns the manifold dimension the symmetric matrix `x` belongs to.
"""
manifoldDimension(x::SymPoint) = size( getValue(x), 1)*(size( getValue(x), 1)+1)/2
"""
    norm(M,x,ξ)
computes the norm of the tangent vector `ξ` from the tangent space at `x`
given on the manifold of symmetric matrices `M` embedded in the Euclidean space,
i.e. by its Frobenius norm.
"""
norm(M::Symmetric,x::SymPoint,ξ::SymTVector) = norm( getValue(ξ) )
"""
    parallelTransport(M,x,y,ξ)
coputes the parallel transport of a tangent vector `ξ` from the tangent space at
`x` to the tangent space at `y` on the manifold `M` of symmetric matrices.
Since the metric is inherited from the embedding space, it is just the identity.
"""
parallelTransport(M::Symmetric,x::SymPoint,y::SymPoint,ξ::SymTVector) = ξ
@doc doc"""
    typicalDistance(M)
returns the typical distance on the [`Symmetric`](@ref)` Sym`: $n$.
"""
typicalDistance(M::Symmetric) = sqrt(2*manifoldDimension(M)-1/4)-1/2 #get back to the n of the R^n by n matrix
@doc doc"""
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
