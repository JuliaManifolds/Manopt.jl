#
# Symetric.jl – The manifold of symmetric matrices
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import LinearAlgebra: norm, dot
import Base: exp, log, show

export Symmetric, SymPoint, SymTVector, show
# also indicates which functions are available (already) for Sym
export distance, dot, exp, norm, dot, manifoldDimension, parallelTransport
export validateMPoint, validateTVector
export zeroTVector, typeofMPoint, typeofTVector
# Types
# ---
@doc doc"""
    Symmetric <: Manifold

The manifold $\mathcal M = \mathrm{Sym}(n)$, where $\mathrm{Sym}(n) = \{
x \in \mathbb R^{n\times n} | x = x^\mathrm{T}
\}$, $n\in\mathbb N$, denotes the manifold of symmetric matrices
equipped with the trace inner product and its induced Forbenius norm.

# Abbreviation
`Sym` or `Sym(n)`, respectively.

# Constructor
    Symmetric(n)

generates the manifold of `n`-by-`n` symmetric matrices.
"""
struct Symmetric <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Symmetric(dimension::Int) = new("$dimension-by-$dimension symmetric matrices",(dimension*(dimension+1)/2),"Sym($dimension)")
end
@doc doc"""
    SymPoint <: MPoint

A point $x$ on the manifold $\mathcal M = \mathrm{Sym}(n)$ of $n\times n$
symmetric matrices, represented in the redundant way of a
symmetric matrix (instead of storing just the upper half).
"""
struct SymPoint{T <: AbstractFloat} <: MPoint
	value::Matrix{T}
	SymPoint{T}(v::Matrix{T}) where {T <: AbstractFloat} = new(v);
end
SymPoint(v::Matrix{T}) where {T <: AbstractFloat} = SymPoint{T}(v)
getValue(x::SymPoint) = x.value
@doc doc"""
    SymTVector <: TVector

A tangent vector $\xi$ in $T_x\mathcal M$ of a symmetric matrix $x\in\mathcal M$.
"""
struct SymTVector{T <: AbstractFloat} <: TVector
	value::Matrix{T}
  	SymTVector{T}(value::Matrix{T}) where {T <: AbstractFloat} = new(value);
end
SymTVector(v::Matrix{T}) where {T <: AbstractFloat} = SymTVector{T}(v)
getValue(ξ::SymTVector) = ξ.value
# Traits
# ---
# (a) P(n) is a matrix manifold
@traitimpl IsMatrixM{Symmetric}
@traitimpl IsMatrixP{SymPoint}
@traitimpl IsMatrixTV{SymTVector}
# Functions
# ---
@doc doc"""
    distance(M,x,y)

distance of two [`SymPoint`](@ref)s `x,y` on the [`Symmetric`](@ref) manifold `M``
inherited
from embedding them in $\mathbb R^{n\times n}$, i.e. use the Frobenious norm
of the difference.
"""
distance(M::Symmetric,x::SymPoint,y::SymPoint) = norm( getValue(x) - getValue(y) )
@doc doc"""
    dot(M,x,ξ,ν)

inner product of two [`SymTVector`](@ref)s `ξ,ν` lying in the tangent
space of the [`SymPoint`](@ref) `x` on the [`Symmetric`](@ref) manifold `M`.
"""
dot(M::Symmetric, x::SymPoint, ξ::SymTVector, ν::SymTVector) = dot( getValue(ξ), getValue(ν) )
@doc doc"""
    exp(M,x,ξ[, t=1.0])

compute the exponential map on the [`Symmetric`](@ref) manifold `M` given a
[`SymPoint`](@ref) `x` and a [`SymTVector`](@ref) `ξ`, as well as an optional
scaling factor `t`. The exponential map is given by

$\exp_{x}ξ = x+ξ.$
"""
exp(M::Symmetric, x::SymPoint, ξ::SymTVector, t::Float64=1.0) = SymPoint( getValue(x) + t*getValue(ξ) )
@doc doc"""
    log(M,x,y)

compute the logarithmic map for two [`SymPoint`](@ref)` x,y` on the [`Symmetric`](@ref) `M`,
which is given by $\log_xy = y-x$.
"""
log(M::Symmetric,x::SymPoint,y::SymPoint) = SymTVector( getValue(y) - getValue(x) )
"""
    manifoldDimension(M)

returns the manifold dimension of the [`Symmetric`](@ref) manifold `M`.
"""
manifoldDimension(M::Symmetric) = M.dimension
"""
    manifoldDimension(x)

returns the manifold dimension the [`SymPoint`](@ref) `x` belongs to.
"""
manifoldDimension(x::SymPoint) = size( getValue(x), 1)*(size( getValue(x), 1)+1)/2
"""
    norm(M,x,ξ)

computes the norm of the [`SymTVector`](@ref) `ξ` in the tangent space of
the [`SymPoint`](@ref) `x` on the [`Symmetric`](@ref) `M` embedded in the
Euclidean space, i.e. by its Frobenius norm.
"""
norm(M::Symmetric,x::SymPoint,ξ::SymTVector) = norm( getValue(ξ) )
"""
    parallelTransport(M,x,y,ξ)

compute the parallel transport of a [`SymTVector`](@ref) `ξ` from the tangent
space at the [`SymPoint`](@ref) `x` to the [`SymPoint`](@ref)` y` on the
[`Symmetric`](@ref) manifold `M`.
Since the metric is inherited from the embedding space, it is just the identity.
"""
parallelTransport(M::Symmetric,x::SymPoint,y::SymPoint,ξ::SymTVector) = ξ

typeofTVector(::Type{SymPoint{T}}) where T = SymTVector{T}
typeofMPoint(::Type{SymTVector{T}}) where T = SymPoint{T} 

@doc doc"""
    typicalDistance(M)

returns the typical distance on the [`Symmetric`](@ref) manifold `M`,
i.e. $\sqrt{n}$.
"""
typicalDistance(M::Symmetric) = sqrt( - 0.5 + sqrt(1/4 + 2*manifoldDimension(M) ) ) # manDim to n

@doc doc"""
    validateMPoint(M,x)

validate, that the [`SymPoint`](@ref) `x` is a valid point on the
[`Symmetric`](@ref) manifold `M`,
i.e. that its dimensions are correct and that the matrix is symmetric.
"""
function validateMPoint(M::Symmetric, x::SymPoint)
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
    return true
end
@doc doc"""
    validateTVector(M,x,ξ)

validate, that the [`SymTVector`](@ref) is a valid tangent vector to the
[`SymPoint`](@ref) `x` on the [`Symmetric`](@ref) manifold `M`,
i.e. that its dimensions are correct and that the matrix is symmetric.
"""
function validateTVector(M::Symmetric, x::SymPoint, ξ::SymTVector)
    ξs = size( getValue(ξ), 1)*(size( getValue(ξ), 1)+1)/2
    if (manifoldDimension(M) ≠ manifoldDimension(x)) || (ξs ≠ manifoldDimension(x))
        throw(DomainError(
            "The tangent vector $ξ of size $(ξs), the point $x ($(manifoldDimension(x))) and the manifold $M ($(manifoldDimension(M))) are not all equal in dimensions, so the tangent vector can not be correct."
        ))
    end
    if norm(getValue(ξ) - transpose(getValue(ξ))) > 10^(-14)
        throw(DomainError(
            "The tangent vector $ξ is not a symmetric matrix and hence can not lie in the tangent space of $x on $M."
        ))
    end
    return true
end
@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`SymPoint`](@ref) `x` on  the [`Symmetric`](@ref) manifold `M`.
"""
zeroTVector(M::Symmetric, x::SymPoint) = SymTVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, M::Symmetric) = print(io, "The Manifold of $(M.name).")
show(io::IO, p::SymPoint) = print(io, "Sym($(p.value))")
show(io::IO, ξ::SymTVector) = print(io, "SymT($(ξ.value))")
