#
#
# General documentation of exp/log/... and its fallbacks in case of non-implemented tuples
#
#
import LinearAlgebra: norm
export addNoise, distance, dot, exp, getValue, log, manifoldDimension, norm
export manifoldDimension, parallelTransport, randomMPoint, randomTVector, tangentONB
export typeofMPoint, typeofTVector, typicalDistance, zeroTVector
export validateMPoint, validateTVector
@doc doc"""
    distance(M,x,y)
computes the gedoesic distance between two [`MPoint`](@ref)s `x` and `y` on
a [`Manifold`](@ref) `M`.
"""
function distance(M::mT, x::T, y::T) where {mT <: Manifold, T <: MPoint}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(y) )
  sig3 = string( typeof(M) )
  throw( DomainError("distance not defined/implemented for a $sig1 and a $sig2 on $sig3." ) )
end
@doc doc"""
    dot(M, x, ξ, ν)
Computes the inner product of two [`TVector`](@ref)s `ξ` and `ν` from the
tangent space at the [`MPoint`](@ref) `x` on the [`Manifold`](@ref) `M`.
"""
function dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPoint, T <: TVector, S <: TVector}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(ν) )
  sig4 = string( typeof(M) )
  throw( DomainError("dot not defined/implemented for a $sig2 and $sig3 in the tangent space of a $sig1 on $sig4." ) )
end
# fallback for forgetting base
function dot(M::mT, ξ::T, ν::S) where {mT <: Manifold, P <: MPoint, T <: TVector, S <: TVector}
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(ν) )
  sig4 = string( typeof(M) )
  throw( ErrorException("dot requires a base point, but the function call only contains a manifold ($sig4) and two tangent vectors ($sig2) and ($sig3)." ) )
end
"""
    exp(M,x,ξ,[t=1.0])
computes the exponential map at an [`MPoint`](@ref) `x` for the
[`TVector`](@ref) `ξ` on the [`Manifold`](@ref) `M`. The optional parameter `t` can be
used to shorten `ξ` to `tξ`.
"""
function exp(M::mT, x::P, ξ::T,t::Float64=1.0) where {mT<:Manifold, P<:MPoint, T<:TVector, N<:Number}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(M) )
  throw( DomainError("exp not defined/implemented for a $sig1 and a $sig2 on $sig3." ) )
end
@doc doc"""
    log(M,x,y)
computes the [`TVector`](@ref) in the tangent space $T_x\mathcal M$ at the
[`MPoint`](@ref) `x` such that the corresponding geodesic reaches the
[`MPoint`](@ref) `y` after time 1 on the [`Manifold`](@ref) `M`.
"""
function log(M::mT,x::P,y::Q)::TVector where {mT<:Manifold, P<:MPoint, Q<:MPoint}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(y) )
  sig3 = string( typeof(M) )
  throw( DomainError("log – not defined/implemented for a $sig1 and a $sig2 on $sig3.") )
end

@doc doc"""
    parallelTransport(M,x,y,ξ)
Parallel transport of a vector `ξ` given at the tangent space $T_x\mathcal M$
of `x` to the tangent space $T_y\mathcal M$ at `y` along the geodesic form `x` to `y`.
If the geodesic is not unique, this function takes the same choice as `geodesic`.
"""
function parallelTransport(M::mT, x::P, y::Q, ξ::T) where {mT<:Manifold, P<:MPoint, Q<:MPoint, T<:TVector}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(y) )
  sig3 = string( typeof(ξ) )
  sig4 = string( typeof(M) )
  throw( DomainError("parallelTransport not defined/implemented for a $sig1, a $sig2, and a $sig3 on $sig4." ) )
end

@doc doc"""
    (Ξ,κ) = tangentONB(M,x,ξ)
    
compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi$ is the
first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$,
$\dot g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

# See also
 [`jacobiField`](@ref), [`adjointJacobiField`](@ref).
"""
function tangentONB(M::mT, x::P, ξ::T) where {mT <: Manifold, P <: MPoint, T <: TVector}
    sig1 = string( typeof(x) )
    sig2 = string( typeof(ξ) )
    sig3 = string( typeof(M) )
    throw( DomainError("tangentONB not defined/implemented for a $sig1 and a $sig2 on $sig3." ) )
end

@doc doc"""
    (Ξ,κ) = tangentONB(M,x,y)

compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi=\log_xy$ is the
first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$,
$\dot g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

# See also
 [`jacobiField`](@ref), [`adjointJacobiField`](@ref).
"""
tangentONB(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPoint, Q <: MPoint} = tangentONB(M,x,log(M,x,y))
"""
    typeofMPoint(T)

return the [`MPoint`](@ref) belonging to the [`TVector`](@ref) type `T`.
"""
function typeofMPoint(ξT::Type{T}) where { T <: TVector }
    sig = string( ξT )
    throw( ErrorException("typeofMPoint not yet implemented for the tangent vector type $sig.") )
end
"""
    typeofTVector(P)

returns the [`TVector`](@ref) belonging to the [`MPoint`](@ref) type `P`.
"""
function typeofTVector(pP::Type{P}) where {P <: MPoint}
    sig = string( pP )
    throw( ErrorException("typeofTVector not yet implemented for the tangent vector type $sig.") )
end
"""
    typicalDistance(M)

returns the typical distance on the [`Manifold`](@ref) `M`, which is for example
the longest distance in a unit cell or injectivity radius.
"""
function typicalDistance(M::mT) where {mT <: Manifold}
  sig2 = string( typeof(M) )
  throw( ErrorException("typicalDistance(M) not implemented on $sig2." ) )
end
@doc doc"""
    validateMPoint(M,x)

check, whether the data in the [`MPoint`](@ref) `x` is a valid point on the
[`Manifold`](@ref) `M`. This is used to validate parameters and results during
computations using [`MPointE`](@ref)s.
Note that the default fallback is just a warning that no validation is available.

The function should throw an error if `x` is not point on the manifold `M`,
otherwise it should return `true`.
"""
function validateMPoint(M::mT, x::P) where {mT <: Manifold, P <: MPoint}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(M) )
  @warn """No valitadion for a $sig1 on $sig2 available. Continuing without
  validation. To turn this warning off, either deactivate the validate flag
  in (one of) your extended MPoints or implement a corresponding validation."""
  return true
end
@doc doc"""
    validateTVector(M,x,ξ)

check, whether the data in the [`TVector`](@ref) `ξ` is a valid tangent vector
#to the [`MPoint`](@ref) `x` on the [`Manifold`](@ref) `M`.
This is used to validate parameters and results during computations when using
[`MPointE`](@ref)s. 
Note that the default fallback is just a warning that no validation is available.

Available validations should throw an error if `x` is not on `M` or `ξ` is not
in the tangent space of `x`. If `ξ` is valid, the function returns true.
"""
function validateTVector(M::mT, x::P, ξ::T) where {mT <: Manifold, P <: MPoint, T <: TVector}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(M) )
  @warn """No valitadion for a $sig1 and a $sig2 on $sig3 available.
  Continuing without validation. To turn this warning off, either deactivate
  the validate flag in (one of) your extended TVectors or MPoints or
  implement a corresponding validation"""
  return true
end
@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`MPoint`](@ref) $x\in\mathcal M$ on the [`Manifold`](@ref) `M`.
"""
function zeroTVector(M::mT, x::P) where {mT <: Manifold, P <: MPoint}
    sig1 = string( typeof(x) )
    sig2 = string( typeof(M) )
    throw( DomainError("zeroTVector not defined/implemented for a $sig1 on $sig2." ) )
end
