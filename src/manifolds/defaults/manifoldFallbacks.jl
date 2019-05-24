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
  throw( ErrorException("distance not implemented for a $sig1 and a $sig2 on $sig3." ) )
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
  throw( ErrorException("dot not implemented for a $sig2 and $sig3 in the tangent space of a $sig1 on $sig4." ) )
end
@doc doc"""
    exp(M,x,ξ,[t=1.0])
computes the exponential map at an [`MPoint`](@ref) `x` for the
[`TVector`](@ref) `ξ` on the [`Manifold`](@ref) `M`. The optional parameter `t` can be
used to shorten `ξ` to `tξ`.
"""
function exp(M::mT, x::P, ξ::T,t::N=1.0) where {mT<:Manifold, P<:MPoint, T<:TVector, N<:Number}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(M) )
  throw( ErrorException("exp not implemented for a $sig1 and a $sig2 on $sig3." ) )
end
@doc doc"""
    getValue(x)
get the actual value representing the point `x` on a manifold.
This should be implemented if you do not use the field x.value to avoid the
try-catch in the fallback implementation.
"""
function getValue(x::P) where {P <: MPoint}
    try
        return x.value
    catch
        sig1 = string( typeof(x) )
        throw( ErrorException("getValue not implemented for a $sig1.") );
    end
end
@doc doc"""
    getValue(ξ)
get the actual value representing the tangent vector `ξ` to a manifold.
This should be implemented if you do not use the field ξ.value to avoid the
try-catch in the fallback implementation.
"""
function getValue(ξ::T) where {T <: TVector}
    try
        return ξ.value
    catch
        sig1 = string( typeof(ξ) )
        throw( ErrorException("getValue – not implemented for tangent vector $sig1.") );
    end
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
  throw( ErrorException("log – not Implemented for Points $sig1 and $sig2 on the manifold $sig3.") )
end
@doc doc"""
    manifoldDimension(x)
returns the dimension of the manifold `M` the point `x` belongs to.
"""
function manifoldDimension(x::P)::Integer where {P<:MPoint}
  sig1 = string( typeof(x) )
  throw( ErrorException("manifoldDimension not Implemented for a $sig1." ) )
end
@doc doc"""
    manifoldDimension(M)
returns the dimension of the manifold `M`.
"""
function manifoldDimension(M::mT)::Integer where {mT<:Manifold}
  sig1 = string( typeof(M) )
  throw( ErrorException("manifoldDimension not Implemented on $sig1." ) )
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
  throw( ErrorException("parallelTransport not implemented for a $sig1, a $sig2, and a $sig3 on $sig4." ) )
end
@doc doc"""
    (Ξ,κ) = tangentONB(M,x,ξ)
compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi$ is the
first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$,
$\dot g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

*See also:* [`jacobiField`](@ref), [`adjointJacobiField`](@ref).
"""
function tangentONB(M::mT, x::P, ξ::T) where {mT <: Manifold, P <: MPoint, T <: TVector}
    sig1 = string( typeof(x) )
    sig2 = string( typeof(ξ) )
    sig3 = string( typeof(M) )
    throw( ErrorException("tangentONB not implemented for a $sig1 and a $sig2 on $sig3." ) )
end
@doc doc"""
    (Ξ,κ) = tangentONB(M,x,y)

compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi=\log_xy$ is the
first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$,
$\dot g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

*See also:* [`jacobiField`](@ref), [`adjointJacobiField`](@ref).
"""
tangentONB(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPoint, Q <: MPoint} = tangentONB(M,x,log(M,x,y))
"""
    typeofMPoint(ξ)
returns the MPoint belonging to the `TVector` subtype of `ξ`.
"""
typeofMPoint(ξ::T) where {T <: TVector} = typeofMPoint(typeof(ξ))
function typeofMPoint(ξT::Type{T}) where { T <: TVector }
    sig = string( ξT )
    throw( ErrorException("typeofMPoint not yet implemented for the tangent vector type $sig.") )
end
"""
    typeofTVector(p)
returns the TVector belonging to the `MPoint` subtype of `p`.
"""
typeofTVector(p::P) where {P <: MPoint} = typeofTVector(typeof(p))
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
  throw( ErrorException("zeroTVector(M) not implemented on $sig2." ) )
end
@doc doc"""
    validateMPoint(M,x)

check, whether the data in the [`MPoint`](@ref) `x` is a valid point on the
[`Manifold`](@ref) `M`. This is used to validate parameters and results during
computations using [`MPointE`](@ref)s.
Note that the default fallback is just a warning that no validation is available.

Throws an error message if `x` is not point on the manifold `M`, otherwise returns true
"""
function validateMPoint(M::mT, x::P) where {mT <: Manifold, P <: MPoint}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(M) )
  @warn """No valitadion for a $sig1 on $sig2 available. Continuing without
  validation. To turn this warning off, either deactivate the validate flag
  in (one of) your extended `MPoint`s or implement a corresponding validation."""
  return true
end
@doc doc"""
    validateTVector(M,x,ξ)

check, whether the data in the [`TVector`](@ref) `ξ` is a valid tangent vector
#to the [`MPoint`](@ref) `x` on the [`Manifold`](@ref) `M`.
This is used to validate parameters and results during computations using [`MPointE`](@ref)s.
Note that the default fallback is just a warning that no validation is available.

Throws an error message if `ξ` is not a tangent vector of `x`, otherwise returns true
"""
function validateTVector(M::mT, x::P, ξ::T) where {mT <: Manifold, P <: MPoint, T <: TVector}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(M) )
  @warn """No valitadion for a $sig1 and a $sig2 on $sig3 available.
  Continuing without validation. To turn this warning off, either deactivate
  the validate flag in (one of) your extended `TVectors`s or `MPoints`s or
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
    throw( ErrorException("zeroTVector(M,x) not implemented for a $sig1 on $sig2." ) )
end
