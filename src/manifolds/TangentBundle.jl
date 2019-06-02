#
#      TB - the manifold of the tangent bundle
#  Point is a Point on the n-dimensional sphere.
#
import LinearAlgebra: norm, dot
import Base: exp, log, show
export TangentBundle, TBPoint, TBTVector, show, getValue, getTangent, getBase
export distance, dot, exp, log, manifoldDimension, norm
export randomMPoint, parallelTransport, zeroTVector, typeofMPoint, typeofTVector
export validateMPoint, validateTVector
#
# Type definitions
#

@doc doc"""
    TangentBundle <: Manifold

The manifold $\mathcal M = T\mathcal N$ obtained by looking at the tangent
bundle of a [`Manifold`](@ref)s tangent spaces. The manifold obtained is of
dimension $2d$, where $d$ is the dimension of the manifold $\mathcal N$
considered.

To keep notations clear, small letters will always refer to points (`x,y`) or
tangent vectors (`ξ,η`) on the manifold $\mathcal N$, while capital letters
(`X, Y, Z` and `Ξ,Η`) will refer to points and tangent vectors in the tangent
bundle respectively.

# Abbreviation
`TB`

# Constructor
    TangentBundle(M)

generates the tangent bundle to the [`Manifold`](@ref) `M`.
"""
struct TangentBundle{M <: Manifold} <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  manifold::M where {M <: Manifold}
  TangentBundle{mT}(M::mT) where {mT <: Manifold} = new{mT}("Tangent bundle of $(M.name)",
    2*manifoldDimension(M),"TB($(M.abbreviation))",M)
end
TangentBundle(M::mT) where {mT <: Manifold} = TangentBundle{mT}(M)
@doc doc"""
    getBase(M)

return the base manifold of the [`TangentBundle`](@ref) [`Manifold`](@ref) `M`.
"""
getBase(M::TangentBundle) = M.manifold

@doc doc"""
    TBPoint <: MPoint

A point $N\in \mathcal M$ on the manifold $\mathcal M = T\mathcal N$
represented by a tuple `(x,ξ)`, where $x\in\mathcal N$ is a point on the manifold
and $\xi=\xi_x\in T_x\mathcal N$ is a point in the tangent space at $x$.

Two constructors are available:
* `TBPoint(x,ξ)` to construct a tangent bundle point by specifying both
  an [`MPoint`](@ref) `x` and a [`TVector`](@ref) `ξ`.
* `TBPoint( (X) )` to construct a tangent bundle point from a tuple `X=(x,ξ)`,
i.e. the value  of another tangent bundle point.
"""
struct TBPoint{P<:MPoint, T <: TVector} <: MPoint
  value::Tuple{P,T}
  TBPoint{P,T}(x::P, ξ::T) where {P <: MPoint, T <: TVector} = new( (x,ξ) )
  TBPoint{P,T}(X::Tuple{P, T}) where {P <: MPoint, T <: TVector} = new( X )
end
TBPoint(x::P, ξ::T) where {P <: MPoint, T <: TVector} = TBPoint{P,T}(x,ξ)
TBPoint(X::Tuple{P,T}) where {P <: MPoint, T <: TVector}= TBPoint{P,T}( X )
@doc doc"""
    getValue(X)
return the value of the [`TBPoint`](@ref) `X`, i.e. the Tuple of a
[`MPoint`](@ref) and its [`TVector`](@ref).
"""
getValue(X::TBPoint{P,T}) where { P <: MPoint, T <: TVector } = X.value
@doc doc"""
    getBase(X)
return the base of the [`TBPoint`](@ref)` X`, i.e. its [`MPoint`](@ref).
"""
getBase(X::TBPoint{P,T}) where { P <: MPoint, T <: TVector } = X.value[1]
# passthrough for Extendeds
getBase(X::MPointE{TBPoint{P,T}}) where { P <: MPoint, T <: TVector } = MPointE( getBase(strip(X)) )

@doc doc"""
    getTangent(X)
return the tangent of the [`TBPoint`](@ref)` X`, i.e. the its [`TVector`](@ref).
"""
getTangent(X::TBPoint{P,T}) where { P <: MPoint, T <: TVector } = X.value[2]
#passthrough for Extendeds
getTangent(X::MPointE{TBPoint{P,T}}) where { P <: MPoint, T <: TVector } = TVectorE( getTangent(strip(X)), getBase(strip(X) ) )

@doc doc"""
    TBTVector <: TVector

A tangent vector $\Xi \in T_X\mathcal M$ on the manifold
$\mathcal M = T\mathcal N$ for the (base) manifold $\mathcal N$.
Both tangent components can be represented by elements from the base point $x$
from within $X=(x,\xi)$. Both components are from the same space since
$TT_x\mathcal N= T_x\mathcal N$, hence the tangent vector is a tuple
$(\xi\,\nu)\in T_x\mathcal N\times T_x\mathcal N$.
As for the [`TBPoint`](@ref) two constructors are available, one for stwo seperate
tangent vectors, one for a tuple of two tangent vectors.
"""
struct TBTVector{T <: TVector} <: TVector
  value::Tuple{T,T}
  TBTVector{T}(ξ::T,ν::T) where {T <: TVector} = new( (ξ,ν) )
  TBTVector{T}(Ξ::Tuple{T,T}) where {T <: TVector} = new( Ξ )
end
TBTVector(ξ::T, ν::T) where {T <: TVector} = TBTVector{T}( ξ,ν )
TBTVector(Ξ::Tuple{T,T}) where {T <:TVector}= TBTVector{T}( Ξ )
@doc doc"""
    getValue(Ξ)
return the Tuple contained in the [`TBTVector`](@ref)` Ξ`, i.e. its tuple of two
[`TVector`](@ref)s.
"""
getValue(Ξ::TBTVector) = Ξ.value;

@doc doc"""
    getBase(Ξ)
return the base of the [`TBTVector`](@ref)` Ξ`, i.e. its first [`TVector`](@ref).
"""
getBase(Ξ::TBTVector) = Ξ.value[1]
# passthrough for Extendeds
getBase(Ξ::TVectorE{TBTVector{T}}) where {T <: TVector} = TVectorE( getBase(strip(Ξ)), getBase(getBasePoint(Ξ)) )

@doc doc"""
    getTangent(Ξ)
return the tangent of the [`TBTVector`](@ref)` Ξ`, i.e. its second
[`TBTVector`](@ref).
"""
getTangent(Ξ::TBTVector) = Ξ.value[2]
getTangent(Ξ::TVectorE{TBTVector{T}}) where {T <: TVector}= TVectorE( getTangent(strip(Ξ)), getBase(getBasePoint(Ξ)) )

*(t::Number, Ξ::TBTVector) = TBTVector(t*getBase(Ξ),t*getTangent(Ξ))

+(Ξ::TBTVector, Η::TBTVector) = TBTVector(getBase(Ξ) + getBase(Η), getTangent(Ξ) + getTangent(Η) )
-(Ξ::TBTVector, Η::TBTVector) = TBTVector(getBase(Ξ) - getBase(Η), getTangent(Ξ) - getTangent(Η) )
-(Ξ::TBTVector) = TBTVector(- getBase(Ξ), -getTangent(Ξ) )
+(Ξ::TBTVector) = TBTVector( getBase(Ξ), getTangent(Ξ) )

# Functions
# ---
@doc doc"""
    distance(M,X,Y)

Compute the Riemannian distance on $\mathcal M=T\mathcal N$ by employing the
distance on the manifold for the base component and the vector norm on the
tangent space, and then take the Eucklidean Norm of the vector from
$\mathbb R^2$.
"""
distance(M::TangentBundle,X::TBPoint, Y::TBPoint) = sqrt(
  distance(M.manifold,getBase(X),getBase(Y))^2 + norm(M.manifold,getBase(X),getTangent(X)-getTangent(Y))^2
)

@doc doc"""
    dot(M,X,Ξ,Η)

Compute the Riemannian inner product for two [`TBTVector`](@ref)s `Ξ` and `Η`
from $T_X\mathcal M$ of the [`TangentBundle`](@ref)` M = TN` given by
the sum of the two inner products of the tangent vector components
"""
dot(M::TangentBundle, X::TBPoint, Ξ::TBTVector, Η::TBTVector) =
dot(M.manifold,getBase(X),getBase(Ξ),getBase(Η)) + dot(M.manifold,getBase(X),getTangent(Ξ),getTangent(Η))

@doc doc"""
    exp(M,X,Ξ[, t=1.0])

Compute the exponential map on the [`TangentBundle`](@ref) `M`$=T\mathcal N$ with
respect to the [`TBPoint`](@ref)` X=(x,ξ)` and the [`TBTVector`](@ref)` Ξ=(Ξx,Ξξ)`,
which consists of the exponential map in the first component (`exp(x,Ξx,t)` and
a (scaled) addition in the second (`ξ + tΞξ`) in the second component followed
by a parallel transport.
"""
function exp(M::TangentBundle,X::TBPoint,Ξ::TBTVector,t::Float64=1.0)
  x = exp(M.manifold,getBase(X),getBase(Ξ),t)
  ξ = parallelTransport(M.manifold, getBase(X), x,
    getTangent(X) + t*getTangent(Ξ)
  )
  return TBPoint(x,ξ)
end
@doc doc"""
    log(M,X,Y)

Compute the logarithmic map on the [`TangentBundle`](@ref)
$\mathcal M=T\mathcal N$, i.e. the `log` for the base manifold component and
a parallel transport and a minus for the tangent components.

"""
function log(M::TangentBundle,X::TBPoint,Y::TBPoint)
  Ξx = log(M.manifold, getBase(X), getBase(Y) )
  Ξξ = getTangent(X) - parallelTransport(M.manifold, getBase(X), getBase(Y), getTangent(Y))
  return TBTVector(Ξx,Ξξ)
end
@doc doc"""
    manifoldDimension(X)

returns the dimension of the [`TangentBundle`](@ref) `M`$=T\mathcal N$ to which `X`
bvelongs, which is twice the dimension of the base manifold.
"""
manifoldDimension(x::TBPoint)::Integer = 2*manifoldDimension(getBase(x))
@doc doc"""
    manifoldDimension(M)

returns the dimension of the [`TangentBundle`](@ref) `M`$=T\mathcal N$, i.e.,
twice the dimension of the base manifold `N`.
"""
manifoldDimension(M::TangentBundle)::Integer = M.dimension
@doc doc"""
    norm(M,X,Ξ)

Computes the norm of the [`TBTVector`](@ref)` Ξ` in the tangent space
$T_x\mathcal M$ at [`TBPoint`](@ref)` X` of the [`TangentBundle`](@ref) `M`.
"""
norm(M::TangentBundle, X::TBPoint, Ξ::TBTVector) = sqrt( dot(M,X,Ξ,Ξ) )
@doc doc"""
    parallelTransport(M,X,Y,Ξ)

Compute the paralllel transport of the [`TBTVector`](@ref)` Ξ` from
the tangent space $T_X\mathcal M$ at [`TBPoint`](@ref)` X` to
$T_Y\mathcal M$ at [`TBPoint`](@ref)` Y` on the [`TangentBundle`](@ref) `M`
provided that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
Then both components of $\Xi=(\Xi_x,\Xi_\xi)$ are parallely transported using the
parallel transport of the underlying base manifold.
"""
parallelTransport(M::TangentBundle, X::TBPoint, Y::TBPoint, Ξ::TBTVector) = TBTVector(
  parallelTransport(M.manifold, getBase(X), getBase(Y), getBase(Ξ) ),
  parallelTransport(M.manifold, getBase(X), getBase(Y), getTangent(Ξ) )
)
@doc doc"""
    randomMPoint(M)

returns a random point on the [`TangentBundle`](@ref) `M` by producing a
[`randomMPoint`](@ref)  random point on the base manifold and
[`randomTVector`](@ref) in the correspoinding tangent plane.
"""
function randomMPoint(M::TangentBundle)::TBPoint
  x = randomMPoint(M.manifold)
  ξ = randomTVector(M.manifold,x)
	return TBPoint(x,ξ)
end
@doc doc"""
    randomTVector(M,x)

returns a random tangent vector the [`TangentBundle`](@ref) `M` by producing
two [`randomTVector`](@ref)s in the correspoinding tangent plane of the
[`getBase`](@ref) of the [`TBPoint`](@ref) `x`.
"""
function randomTVector(M::TangentBundle, x::TBPoint)
  xξ = randomTVector(M.manifold,getBase(x))
  ξξ = randomTVector(M.manifold,getBase(x))
	return TBTVector(xξ,ξξ)
end
@doc doc"""
    tangentONB(M,X,Y)

constructs a tangent ONB in the tangent space of the [`TBPoint`](@ref)` X` on
the [`TangentBundle`](@ref) `M`, where $\log_XY$ is the first component.
"""
tangentONB(M::TangentBundle, X::TBPoint, Y::TBPoint) = tangentONB(M,X,log(M,X,Y))
@doc doc"""
    Η,κ = tangentONB(M,X,Ξ)

constructs a tangent ONB in $T_X\mathcal M$, i.e. in the tangent space of the
[`TBPoint`](@ref) `x` on the [`TangentBundle`](@ref) `M`
whose first vector is given by the [`TBTVector`](@ref)` Ξ`.
It is constructed by using twice the tangent ONB of the base manifold.
"""
function tangentONB(M::TangentBundle,X::TBPoint,Ξ::TBTVector)
  BaseONB,κx = tangentONB(M.manifold, getBase(X), getBase(Ξ))
  TangentONB,κξ = tangentONB(M.manifold, getBase(X), getTangent(Ξ) )
  baseTVs = [BaseONB..., [zeroTVector(M.manifold,getBase(X)) for i=1:length(TangentONB)]...]
  tangentTVs = [ [zeroTVector(M.manifold,getBase(X)) for i=1:length(BaseONB)]..., TangentONB...]
  return TBTVector.(baseTVs, tangentTVs), [κx...,κξ...]
end

typeofTVector(::Type{TBPoint{P,T}})  where {P <: MPoint, T <: TVector} = TBTVector{typeofTVector(P)}
typeofMPoint(::Type{TBTVector{T}}) where {T <: TVector} = TBPoint{typeofMPoint(T),T}

@doc doc"""
    typicalDistance(M)

returns the typical distance on the [`TangentBundle`](@ref) `M`, i.e. for
$\mathcal M = T\mathcal N$ we
obtain $t_{\mathcal M} = \sqrt{t_{\mathcal N}^2 + d_{\mathcal N}^2}$, where $d$
denotes the manifold dimension.
"""
typicalDistance(M::TangentBundle) = sqrt(
  typicalDistance(M.manifold)^2 + manifoldDimension(M.manifold)
)
@doc doc"""
    validateMPoint(M,X)

validate that the [`TBPoint`](@ref)` X` is a valid point on the
[`TangentBundle`](@ref) `M`, i.e. the first component is a point on the base
manifold and the second a tangent vector is the tangent space of the first 
"""
function validateMPoint(M::TangentBundle,X::TBPoint)
  return validateMPoint(M.manifold,getBase(X)) && validateTVector(M.manifold,getBase(X),getTangent(X))
  return true
end
@doc doc"""
    validateTVector(M,X,Ξ)

validate, that the [`TBTVector`](@ref)` Ξ` is a valid tangent vector in the
tangent space of the [`TBPoint`](@ref)` X` on the [`TangentBundle`](@ref) `M`,
i.e. both components of `Ξ` are tangent vectors in the tangent space of the base
component of `X`, since the tangent space of the tangent space is represented as
the tangent space itself.
"""
function validateTVector(M::TangentBundle, X::TBPoint, Ξ::TBTVector)
  # both components have to be tangent vectors, since TM = TTM
  return validateTVector(M.manifold, getBase(X), getTangent(Ξ)) && validateTVector(M.manifold, getBase(X), getTangent(Ξ))
end
@doc doc"""
    zeroTVector(M,X)

returns a zero vector in the tangent space $T_X\mathcal M$ of the
[`TangentBundle`](@ref) $X=(x,ξ)\in T\mathcal N$ by creating two zero vectors in $T_x\mathcal M$.
"""
zeroTVector(M::TangentBundle{Mt}, X::TBPoint{P}) where {Mt <: Manifold, P <: MPoint} = 
TBTVector{typeofTVector(P)}(
  zeroTVector(M.manifold,getBase(X)),
  zeroTVector(M.manifold,getBase(X))
)
# Display
# ---
show(io::IO, M::TangentBundle) = print(io,"The Tangent bundle of <$(M.manifold)>")
show(io::IO, X::TBPoint) = print(io, "TB($(getBase(X)), $(getTangent(X)))")
show(io::IO, Ξ::TBTVector) = print(io, "TBT($( getBase(Ξ) ), $( getTangent( Ξ )))")
