#
#      TB - the manifold of the tangent bundle
#  Point is a Point on the n-dimensional sphere.
#
import LinearAlgebra: norm, dot
import Base: exp, log, show
export TangentBundle, TBPoint, TBTVector, show, getValue
export addNoise, distance, dot, exp, log, manifoldDimension, norm
export randomPoint, parallelTransport, zeroTVector
#
# Type definitions
#

@doc doc"""
    TangentBundle <: Manifold
The manifold $\mathcal M = T\mathcal N$ obtained by looking at the tangent bundle
of a [`Manifold`](@ref)s tangent spaces. The manifold obtained is of dimension
$2d$, where $d$ is the dimension of the manifold $\mathcal N$ considered.
Its abbreviation is `TB`.  Its abbreviation String is `T` and the abbreviation
of the manifold $\mathcal N$.

To keep notations clear, small letters will always refer to points (`x,y`) or
tangent vectors (`ξ,η`) on the manifold $\mathcal N$, while capital letters
(`X, Y, Z` and `Ξ,Η`) will refer to points and tangent vectors in the tangent
bundle respectively.
"""
struct TangentBundle <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  manifold::M where {M <: Manifold}
  TangentBundle(M::mT) where {mT <: Manifold} = new("Tangent bundle of $(M.name)",2*manifoldDimension(M),"TB($(M.abbreviation))")
end
@doc doc"""
    TBPoint <: MPoint
A point $N\in \mathcal M$ on the manifold $\mathcal M = T\mathcal N$
represented by a tuple `(x,ξ)`, where $x\in\mathcal N$ is a point on the manifold
and $\xi=\xi_x\in T_x\mathcal N$ is a point in the tangent space at $x$.

Two constructors are available:
* `TBPoint(x,ξ)` to construct a tangent bundle point by specifying
both an [ `MPoint`](@ref)` x` and a [`TVector`](@ref)' ξ'.
* `TBPoint( (X) )` to construct a tangent bundle point from a tuple `X=(x,ξ)`,
i.e. the value  of another tangent bundle point.
"""
struct TBPoint <: MPoint
  value::Tuple{P,T} where {P <: MPoint, T <: TVector}
  TBPoint(x::Pt, ξ::Tt) where {Pt <: MPoint, Tt <: TVector} = new( (x,ξ) )
  TBPoint(X::Tuple{Pt,Tt} where {Pt <: MPoint, Tt <: TVector}) = new( X )
end
getValue(X::TBPoint) = X.value;
getBase(X::TBPoint) = X.value(1)
getTangent(X::TBPoint) = X.value(2)

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
struct TBTVector <: TVector
  value::Tuple{T,T} where { T <: TVector }
  STBVector(ξ::Tt,ν::Tt) where {Tt <: TVector} = new( (ξ,ν) )
  STBVector(Ξ::Tuple{Tt,Tt} where {Tt <: TVector}) = new( Ξ )
end
getValue(Ξ::TBTVector) = Ξ.value;
getBaseTangent(Ξ::TBTVector) = Ξ.value(1)
getTangentTangent(Ξ::TBTVector) = Ξ.value(2)

# Functions
# ---
@doc doc"""
    addNoise(M,X,σ)
add noise to both components of a [`TBPoint`]` X=(x,ξ)`, by employing the manifolds
`addNoise` for the base component `x` and the classical component wise `randn` for
the (vector space) tangent vector component `ξ`.
"""
addNoise(M::TangentBundle, X::TBPoint, σ::Real) = TBPoint( addNoise(M.manifold,getBase(X)), getTangent(X) + randn(size(getValue(getTangent))))
@doc doc"""
    distance(M,X,Y)
Compute the Riemannian distance on $\mathcal M=T\mathcal N$ by employing the
distance on the manifold for the base component and the vector norm on the tangent space,
and then take the Eucklidean Norm of the vector from $\mathbb R^2$.
"""
distance(M::TangentBundle,X::TBPoint, Y::TBPoint) = sqrt(
  distance(M.manifold,getBase(X),getBase(Y))^2 + norm(getTangent(X)-getTangent(Y))^2
)

@doc doc"""
    dot(M,X,Ξ,Η)
Compute the Riemannian inner product for two [`TBTVector`](@ref)s `Ξ` and `Η`
from $T_X\mathcal M$ of the [`TangentBundle`](@ref)` M = TN` given by
the sum of the two inner products of the tangent vector components
"""
dot(M::TangentBundle, X::SnPoint, Ξ::SnTVector, Η::SnTVector) =
dot(M.manifold,getBaseTangent(Ξ),getBaseTangent(Η)) + dot(M.manifold,getTangentTangent(Ξ),getTangentTangent(Η))

@doc doc"""
    exp(M,X,Ξ,[t=1.0])
Compute the exponential map on the [`TangentBundle`](@ref)` M`$=T\mathcal N$ with
respect to the [`STBPoint`](@ref)` X=(x,ξ)` and the [`TBTVector`](@ref)` Ξ=(Ξx,Ξξ)`,
which consists of the exponential map in the first component (`exp(x,Ξx,t)` and
a (scaled) addition in the second (`ξ + tΞξ`) in the second component followed
by a parallel transport.
"""
function exp(M::TangentBundle,X::TBTVector,Ξ::TBTVector,t::Float64=1.0)
  x = exp(M.manifold,getBase(X),getBaseTangent(Ξ),t)
  ξ = parallelTransport(M.manifold, getBase(X), x,
    getTangent(X) + t*getTangentTangent(Ξ)
  )
  return TBPoint(x,ξ)
end
@doc doc"""
    log(M,X,Y)
Compute the logarithmic map on the [`TangentBundle`](@ref)
$\mathcal M=T\matcal N$, i.e. the `log` for the base manifold component and
a parallel transport and a minus for the tangent components.

"""
function log(M::TangentBundle,X::TBPoint,Y::TBPoint)
  Ξx = log(M.manifold, getBase(X), getBase(Y) )
  Ξξ = getTangent(X) - parallelTransport(M.manifold, getBase(X), getBase(Y), getTangent(Y))
  return TBTVector(Ξx,Ξξ)
end
@doc doc"""
    manifoldDimension(X)
returns the dimension of the [`TangentBundle`](@ref)` M`$=T\mathcal N$ to which `X`
bvelongs, which is twice the dimension of the base manifold.
"""
manifoldDimension(x::TBPoint)::Integer = 2*manifoldDimension(getBase(X))
@doc doc"""
    manifoldDimension(M)
returns the dimension of the [`TangentBundle`](@ref)` M`$=T\mathccal N$, i.e.,
twice the dimension of the base manifold `N`.
"""
manifoldDimension(M::TangentBundle)::Integer = M.dimension
@doc doc"""
    norm(M,X,Ξ)
Computes the norm of the [`TBTVector`](@ref)` Ξ` in the tangent space
$T_x\mathcal M$ at [`TBPoint`](@ref)` X` of the [`TangentBundle`](@ref)` M`.
"""
norm(M::TangentBundle, X::TBPoint, Ξ::TBTVector) = sqrt( dot(M,X,Ξ,Ξ) )
@doc doc"""
    parallelTransport(M,X,Y,Ξ)
Compute the paralllel transport of the [`TBTVector`](@ref)` Ξ` from
the tangent space $T_X\mathcal M$ at [`TBPoint`](@ref)` X` to
$T_Y\mathcal M$ at [`TBPoint`](@ref)` Y` on the [`TangentBundle`](@ref)` M`
provided that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
Then both components of $\Xi=(\Xi_x,\Xi_\xi)$ are parallely transported using the
parallel transport of the underlying base manifold.
"""
parallelTransport(M::TangentBundle, X::TBPoint, Y::TBPoint, Ξ::TBTVector) = TBTVector(
  parallelTransport(M.manifold, getBase(X), getBase(Y), getBaseTangent(Ξ) ),
  parallelTransport(M.manifold, getBase(X), getBase(Y), getTangentTangent(Ξ) )
)
@doc doc"""
    randomPoint(M)
returns a random point on the tangent bundle by producing a random point on the
manifold and a random vector in its tangent plane
"""
function randomPoint(M::TangentBundle)::TBPoint
  x = randomPoint(M.manifold)
  ξ = randomTVector(M.manifold,x)
	return TBPoint(x,ξ)
end
@doc doc"""
   tangentONB(M,X,Y)
constructs a tangent ONB on the [`TangentBundle`](@ref) where $\log_XY$ is the
first component.
"""
tangentONB(M::TangentBundle, X::TBPoint, Y::TBPoint) = tangentONB(M,X,log(M,X,Y))
@doc doc"""
  Η,κ  tangentONB(M,X,Ξ)
constructs a tangent ONB in $%T_X\mathcal M$ on the [`TangentBundle`](@ref)
whose first vector is `\Xi`. It is constructed by using twice the tangent ONB
of the base manifold and .
"""
function tangentONB(M::TangentBundle,X::TBPoint,Ξ::TBTVector)
  BaseONB,κx = tangentONB(M.manifold, getBase(X), getBaseTangent(Ξ))
  TangentONB,κξ = tangentONB(M.manifold, getBase(X), getTangentTangent(Ξ) )
  return TBTVector.(BaseONB, TangentONB), κx
end
@doc doc"""
    typicalDistance(M)
returns the typical distance on the [`TangentBundle`](@ref), i.e. for
$\mathcal M = T\mathcal N$ we
obtain $t_{\mathcal M} = \sqrt{t_{\mathcal N}^2 + d_{\mathcal N}^2}$, where $d$
denotes the manifold dimension.
"""
typicalDistance(M::TangentBundle) = sqrt(
  typicalDistance(M.manifold)^2 + manifoldDimension(M.manifold)
)
@doc doc"""
    zeroTVector(M,X)
returns a zero vector in the tangent space $T_X\mathcal M$ of the
[`TangentBundle`](@ref) $X=(x,ξ)\in T\mathcal N$ by creating two zero vectors in $T_x\mathcal M$.
"""
zeroTVector(M::TangentBundle, X::TBPoint) = TBTVector(
  zeroTVector(M.manifold,getBase(X)),
  zeroTVector(M.manifold,getBase(X))
)
# Display
# ---
show(io::IO, X::TBPoint) = print(io, "TB($(getBast(X)), $(getTangent(X)))")
show(io::IO, Ξ::TBTVector) = print(io, "TBT($( getBaseTangent(Ξ) ), $( getTangentTangent( Ξ )))")
