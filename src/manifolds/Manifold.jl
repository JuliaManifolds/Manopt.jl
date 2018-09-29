#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, MPoint
#  * A point in an tangential space TVector
#
# Manopt.jl, R. Bergmann, 2018-06-26
import LinearAlgebra: norm, dot
import Base: exp, log, +, -, *, /, ==, show
# introcude new types
export Manifold, MPoint, TVector, MPointE, TVectorE
# introduce new functions
export geodesic, midPoint, reflection, jacobiField, adjointJacobiField
export +, -, *, /, ==, show
@doc doc"""
An abstract manifold $\mathcal M$ to keep global information on a specific manifold
"""
abstract type Manifold end

@doc doc"""
An abstract point $x$ on a manifold $\mathcal M$.
"""
abstract type MPoint end

@doc doc"""
A point on a tangent plane $T_x\mathcal M$ at a point $x$ on a
manifold $\mathcal M$.
"""
abstract type TVector end

# scale tangential vectors
*(ξ::T,s::Number) where {T <: TVector} = T(s* getValue(ξ) )
*(s::Number, ξ::T) where {T <: TVector} = T(s* getValue(ξ) )
*(ξ::Vector{T},s::Number) where {T <: TVector} = [ξe*s for ξe in ξ]
*(s::Number, ξ::Vector{T}) where {T <: TVector} = [s*ξe for ξe in ξ]
# /
/(ξ::T,s::Number) where {T <: TVector} = T( getValue(ξ) ./s)
/(s::Number, ξ::T) where {T <: TVector} = T(s./ getValue(ξ) )
/(ξ::Vector{T},s::Number) where {T <: TVector} = [ξe/s for ξe in ξ]
/(s::Number, ξ::Vector{T}) where {T <: TVector} = [s/ξe for ξe in ξ]
# + - of TVectors
function +(ξ::T,ν::T) where {T <: TVector}
    return T( getValue(ξ) + getValue(ν) )
end
function -(ξ::T,ν::T) where {T <: TVector}
    return T( getValue(ξ) - getValue(ν) )
end
# unary operators
-(ξ::T) where {T <: TVector} = T(- getValue(ξ))
+(ξ::T) where {T <: TVector} = T(getValue(ξ))

# compare Points & vectors
==(x::T, y::T) where {T <: MPoint} = all(getValue(x) == getValue(y) )
==(ξ::T,ν::T) where {T <: TVector} = (  all( getValue(ξ) == getValue(ν) )  )
#
#
# General functions available on manifolds based on exp/log/dist
#
#
@doc doc"""
    ζ = adjointJacobiField(M,x,y,t,η,w)
Compute the AdjointJacobiField $J$ along the geodesic $g_{x,y}$ on the manifold
$\mathcal M$ with initial conditions (depending on the application) $\eta\in T_{g(t;x,y)\mathcal M}$ and
weights $\beta$. The result is a vector $\zeta \in T_x\mathcal M$
The main difference to [`jacobiField`](@ref) is the inversion, that the input $\eta$ and the output $\zeta$ switched tangent spaces.

For detais see [`jacobiField`](@ref)
"""
function adjointJacobiField(M::mT,x::P,y::P,t::Number,η::T,β::Function=βDgx) where {mT<:Manifold, P<:MPoint, T<:TVector}
    z = geodesic(M,x,y,t); # Point the TzM of the resulting vector lies in
    Ξ,κ = tangentONB(M,x,y) # ONB at x
    Θ = parallelTransport.(Ref(M),Ref(x),Ref(z),Ξ) # Frame at z
    # Decompose wrt. Ξ, multiply with the weights from w and recompose with Θ.
    ξ = sum( ( dot.(Ref(M),Ref(x),Ref(η),Θ) ).* ( β.(κ,Ref(t),distance(M,x,y)) ).*Ξ )
end
"""
   midPoint(M,x,y,z)
computes the mid point between x and y. If there is more than one mid point
of (not neccessarily miniizing) geodesics (i.e. on the sphere), the one nearest
to z.
"""
function midPoint(M::mT,x::T,y::T,z::T)::T where {mT <: Manifold, T <: MPoint}
    # since this is the fallback, it just uses the non-nearest one
    return midPoint(M,x,y)
end
"""
    midPoint(M,x,y)
Compute the (geodesic) mid point of x and y.
# Arguments
* `M` – a manifold
* `x`,`y` – two `MPoint`s on `M`
# Output
* `m` – resulting mid point
"""
function midPoint(M::mT,x::T, y::T)::T where {mT <: Manifold, T <: MPoint}
  return exp(M,x,0.5*log(M,x,y))
end
"""
    geodesic(M,x,y)
return a function to evaluate the geodesic connecting `x` and `y`
on the manifold `M`.
"""
function geodesic(M::mT, x::T,y::T)::Function where {mT <: Manifold, T <: MPoint}
  g(t) =  exp(M,x,t*log(M,x,y))
  return g
end
"""
    geodesic(M,x,y,n)
returns vector containing the equispaced n sample-values along the geodesic
from `x`to `y` on the manifold `M`.
"""
function geodesic(M::mT, x::T,y::T,n::Integer)::Vector{T} where {mT <: Manifold, T <: MPoint}
  geo = geodesic(M,x,y);
  return [geo(t) for t in range(0., stop=1.,length=n)]
end
"""
    geodesic(M,x,y,t)
returns the point along the geodesic from `x`to `y` given by the `t`(in [0,1]) on the manifold `M`
"""
geodesic(M::mT,x::T,y::T,t::N) where {mT <: Manifold, T <: MPoint, N <: Number} = geodesic(M,x,y)(t)
"""
    geodesic(M,x,y,T)
returns vector containing the MPoints along the geodesic from `x` to `y` on
the manfiold `M` specified by the points from the vector `T` (of numbers between 0 and 1).
"""
function geodesic(M::mT, x::P,y::P,T::Vector{S})::Vector{T} where {mT <: Manifold, P <: MPoint, S <: Number}
  geo = geodesic(M,x,y);
  return [geo(t) for t in T]
end
@doc doc"""
    ζ = jacobiField(M,x,y,t,η,β)
Compute the jacobiField $J$ along the geodesic $g_{x,y}$ on the manifold
$\mathcal M$ with initial conditions (depending on the application) $\eta\in T_x\mathcal M$ and
weights $\beta$. The result is a tangent vector in $\zeta \in T_{g(t;x,y)}\mathcal M$.

*See also:* [`adjointJacobiField`](@ref)
"""
function jacobiField(M::mT,x::P,y::P,t::Number,η::T,β::Function=βDgx) where {mT<:Manifold, P<:MPoint, T<:TVector}
    z = geodesic(M,x,y,t); # Point the TzM of the resulting vector lies in
    Ξ,κ = tangentONB(M,x,y) # ONB at x
    Θ = parallelTransport.(Ref(M),Ref(x),Ref(z),Ξ) # Frame at z
    # Decompose wrt. Ξ, multiply with the weights from w and recompose with Θ.
    ξ = sum( ( dot.(Ref(M),Ref(x),Ref(η),Ξ) ).* ( β.(κ,Ref(t),Ref(distance(M,x,y))) ).*Θ )
end

@doc doc"""
    y = reflection(M,p,x)
reflect the `MPoint x` at `MPoint p`, i.e. compute
$y = R_p(x) = \exp_p(-\log_px)$. On Euclidean space this results in the point
reflection $R_p(x) = p - (x-p) = 2p-x$.

# Arguments
* `M`  :   a manifold $\mathcal M$
* `p`  :   an `MPoint` $p\in\mathcal M$ to relfect at
* `x`  :   an `MPoint` $x\in\mathcal M$ that is reflected

# Output
* `y`  :  the resulting reflection.
"""
reflection(M::mT, p::P, x::P) where {mT <: Manifold, P<: MPoint} = exp(M,p,-log(M,p,x))
# errors for false combinations of types or nonimplemented cases
include("defaults/manifoldFallbacks.jl")
# Extended Vector decorations
include("defaults/extendedData.jl")

function show(io::IO, M::mT) where {mT<:Manifold}
    try # works if M has a .name field
        print(io, "The Manifold $(M.name).")
    catch
        throw(
            ErrorException("The manifold $( typeof(M) ) seems to not have a `.name` field. Please implement a seperate `show` function.")
        );
    end
end
