#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, MPoint
#  * A point in an tangential space TVector
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base.LinAlg: norm, dot
import Base: exp, log, +, -, *, /, ==, show
# introcude new types
export Manifold, MPoint, TVector, MPointE, TVectorE
# introduce new functions
export geodesic, midPoint, reflection, jacobiField, AdjointJacobiField
export +, -, *, /, ==, show
doc"""
An abstract manifold $\mathcal M$ to keep global information on a specific manifold
"""
abstract type Manifold end

doc"""
An abstract point $x$ on a manifold $\mathcal M$.
"""
abstract type MPoint end

doc"""
A point on a tangent plane $T_x\mathcal M$ at a point $x$ on a
manifold $\mathcal M$.
"""
abstract type TVector end

# scale tangential vectors
*{T <: TVector}(ξ::T,s::Number)::T = T(s* getValue(ξ) )
*{T <: TVector}(s::Number, ξ::T)::T = T(s* getValue(ξ) )
*{T <: TVector}(ξ::Vector{T},s::Number)::T = [ξe*s for ξe in ξ]
*{T <: TVector}(s::Number, ξ::Vector{T}) = [s*ξe for ξe in ξ]
# /
/{T <: TVector}(ξ::T,s::Number)::T = T( getValue(ξ) ./s)
/{T <: TVector}(s::Number, ξ::T)::T = T(s./ getValue(ξ) )
/{T <: TVector}(ξ::Vector{T},s::Number) = [ξe/s for ξe in ξ]
/{T <: TVector}(s::Number, ξ::Vector{T}) = [s/ξe for ξe in ξ]
# + - of TVectors
function +{T <: TVector}(ξ::T,ν::T)
    return T( getValue(ξ) + getValue(ν) )
end
function -{T <: TVector}(ξ::T,ν::T)::T
    return T( getValue(ξ) - getValue(ν) )
end
# unary operators
-{T <: TVector}(ξ::T)::T = T(- getValue(ξ))
+{T <: TVector}(ξ::T)::T = T(getValue(ξ))

# compare Points & vectors
=={T <: MPoint}(x::T, y::T)::Bool = all(getValue(x) == getValue(y) )
=={T <: TVector}(ξ::T,ν::T)::Bool = (  all( getValue(ξ) == getValue(ν) )  )
#
#
# General functions available on manifolds based on exp/log/dist
#
#
doc"""
    ζ = adjointJacobiField(M,x,y,t,η,w)
Compute the AdjointJacobiField $J$ along the geodesic $g_{x,y}$ on the manifold
$\mathcal M$ with initial conditions (depending on the application) $\eta\in T_{g(t;x,y)\mathcal M}$ and
weights $\beta$. The result is a vector $\zeta \in T_x\mathcal M$
The main difference to [`jacobiField`](@ref) is the inversion, that the input $\eta$ and the output $\zeta$ switched tangent spaces.

For detais see [JacobiFields](@ref)
"""
function adjointJacobiField{mT<:Manifold, P<:MPoint, T<:TVector}(M::mT,x::P,y::P,t::Number,η::T,β::Function=βDgx)
    z = geodesic(M,x,y,t); # Point the TzM of the resulting vector lies in
    Ξ,κ = tangentONB(M,x,y) # ONB at x
    Θ = parallelTransport.(M,x,z,Ξ) # Frame at z
    # Decompose wrt. Ξ, multiply with the weights from w and recompose with Θ.
    ξ = sum( ( dot.(M,x,η,Θ) ).* ( β.(κ,t,dist(M,x,y)) ).*Ξ )
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
function midPoint{mT <: Manifold, T <: MPoint}(M::mT,x::T, y::T)::T
  return exp(M,x,0.5*log(x,y))
end
"""
    geodesic(M,x,y)
return a function to evaluate the geodesic connecting `x` and `y`
on the manifold `M`.
"""
function geodesic{mT <: Manifold, T <: MPoint}(M::mT, x::T,y::T)::Function
  return (t::Float64 -> exp(M,x,t*log(M,x,y)))
end
"""
    geodesic(M,x,y,n)
returns vector containing the equispaced n sample-values along the geodesic
from `x`to `y` on the manifold `M`.
"""
function geodesic{mT <: Manifold, T <: MPoint}(M::mT, x::T,y::T,n::Integer)::Vector{T}
  geo = geodesic(M,x,y);
  return [geo(t) for t in linspace(0.,1.,n)]
end
"""
    geodesic(M,x,y,t)
returns the point along the geodesic from `x`to `y` given by the `t`(in [0,1]) on the manifold `M`
"""
geodesic{mT <: Manifold, T <: MPoint}(M::mT,x::T,y::T,t::Number)::T = geodesic(x,y)(t)
"""
    geodesic(M,x,y,T)
returns vector containing the MPoints along the geodesic from `x` to `y` on
the manfiold `M` specified by the points from the vector `T` (of numbers between 0 and 1).
"""
function geodesic{mT <: Manifold, P <: MPoint, S <: Number}(M::mT, x::P,y::P,T::Vector{S})::Vector{T}
  geo = geodesic(M,x,y);
  return [geo(t) for t in T]
end
doc"""
    ζ = jacobiField(M,x,y,t,η,β)
Compute the jacobiField $J$ along the geodesic $g_{x,y}$ on the manifold
$\mathcal M$ with initial conditions (depending on the application) $\eta\in T_x\mathcal M$ and
weights $\beta$. The result is a tangent vector in $\zeta \in T_{g(t;x,y)}\mathcal M$.

For detais see [JacobiFields](@ref).

*See also:* [`adjointJacobiField`](@ref)
"""
function jacobiField{mT<:Manifold, P<:MPoint, T<:TVector}(M::mT,x::P,y::P,t::Number,η::T,β::Function=βDgx)
    z = geodesic(M,x,y,t); # Point the TzM of the resulting vector lies in
    Ξ,κ = tangentONB(M,x,y) # ONB at x
    Θ = parallelTransport.(M,x,z,Ξ) # Frame at z
    # Decompose wrt. Ξ, multiply with the weights from w and recompose with Θ.
    ξ = sum( ( dot.(M,x,η,Ξ) ).* ( β.(κ,t,dist(M,x,y)) ).*Θ )
end
doc"""
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
reflection{mT <: Manifold, P<: MPoint}(M::mT, p::P, x::P) = exp(M,p,-log(M,p,x))
# errors for false combinations of types or nonimplemented cases
include("defaults/manifoldFallbacks.jl")
# Extended Vector decorations
include("defaults/extendedData.jl")

function show{mT<:Manifold}(io::IO, M::mT)
    try # works if M has a .name field
        print(io, "The Manifold $(M.name).")
    catch
        throw(
            ErrorException("The manifold $( typeof(M) ) seems to not have a `.name` field. Please implement a seperate `show` function.")
        );
    end
end
