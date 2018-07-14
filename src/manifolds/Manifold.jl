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
export distance, exp, log, norm, dot, manifoldDimension
export geodesic, midPoint, addNoise, reflection, jacobiField, AdjointJacobiField
export +, -, *, /, ==, show
export getValue, getBase, checkBase
# introcude new algorithms
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

#
#
# General documentation of exp/log/... and its fallbacks in case of non-implemented tuples
#
#
"""
    addNoise(M,x,σ)
adds noise of standard deviation `σ` to the MPoint `x` on the manifold `M`.
"""
function addNoise(M::mT,x::T,σ::Number)::T where {mT <: Manifold, T <: MPoint}
  sig1 = string( typeof(x) )
  sig2 = string( typeof(σ) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" addNoise – not Implemented for Point $sig1 and standard deviation of type $sig2 on the manifold $sig3.") )
end
"""
    distance(M,x,y)
computes the gedoesic distance between two [`MPoint`](@ref)s `x` and `y` on
a [`Manifold`](@ref) `M`.
"""
function distance{mT <: Manifold, T <: MPoint}(M::mT, x::T, y::T)
  sig1 = string( typeof(x) )
  sig2 = string( typeof(y) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" distance – not Implemented for the two points $sig1 and $sig2 on the manifold $sig3." ) )
end
doc"""
    dot(M, x, ξ, ν)
Computes the inner product of two [`TVector`](@ref)s `ξ` and `ν` from the
tangent space at the [`MPoint`](@ref) `x` on the [`Manifold`](@ref) `M`.
"""
function dot{mT <: Manifold, P <: MPoint, T <: TVector}(M::mT, x::P, ξ::T, ν::T)
  sig1 = string( typeof(ξ) )
  sig2 = string( typeof(ν) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" dot – not Implemented for the two tangential vectors $sig1 and $sig2 on the manifold $sig3." ) )
end
doc"""
    exp(M,x,ξ,[t=1.0])
computes the exponential map at an [`MPoint`](@ref) `x` for the
[`TVector`](@ref) `ξ` on the [`Manifold`](@ref) `M`. The optional parameter `t` can be
used to shorten `ξ` to `tξ`.
"""
function exp{mT<:Manifold, P<:MPoint, T<:TVector}(M::mT, x::P, ξ::T,t::Number=1.0)
  sig1 = string( typeof(x) )
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Exp – not Implemented for Point $sig1 and tangential vector $sig2 on the manifold $sig3." ) )
end
"""
    getValue(x)
get the actual value representing the point `x` on a manifold.
This should be implemented if you do not use the field x.value to avoid the
try-catch in the fallback implementation.
"""
function getValue{P <: MPoint}(x::P)
    try
        return x.value
    catch
        sig1 = string( typeof(x) )
        throw( ErrorException("getValue – not implemented for manifold point $sig1.") );
    end
end
"""
    getValue(ξ)
get the actual value representing the tangent vector `ξ` to a manifold.
This should be implemented if you do not use the field ξ.value to avoid the
try-catch in the fallback implementation.
"""
function getValue{T <: TVector}(ξ::T)
    try
        return ξ.value
    catch
        sig1 = string( typeof(ξ) )
        throw( ErrorException("getValue – not implemented for tangent vector $sig1.") );
    end
end
"""
    log(M,x,y)
computes the [`TVector`](@ref) in the tangent space ``T_x\mathcal M`` at the
[`MPoint`](@ref) `x` such that the corresponding geodesic reaches the
[`MPoint`](@ref) `y` after time 1 on the [`Manifold`](@ref) `M`.
"""
function log{mT<:Manifold, T<:MPoint, S<:MPoint}(M::mT,x::T,y::S)::TVector
  sig1 = string( typeof(x) )
  sig2 = string( typeof(y) )
  sig3 = string( typeof(M) )
  throw( ErrorException("log – not Implemented for Points $sig1 and $sig2 on the manifold $sig3.") )
end
"""
    manifoldDimension(x)
returns the dimension of the manifold `M` the point `x` belongs to.
"""
function manifoldDimension{T<:MPoint}(x::T)::Integer
  sig1 = string( typeof(x) )
  throw( ErrorException(" Not Implemented for manifold points $sig1 " ) )
end
"""
    manifoldDimension(M)
returns the dimension of the manifold `M`.
"""
function manifoldDimension{T<:Manifold}(M::T)::Integer
  sig1 = string( typeof(M) )
  throw( ErrorException(" Not Implemented for manifold $sig1 " ) )
end
doc"""
    norm(M,x,ξ)
  computes the length of a tangential vector $\xi\in T_x\mathcal M$
"""
function norm{mT<:Manifold, T<: MPoint, S<:TVector}(M::mT,x::T,ξ::S)::Number
	sig1 = string( typeof(ξ) )
	sig2 = string( typeof(x) )
	sig3 = string( typeof(M) )
  throw( ErrorException("Norm - Not Implemented for types $sig1 in the tangent space of a $sig2 on the manifold $sig3" ) )
end
doc"""
    parallelTransport(M,x,y,ξ)
Parallel transport of a vector `ξ` given at the tangent space $T_x\mathcal M$
of `x` to the tangent space $T_y\mathcal M$ at `y` along the geodesic form `x` to `y`.
If the geodesic is not unique, this function takes the same choice as `geodesic`.
"""
function parallelTransport{mT<:Manifold, P<:MPoint, Q<:MPoint, T<:TVector}(M::mT, x::P, y::Q, ξ::T)
  sig1 = string( typeof(x) )
  sig2 = string( typeof(y) )
  sig3 = string( typeof(ξ) )
  sig4 = string( typeof(M) )
  throw( ErrorException(" parallelTransport not implemented for Points $sig1 and $sig2, and a tangential vector $sig3 on the manifold $sig4." ) )
end
doc"""
    (Ξ,κ) = tangentONB(M,x,ξ)
compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi$ is the
first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$,
$\dot g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

See also `jacobiField`.
"""
function tangentONB{mT <: Manifold, P <: MPoint, T <: TVector}(M::mT, x::P, ξ::T)
    sig1 = string( typeof(x) )
    sig2 = string( typeof(y) )
    sig3 = string( typeof(ξ) )
    throw( ErrorException("tangentONB Point $sig1 and tangent vector $sig2 on the manifold $sig3." ) )
end
doc"""
    (Ξ,κ) = tangentONB(M,x,y)
compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi=\log_xy$ is the
first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$,
$\dot g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

See also `jacobiField`.
"""
tangentONB{mT <: Manifold, P <: MPoint}(M::mT, x::P, y::P) = tangentONB(M,x,log(M,x,y))
# The extended types for more information/security on base points of tangent vectors
# ---
"""
A decorator pattern based extension of TVector to additionally store the base
point. The decorator is then used to verify, that exp and dot are only called
with correct base points.
"""
struct TVectorE{T <: TVector, P <: MPoint} <: TVector
    vector::T
    base::P
end
getValue{T <: TVectorE}(ξ::T) = getValue(ξ.vector)
"""
    getBase(ξ)
returns the base point of an extended tangent vector.
"""
getBase{T <: TVectorE}(ξ::T) = ξ.base
"""
    getVector(ξ)
returns the internal TVector point of an extended tangent vector.
"""
getVector{T <: TVectorE}(ξ::T) = ξ.vector
"""
A decorator pattern based extension of MPoint to identify when to switch
to the extended `TVectorE` for functions just working on points, e.g. `log`
"""
struct MPointE{P <: MPoint} <: MPoint
    base::P
end
getValue{P <: MPointE}(x::P) = getValue( getBase(x) );
show(io::IO, x::MPointE) = print(io, "$(x)E")
"""
    getBase(x)
returns the point this extended manifold point stores internally.
"""
getBase{P <: MPointE}(x::P) = x.base;

show(io::IO, ξ::TVectorE) = print(io, "$( getValue(ξ) )_$( getValue( getBase(ξ) ) )")
function +{T <: TVectorE}(ξ::T,ν::T)
    checkBase(ξ,ν)
    return T(ξ.value+ν.value,ξ.base)
end
function -{T <: TVectorE}(ξ::T,ν::T)::T
    checkBase(ξ,ν)
    return T(ξ.value-ν.value,ξ.base)
end
"""
    checkBase(ξ,ν)
checks, whether the base of two tangent vectors is identical, if both tangent
vectors are of type `TVectorE`. If one of them is not an extended vector, the
function returns true, expecting the tangent vector implicitly to be correct.
"""
function checkBase{T <: TVectorE}(ξ::T,ν::T)
    if getValue( getBase(ξ) ) != getValue( getBase(ν) )
        throw(
            ErrorException("The two tangent vectors $ξ and $ν do not have the same base.")
        );
    else
        return true;
    end
end
checkBase{T <: TVectorE, S <: TVector}(ξ::T,ν::S) = true
checkBase{T <: TVectorE, S <: TVector}(ξ::S,ν::T) = true
"""
    checkBase(ξ,x)
checks, whether the base of the tangent vector `ξ` is `x`. If `ξ` is not an
extended tangent vector `TVectorE` the function returns true, assuming the base
implicitly to be correct
"""
function checkBase{T <: TVectorE, P <: MPoint}(ξ::T,x::P)
    if getValue( getBase(ξ) ) != getValue(x)
        throw(
            ErrorException("The tangent vector $ξ is not from the tangent space of $x")
        );
    else
        return true;
    end
end
checkBase{T<: TVector, P<: MPoint}(ξ::T,x::P) = true
# unary operators
*{T <: TVectorE}(ξ::T,s::Number)::T = T(s*ξ.value,ξ.base)
*{T <: TVectorE}(s::Number, ξ::T)::T = T(s*ξ.value,ξ.base)
# /
/{T <: TVectorE}(ξ::T,s::Number)::T = T(ξ.value./s,ξ.base)
/{T <: TVectorE}(s::Number, ξ::T)::T = T(s./ξ.value,ξ.base)
-{T <: TVectorE}(ξ::T)::T = T(-ξ.value,ξ.base)
+{T <: TVectorE}(ξ::T)::T = T(ξ.value,ξ.base)

# compare extended vectors
=={T <: TVectorE}(ξ::T,ν::T)::Bool = ( checkBase(ξ,ν) && all(ξ.value==ν.value) )

# extended exp check base and return exp of value if that did not fail
exp{mT<:Manifold, T<:TVectorE, S<:MPointE}(M::mT,x::S,ξ::T)::T = exp(M,getBase(x),ξ)
function exp{mT<:Manifold, T<:TVectorE, S<:MPoint}(M::mT,x::S,ξ::T)::T
    checkBase(ξ,x);
    return exp(M,x, getVector(ξ) );
end
# for extended vectors set the base to true
log{mT<:Manifold, P<:MPointE}(M::mT,x::P,y::P) = TVectorE(log(M,getBase(x),getBase(y)),x);
log{mT<:Manifold, P<:MPointE, Q<:MPoint}(M::mT,x::P,y::Q) = TVectorE(log(M,getVector(x),y),x);
log{mT<:Manifold, P<:MPointE, Q<:MPoint}(M::mT,x::Q,y::P) = TVectorE(log(M,x,getVector(y)),x);
# break down to inner if base
function dot{mT<:Manifold, P <: MPoint, T<:TVectorE}(M::mT, x::P, ξ::T, ν::T)::Float64
    checkBase(ξ,x);
    checkBase(ξ,ν);
    return dot(M, getVector(ξ), getVector(ν) );
end
dot{mT<:Manifold, P <: MPoint, T<:TVectorE, S<:TVector}(M::mT, x::P, ξ::T, ν::S) = dot(M,x, getVector(ξ), ν);
dot{mT<:Manifold, P <: MPoint, T<:TVectorE, S<:TVector}(M::mT, x::P, ξ::S, ν::T) = dot(M,x,ξ, getVector(ν) );
# break down to inner if base is checked
function norm{mT<:Manifold, P <: MPoint, T<:TVectorE}(M::mT, x::P, ξ::T, ν::T)::Float64
    checkBase(ξ,x);
    checkBase(ξ,ν);
    return norm(M,ξ.value,ν.value);
end
norm{mT<:Manifold, P <: MPointE, T<:TVector, S<:TVector}(M::mT,x::P,ξ::T,ν::S) = dot(M,getBase(x),ξ,ν);
norm{mT<:Manifold, P <: MPoint, T<:TVectorE, S<:TVector}(M::mT,x::P,ξ::T,ν::S) = dot(M, getVector(ξ) ,ν);
norm{mT<:Manifold, P <: MPoint, T<:TVectorE, S<:TVector}(M::mT,x::P,ξ::S,ν::T) = dot(M, ξ, getVector(ν));

function show{mT<:Manifold}(io::IO, M::mT)
    try # works if M has a .name field
        print(io, "The Manifold $(M.name).")
    catch
        throw(
            ErrorException("The manifold $( typeof(M) ) seems to not have a `.name` field. Please implement a seperate `show` function.")
        );
    end
end
