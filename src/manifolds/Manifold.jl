#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, MPoint
#  * A point in an tangential space TVector
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base.LinAlg: norm, dot
import Base: exp, log, +, -, *, /, ==, show
# introcude new types
export Manifold, MPoint, TVector
# introduce new functions
export distance, exp, log, norm, dot, manifoldDimension
export geodesic, midPoint, addNoise
export +, -, *, /, ==, show
# introcude new algorithms
"""
Manifold - an abstract Manifold to keep global information on a specific manifold
"""
abstract type Manifold end

"""
MPoint - an abstract point on a Manifold
"""
abstract type MPoint end

"""
TVector - a point on a tangent plane of a base point, which might
be null if the tangent space is fixed/known to spare memory.
"""
abstract type TVector end

# scale tangential vectors
*{T <: TVector}(ξ::T,s::Number)::T = T(s*ξ.value)
*{T <: TVector}(s::Number, ξ::T)::T = T(s*ξ.value)
*{T <: TVector}(ξ::Vector{T},s::Number)::T = [ξe*s for ξe in ξ]
*{T <: TVector}(s::Number, ξ::Vector{T}) = [s*ξe for ξe in ξ]
# /
/{T <: TVector}(ξ::T,s::Number)::T = T(ξ.value./s)
/{T <: TVector}(s::Number, ξ::T)::T = T(s./ξ.value)
/{T <: TVector}(ξ::Vector{T},s::Number) = [ξe/s for ξe in ξ]
/{T <: TVector}(s::Number, ξ::Vector{T}) = [s/ξe for ξe in ξ]
# + - of TVectors
function +{T <: TVector}(ξ::T,ν::T)
    return T(ξ.value+ν.value)
end
function -{T <: TVector}(ξ::T,ν::T)::T
    return T(ξ.value-ν.value,ξ.base)
end
# unary operators
-{T <: TVector}(ξ::T)::T = T(-ξ.value)
+{T <: TVector}(ξ::T)::T = T(ξ.value)

# compare Points & vectors
=={T <: MPoint}(p::T, q::T)::Bool = all(p.value == q.value)
=={T <: TVector}(ξ::T,ν::T)::Bool = ( checkBase(ξ,ν) && all(ξ.value==ν.value) )

# Decorator pattern for keeping and checking base
struct TVectorE{T <: TVector, P <: MPoint} <: TVector
    vector::T
    base::P
    TVectorE{T,P}(value::T,base::P) where {T <: TVector, P <: MPoint} = new(value,base)
end
show(io::IO, ξ::TVectorE) = print(io, "$(ξ.value)_$(ξ.base)")
function +{T <: TVectorE}(ξ::T,ν::T)
    checkBase(ξ,ν)
    return T(ξ.value+ν.value,ξ.base)
end
function -{T <: TVectorE}(ξ::T,ν::T)::T
    checkBase(ξ,ν)
    return T(ξ.value-ν.value,ξ.base)
end
function checkBase{T <: TVectorE}(ξ::T,ν::T)
    if ξ.base != ν.base
        throw(
            ErrorException("The two tangent vectors do not have the same base.")
        );
    else
        return true;
    end
end
function checkBase{T <: TVectorE, P <: MPoint}(ξ::T,x::P)
    if ξ.base != x
        throw(
            ErrorException("The tangent vector is not from the tangent space of $x")
        );
    else
        return true;
    end
end
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
function exp{mT<:Manifold, T<:TVectorE, S<:MPoint}(M::mT,p::S,ξ::T)::T
    checkBase(p,ξ);
    return exp(p,ξ.value);
end
# for extended vectors set the base to true
function log{mT<:Manifold, S<:MPoint}(M::mT,p::S,q::S,base=false)
    if base
        return TVectorE(log(M,p,q),p);
    else
        return log(M,p,q);
    end
end
# break down to inner if base
function dot{mT<:Manifold, T<:TVectorE}(M::mT,ξ::T,ν::T)::Float64
    checkBase()
    return dot(M,ξ.value,ν.value);
end
# break down to inner for differents
function dot{mT<:Manifold, T<:TVectorE, S<:TVector}(M::mT,ξ::T,ν::S)::Float64
    return dot(M,ξ.value,ν);
end
function dot{mT<:Manifold, T<:TVectorE, S<:TVector}(M::mT,ξ::S,ν::T)::Float64
    return dot(M,ξ.value,ν);
end
# do the log and ep with extended points
#
#
# Mid point and geodesics
"""
    midPoint(M,x,z)
  Compute the (geodesic) mid point of x and z.
  # Arguments
  * 'M' – a manifold
  * `p`,`q` – two `MPoint`s
  # Output
  * `m` – resulting mid point
"""
function midPoint{mT <: Manifold, T <: MPoint}(M::mT,p::T, q::T)::T
  return exp(M,p,0.5*log(p,q))
end
"""
    geodesic(M,p,q)
  return a function to evaluate the geodesic connecting `p` and `q`
  on the manifold `M`.
"""
function geodesic{mT <: Manifold, T <: MPoint}(M::mT, p::T,q::T)::Function
  return (t::Float64 -> exp(M,p,t*log(M,p,q)))
end
"""
    geodesic(M,p,q,n)
  returns vector containing the equispaced n sample-values along the geodesic
  from `p`to `q` on the manifold `M`.
"""
function geodesic{mT <: Manifold, T <: MPoint}(M::mT, p::T,q::T,n::Integer)::Vector{T}
  geo = geodesic(M,p,q);
  return [geo(t) for t in linspace(0.,1.,n)]
end
"""
    geodesic(M,p,q,t)
  returns the point along the geodesic from `p`to `q` given by the `t`(in [0,1]) on the manifold `M`
"""
geodesic{mT <: Manifold, T <: MPoint}(M::mT,p::T,q::T,t::Number)::T = geodesic(p,q)(t)
"""
    geodesic(Mp,q,T)
  returns vector containing the MPoints along the geodesic from `p`to `q` on
  the manfiold `M` specified within the vector `T` (of numbers between 0 and 1).
"""
function geodesic{mT <: Manifold, T <: MPoint, S <: Number}(M::mT, p::T,q::T,v::Vector{S})::Vector{T}
  geo = geodesic(M,p,q);
  return [geo(t) for t in v]
end
#
# fallback functions for not yet implemented cases – for example also for the
# cases where you take the wrong manifolg for certain points
"""
    addNoise(M,P,σ)
  adds noise of standard deviation `σ` to the MPoint `p` on the manifold `M`.
"""
function addNoise{mT <: Manifold, T <: MPoint}(M::mT,P::T,σ::Number)::T
  sig1 = string( typeof(P) )
  sig2 = string( typeof(σ) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" addNoise – not Implemented for Point $sig1 and standard deviation of type $sig2 on the manifold $sig3.") )
end
"""
    distance(M,p,q)
  computes the gedoesic distance between two points `p`and `q`
  on a manifold `M`.
"""
function distance{mT <: Manifold, T <: MPoint}(M::mT, p::T,q::T)::Number
  sig1 = string( typeof(p) )
  sig2 = string( typeof(q) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Distance – not Implemented for the two points $sig1 and $sig2 on the manifold $sig3." ) )
end
"""
    dot(M,ξ,ν)
  computes the inner product of two tangential vectors ξ=ξp and ν=νp in TpM
  of p on the manifold `M`.
"""
function dot{mT <: Manifold, T <: TVector}(M::mT, ξ::T, ν::T)::Number
  sig1 = string( typeof(ξ) )
  sig2 = string( typeof(ν) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Dot – not Implemented for the two tangential vectors $sig1 and $sig2 on the manifold $sig3." ) )
end
"""
    exp(M,p,ξ)
  computes the exponential map at p for the tangential vector ξ=ξ_p
  on the manifold `M`.

	Optional Arguments (standard value)
	* cache (true) : cache intermediate results for a faster exponential map at p for further runs
"""
function exp{mT<:Manifold, T<:MPoint, S<:TVector}(M::mT, p::T, ξ::S)::T
  sig1 = string( typeof(p) )
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Exp – not Implemented for Point $sig1 and tangential vector $sig2 on the manifold $sig3." ) )
end
"""
    log(M,p,q)
  computes the tangential vector at p whose unit speed geodesic reaches q after time T = distance(Mp,q) (not t) (note that the geodesic above is [0,1]
  parametrized).

	Optional Arguments (standard value)
	* includeBase (false) : include the base withing the resulting TVector,
	 	makes validity checks more reliable/possivle but also requires more memory
	* cache (true) : cache intermediate results for a faster exponential map at p for further runs

"""
function log{mT<:Manifold, T<:MPoint, S<:MPoint}(M::mT,p::T,q::S)::TVector
  sig1 = string( typeof(p) )
  sig2 = string( typeof(q) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Log – not Implemented for Points $sig1 and $sig2 on the manifold $sig3.") )
end
"""
    manifoldDimension(M) or manifoldDimension(p)
  returns the dimension of the manifold `M` the point p belongs to.
"""
function manifoldDimension{T<:MPoint}(p::T)::Integer
  sig1 = string( typeof(p) )
  throw( ErrorException(" Not Implemented for manifodl points $sig1 " ) )
end
function manifoldDimension{T<:Manifold}(M::T)::Integer
  sig1 = string( typeof(M) )
  throw( ErrorException(" Not Implemented for manifold $sig1 " ) )
end
"""
    norm(M,x,ξ)
  computes the lenth of a tangential vector in TxM
"""
function norm{mT<:Manifold, T<: MPoint, S<:TVector}(M::mT,x::T,ξ::S)::Number
	sig1 = string( typeof(ξ) )
	sig2 = string( typeof(x) )
	sig3 = string( typeof(M) )
  throw( ErrorException("Norm - Not Implemented for types $sig1 in the tangent space of a $sig2 on the manifold $sig3" ) )
end
