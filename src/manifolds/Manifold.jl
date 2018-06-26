#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, MPoint
#  * A point in an tangential space MTVector
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base.LinAlg: norm, dot
import Base: exp, log, +, -, *, /, ==, show
# introcude new types
export MPoint, MTVector
# introduce new functions
export distance, exp, log, norm, dot, manifoldDimension
export geodesic, midPoint, addNoise
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
      MTVector - a point on a tangent plane of a base point, which might
  be null if the tangent space is fixed/known to spare memory.
"""
abstract type MTVector end

# scale tangential vectors
*{T <: MTVector}(ξ::T,s::Number)::T = T(s*ξ.value,ξ.base)
*{T <: MTVector}(s::Number, ξ::T)::T = T(s*ξ.value,ξ.base)
*{T <: MTVector}(ξ::Vector{T},s::Number)::T = [ξe*s for ξe in ξ]
*{T <: MTVector}(s::Number, ξ::Vector{T}) = [s*ξe for ξe in ξ]
# /
/{T <: MTVector}(ξ::T,s::Number)::T = T(ξ.value./s,ξ.base)
/{T <: MTVector}(s::Number, ξ::T)::T = T(s./ξ.value,ξ.base)
/{T <: MTVector}(ξ::Vector{T},s::Number) = [ξe/s for ξe in ξ]
/{T <: MTVector}(s::Number, ξ::Vector{T}) = [s/ξe for ξe in ξ]
# + - of MTVectors
function +{T <: MTVector}(ξ::T,ν::T)::T
  if checkBase(ξ,ν)
    return T(ξ.value+ν.value,ξ.base)
  else
    throw(ErrorException("Can't add two tangential vectors belonging to
      different tangential spaces."))
  end
end
function -{T <: MTVector}(ξ::T,ν::T)::T
  if checkBase(ξ,ν)
    return T(ξ.value-ν.value,ξ.base)
  else
    throw(ErrorException("Can't subtract two tangential vectors belonging to
    different tangential spaces."))
  end
end
# unary operators
-{T <: MTVector}(ξ::T)::T = T(-ξ.value,ξ.base)
+{T <: MTVector}(ξ::T)::T = T(ξ.value,ξ.base)

# compare Points & vectors
=={T <: MPoint}(p::T, q::T)::Bool = all(p.value == q.value)
=={T <: MTVector}(ξ::T,ν::T)::Bool = ( checkBase(ξ,ν) && all(ξ.value==ν.value) )

function checkBase{T <: MTVector}(ξ::T, ν::T)::Bool
  if (!isnull(ξ.base)) && (!isnull(ν.base)) &&
		 		(get(ξ.base).value != get(ν.base).value)
	  	throw( ErrorException("checkBase - The two tangent vectors "string(ξ)" and ("string(ν)") do not have the same (non-null) base vector) – use parallel transport before combining these.") )
			return false
  end
	return true
end
function checkBase{T <: MPoint, S <: MTVector}(p::T, ξ::S)::Bool
  if (!isnull(ξ.base)) && (get(ξ.base).value != p.value)
		throw( ErrorException("checkBase - The tangent vector "string(ξ)" does not have the manifold point "string(p)" as a (non-null) base.") )
		return false
	end
	return true
end
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
function dot{mT <: Manifold, T <: MTVector}(M::mT, ξ::T, ν::T)::Number
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
function exp{mT<:Manifold, T<:MPoint, S<:MTVector}(M::mT, p::T, ξ::S,cache::Bool=true)::MPoint
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
	* includeBase (false) : include the base withing the resulting MTVector,
	 	makes validity checks more reliable/possivle but also requires more memory
	* cache (true) : cache intermediate results for a faster exponential map at p for further runs

"""
function log{mT<:Manifold, T<:MPoint, S<:MPoint}(M::mT,p::T,q::S,includeBase::Bool=false,cache::Bool=true)::MTVector
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
    norm(M,ξ)
  computes the lenth of a tangential vector
"""
function norm{mT<:Manifold, T<: MPoint, S<:MTVector}(M::mT,p::T,ξ::S)::Number
	sig1 = string( typeof(ξ) )
	sig2 = string( typeof(p) )
	sig3 = string( typeof(M) )
  throw( ErrorException("Norm - Not Implemented for types $sig1 in the tangent space of a $sig2 on the manifold $sig3" ) )
end
