#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, MPoint
#  * A point in an tangential space MTVector
#
import Base.LinAlg: norm, dot
import Base: exp, log, mean, median, +, -, *, /, ==, show
# introcude new types
export MPoint, MTVector
# introduce new functions
export distance, exp, log, norm, dot, manifoldDimension
export mean, median, variance, geodesic, midPoint, addNoise
# introcude new algorithms
export proxTV, proxDistanceSquared, proxTVSquared
"""
    Manifold - an abstract Manifold to keep global information on a specific manifold
"""
abstract Manifold

"""
    MPoint - an abstract point on a Manifold
"""
abstract MPoint

"""
      MTVector - a point on a tangent plane of a base point, which might
  be null if the tangent space is fixed/known to spare memory.
"""
abstract MTVector

# scale tangential vectors
*{T <: MTVector}(ξ::T,s::Number)::T = T(s*ξ.value,ξ.base)
*{T <: MTVector}(s::Number, ξ::T) = T(s*ξ.value,ξ.base)
*{T <: MTVector}(ξ::Vector{T},s::Number) = s*ones(length(ξ))*ξ
*{T <: MTVector}(s::Number, ξ::Vector{T}) = s*ones(length(ξ))*ξ
# /
/{T <: MTVector}(ξ::T,s::Number) = T(s/ξ.value,ξ.base)
/{T <: MTVector}(s::Number, ξ::T) = T(s/ξ.value,ξ.base)
/{T <: MTVector}(ξ::Vector{T},s::Number) = s*ones(length(ξ))/ξ
/{T <: MTVector}(s::Number, ξ::Vector{T}) = s*ones(length(ξ))/ξ
# + - of MTVectors
function +{T <: MTVector}(ξ::T,ν::T)::T
  if sameBase(ξ,ν)
    return T(ξ.value+ν.value,ξ.base)
  else
    throw(ErrorException("Can't add two tangential vectors belonging to
      different tangential spaces."))
  end
end
function -{T <: MTVector}(ξ::T,ν::T)::T
  if sameBase(ξ,ν)
    return T(ξ.value-ν.value,ξ.base)
  else
    throw(ErrorException("Can't subtract two tangential vectors belonging to
    different tangential spaces."))
  end
end

# compare Points
=={T <: MPoint}(p::T, q::T)::Bool = all(p.value == q.value)
=={T <: MTVector}(ξ::T,ν::T)::Bool = ( sameBase(ξ,ν) && all(ξ.value==ν.value) )

function sameBase{T <: MTVector}(ξ::T, ν::T)::Bool
  if (isnull(ξ.base) || isnull(ν.base))
    return true # one base null is treated as correct
  elseif ξ.base.value == ν.base.value
    return true # if both are given and are the same
  else
    return false
  end
end
#
# proximal Maps
"""
    proxDistance(f,g,λ)
  compute the proximal map with parameter λ of `distance(f,x)` for some fixed
  `MPoint` f
"""
function proxDistance{T <: MPoint}(f::T,x::T,λ::Number)::T
  exp(x, min(λ, distance(f,x))*log(x,f))
end

"""
    proxDistanceSquared(f,g,λ)
  computes the proximal map of distance^2(f,x) for some
  fixed `MPoint` f with parameter `λ`
"""
function proxDistanceSquared{T <: MPoint}(f::T,λ::Number,x::T)::T
  return exp(x, λ/(1+λ)*log(x,f) )
end
"""
    proxTuple = proxTV(pointTuple,λ)
Compute the proximal map prox_f(x,y) for f(x,y) = dist(x,y) with parameter
`λ`
# Arguments
* `pointTuple` : a tuple of size 2 containing two MPoints x and y
* `λ` : a real value, parameter of the proximal map
# Returns
* `proxTuple` : resulting two-MPoint-Tuple of the proximal map
"""
function proxTV{T <: MPoint}(pointTuple::Tuple{T,T},λ::Number)::Tuple{T,T}
  step = min(0.5, λ/distance(pointTuple[1],pointTuple[2]))
  return (  exp(pointTuple[1], step*log(pointTuple[1],pointTuple[2])),
            exp(pointTuple[2], step*log(pointTuple[2],pointTuple[1])) )
end
"""
    proxTuple = proxTVSquared(pointTuple,λ)
 Compute the proximal map prox_f(x,y) for f(x,y) = dist(x,y)^2 with parameter
 `λ`
 # Arguments
 * `pointTuple` : a tuple of size 2 containing two MPoints x and y
 * `λ` : a real value, parameter of the proximal map
 # OUTPUT
 * `proxTuple` : resulting two-MPoint-Tuple of the proximal map
 ---
 ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function proxTVSquared{T <: MPoint}(pointTuple::Tuple{T,T},λ::Number)::Tuple{T,T}
  step = λ/(1+2*λ)*distance(pointTuple[1],pointTuple[2])
  return (  exp(pointTuple[1], step*log(pointTuple[1],pointTuple[2])),
            exp(pointTuple[2], step*log(pointTuple[2],pointTuple[1])) )
end
#
# median & mean
"""
    mean(f;initialValue=[], MaxIterations=50, MinimalChange=5*10.0^(-7),
    Weights=[])

  calculates the Riemannian Center of Mass (Karcher mean) of the input data `f`
  with a gradient descent algorithm.
  >This implementation is based on
  >B. Afsari, Riemannian Lp center of mass: Existence, uniqueness, and convexity,
  >Proc. AMS 139(2), pp.655-673, 2011.

  # Input
  * `f` an array of `MPoint`s
  # Output
  * `x` the mean of the values from `f`
  # Optional Parameters
  * `initialValue` (`[]`) start the algorithm with a special initialisation of
  `x`, if not specified, the first value `f[1]` is unsigned
  * `λ` (`2`) initial value for the λ of the CPP algorithm
  * `MaxIterations` (`500`) maximal number of iterations
  * `Method` (`GD`) wether to use Gradient Descent (`GD`) or Cyclic Proximal
    Point (`CPP`) algorithm
  * `MinimalChange` (`5*10.0^(-7)`) minimal change for the algorithm to stop
  * `Weights` (`[]`) cimpute a weigthed mean, if not specified (`[]`),
  all are choren equal, i.e. `1/n*ones(n)` for `n=length(f)`.
  ---
  ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function mean{T <: MPoint}(f::Vector{T}; kwargs...)::T
  # collect optional values
  kwargs_dict = Dict(kwargs);
  x = get(kwargs_dict, "initialValue", f[1])
  Weights = get(kwargs_dict, "Weights", 1/length(f)*ones(length(f)))
  MaxIterations = get(kwargs_dict, "MaxIterations", 50)
  MinimalChange = get(kwargs_dict, "MinimalChange", 5*10.0^(-7))
  Method = get(kwargs_dict, "Method", "Geodesic Distance")
  λ = get(kwargs_dict, "λ", 2)
  iter=0
  xold = x
  if Method == "GD"
    while (  ( (distance(x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
      xold = x
      x = exp(x, sum(Weights.*[log(x,fi) for fi in f]))
      iter += 1
    end
  elseif Method == "CPP"
    while (  ( (distance(x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
      xold = x
      for i=1:lengthh(f)
        x = proxDistanceSquared(x,f[i])
      end
      iter += 1
    end
  else
    throw(ErrorException("Unknown Method to compute the mean."))
  end
  return x
end
"""
    median(f;initialValue=[], MaxIterations=50, MinimalChange=5*10.0^(-7),
    StepSize=1, Weights=[])

  calculates the Riemannian Center of Mass (Karcher mean) of the input data `f`
  with a gradient descent algorithm. This implementation is based on
  >B. Afsari, Riemannian Lp center of mass: Existence, uniqueness, and convexity,
  >Proc. AMS 139(2), pp.655–673, 2011.

  and
  > P. T. Fletcher, S. Venkatasubramanian, and S. Joshi: The geometric median on
  > Riemannian manifolds with application to robust atlas estimation,
  > NeuroImage 45, pp. S143–152

  # Input
  * `f` an array of `MPoint`s
  # Output
  * `x` the mean of the values from `f`
  # Optional Parameters
  * `initialValue` (`[]`) start the algorithm with a special initialisation of
  `x`, if not specified, the first value `f[1]` is unsigned
  * `λ` (`2`) initial value for the λ of the CPP algorithm
  * `MaxIterations` (`500`) maximal number of iterations
  * `Method` (`GD`) wether to use Gradient Descent (`GD`) or Cyclic Proximal
    Point (`CPP`) algorithm
  * `MinimalChange` (`5*10.0^(-7)`) minimal change for the algorithm to stop
  * `StepSize` (`1`)
  * `Weights` (`[]`) cimpute a weigthed mean, if not specified (`[]`),
  all are choren equal, i.e. `1/n*ones(n)` for `n=length(f)`.
  ---
  ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function median{T <: MPoint}(f::Vector{T}; kwargs...)::T
  # collect optional values
  kwargs_dict = Dict(kwargs);
  x = get(kwargs_dict, "initialValue", f[1])
  MaxIterations = get(kwargs_dict, "MaxIterations", 50)
  MinimalChange = get(kwargs_dict, "MinimalChange", 5*10.0^(-7))
  StepSize = get(kwargs_dict, "StepSize",1)
  Weights = get(kwargs_dict, "Weights", 1/length(f)*ones(length(f)))
  iter=0
  xold = x
  if Method == "GD"
    while (  ( (distance(x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
      xold = x
      sumDistances = sum( Weights.*[distance(x,fi) for fi in f] )
      x = exp(x, StepSize/sumDistances * sum(Weights.* [ 1/( (distance(x,fi)==0)?1:distance(x,fi) )*log(x,fi) for fi in f]))
      iter += 1
    end
  elseif Method == "CPP"
    while (  ( (distance(x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
      xold = x
      for i=1:lengthh(f)
        x = proxDistance(x,f[i])
      end
      iter += 1
    end
  else
    throw(ErrorException("Unknown Method to compute the mean."))
  end
  return x
end
#
#
# Mid point and geodesics
"""
    midPoint(x,z)
  Compute the (geodesic) mid point of x and z.
 # Arguments
 * `p`,`q` : two `MPoint`s
 # Output
 * `m` : resulting mid point
"""
function midPoint{T <: MPoint}(p::T, q::T)::T
  return exp(p,0.5*log(p,q))
end
"""
    geodesic(p,q)
  return a function to evaluate the geodesic connecting p and q
"""
function geodesic{T <: MPoint}(p::T,q::T)::Function
  return (t -> exp(p,t*log(p,q)))
end
"""
    geodesic(p,q,n)
  returns vector containing the equispaced n sample-values along the geodesic
"""
function geodesic{T <: MPoint}(p::T,q::T,n::Integer)::Vector{T}
  geo = geodesic(p,q);
  return [geo(t) for t in linspace(0,1,n)]
end
"""
    geodesic(p,q,t)
  returns the point along the geodesic from `p`to `q` given by the `Float64` `t` (0,1).
"""
geodesic{T <: MPoint}(p::T,q::T,t::Number)::T = geodesic(p,q)(t)
"""
    geodesic(p,q,T)
  returns vector containing the points along the geodesic from `p`to `q` given
  by vector of `Number`s.
"""
function geodesic{T <: MPoint, S <: Number}(p::T,q::T,v::Vector{S})::Vector{T}
  geo = geodesic(p,q);
  return [geo(t) for t in v]
end
#
# fallback functions for not yet implemented cases
"""
    addNoise(P,σ)
  adds noise of standard deviation `σ` to the manifod valued data array `P`.
"""
function addNoise{T <: MPoint}(P::T,σ::Number)::T
  sig1 = string( typeof(P) ); sig2 = string( typeof(aigma) )
  throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
"""
    distance(p,q)
  computes the gedoesic distance between two points on a manifold
"""
function distance(p::MPoint,q::MPoint)::Number
  sig1 = string( typeof(p) ); sig2 = string( typeof(q) )
  throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
"""
    dot(ξ,ν)
  computes the inner product of two tangential vectors, if they are in the same
  tangential space
"""
function dot(ξ::MTVector,ν::MTVector)::Number
  sig1 = string( typeof(ξ) ); sig2 = string( typeof(ν) )
  throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
"""
    exp(p,ξ)
  computes the exponential map at p for the tangential vector ξ
"""
function exp(p::MPoint,ξ::MTVector)::MPoint
  sig1 = string( typeof(p) ); sig2 = string( typeof(ξ) )
  throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
"""
    log(p,q)
  computes the tangential vector at p whose geodesic reaches q after time
  T = distance(p,q)
"""
function log(p::MPoint,q::MPoint)::MTVector
  sig1 = string( typeof(p) ); sig2 = string( typeof(q) )
  throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
"""
    manifoldDimension(p)
  returns the dimension of the manifold the point p belongs to.
"""
function manifoldDimension(p::MPoint)::Integer
  sig1 = string( typeof(p) );
  throw( ErrorException(" Not Implemented for types $sig1 " ) )
end
"""
    norm(ξ)
  computes the lenth of a tangential vector
"""
function norm(ξ::MTVector)::Number
  sig1 = string( typeof(ξ) );
  throw( ErrorException(" Not Implemented for types $sig1 " ) )
end
"""
    variance(f)
  returns the variance of the set of pints on a maniofold.
"""
function variance{T<:MPoint}(f::Vector{T})::Number
  meanF = mean(f);
  return 1/( (length(f)-1)*manifoldDimension(f[1]) ) * sum( [ dist(meanF,fi)^2 for fi in f])
end
