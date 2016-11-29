#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, ManifoldPoint
#  * A point in an tangential space ManifoldTangentialPoint
#
import Base.LinAlg: norm, dot
import Base: exp, log, mean, median, +, -, *, /, ==
# introcude new types
export ManifoldPoint, ManifoldTangentialPoint
# introduce new functions
export distance, exp, log, norm, dot, manifoldDimension, mean
# introcude new algorithms
export proxTV
"""
    ManifoldPoint - an abstract point on a Manifold
"""
abstract ManifoldPoint

"""
      ManifoldTangentialPoint - a point on a tangent plane of a base point, which might
  be NULL if the tangent space is fixed/known to spare memory.
"""
abstract ManifoldTangentialPoint

#
# Short hand notations for general exp and log
+(p::ManifoldPoint,xi::ManifoldTangentialPoint) = exp(p,xi)
-(p::ManifoldPoint,q::ManifoldPoint) = log(p,q)
# scale tangential vectors
*{T <: ManifoldTangentialPoint}(xi::T,s::Number)::T = T(s*xi.value,xi.base)
*{T <: ManifoldTangentialPoint}(s::Number, xi::T) = T(s*xi.value,xi.base)
*{T <: ManifoldTangentialPoint}(xi::Vector{T},s::Number) = s*ones(length(xi)).*xi
*{T <: ManifoldTangentialPoint}(s::Number, xi::Vector{T}) = s*ones(length(xi)).*xi
# /
/{T <: ManifoldTangentialPoint}(xi::T,s::Number) = T(s./xi.value,xi.base)
/{T <: ManifoldTangentialPoint}(s::Number, xi::T) = T(s./xi.value,xi.base)
/{T <: ManifoldTangentialPoint}(xi::Vector{T},s::Number) = s*ones(length(xi))./xi
/{T <: ManifoldTangentialPoint}(s::Number, xi::Vector{T}) = s*ones(length(xi))./xi
# + -
function +{T <: ManifoldTangentialPoint}(xi::T,nu::T)::T
  if sameBase(xi,nu)
    return T(xi.value+nu.value,xi.base)
  else
    throw(ErrorException("Can't add two tangential vectors belonging to
      different tangential spaces."))
  end
end
function -{T <: ManifoldTangentialPoint}(xi::T,nu::T)::T
  if sameBase(xi,nu)
    return T(xi.value-nu.value,xi.base)
  else
    throw(ErrorException("Can't subtract two tangential vectors belonging to
    different tangential spaces."))
  end
end

# compare Points
=={T <: ManifoldPoint}(p::T, q::T)::Bool = all(p.value == q.value)
=={T <: ManifoldTangentialPoint}(xi::T,nu::T)::Bool = ( sameBase(xi,nu) && all(xi.value==nu.value) )

function sameBase{T <: ManifoldTangentialPoint}(xi::T,nu::T)::Bool
  if (isnull(xi.base) || isnull(nu.base))
    return true # one base null is treated as correct
  elseif xi.base.value == nu.base.value
    return true # if both are given and are the same
  else
    return false
  end
end
#
# proximal Maps
"""
    proxDistanceSquared(f,lambda,g) - proximal map with parameter lambda of
  distance(f,x) for some fixed ManifoldPoint f
"""
function proxDistanceSquared(f::ManifoldPoint,lambda::Float16,x::ManifoldPoint)::ManifoldPoint
  exp(x, lambda/(1+lambda)*log(x,f))
end

"""
    proxTuple = proxTV(lambda,pointTuple)
Compute the proximal map prox_f(x,y) for f(x,y) = dist(x,y) with parameter
lambda
# Arguments
* `lambda` : a real value, parameter of the proximal map
* `pointTuple` : a tuple of size 2 containing two ManifoldPoints x and y
# Returns
* `proxTuple` : resulting two-ManifoldPoint-Tuple of the proximal map
"""
function proxTV(lambda::Float64,pointTuple::Tuple{ManifoldPoint,ManifoldPoint})::Tuple{ManifoldPoint,ManifoldPoint}
  step = min(0.5, lambda/distance(pointTuple[1],pointTuple[2]))
  return (  exp(pointTuple[1], step*log(pointTuple[1],pointTuple[2])),
            exp(pointTuple[2], step*log(pointTuple[2],pointTuple[1])) )
end
"""
    proxTuple = proxTVSquared(lambda,pointTuple)
 Compute the proximal map prox_f(x,y) for f(x,y) = dist(x,y)^2 with parameter
 `lambda`
 # Arguments
 * `lambda` : a real value, parameter of the proximal map
 * `pointTuple` : a tuple of size 2 containing two ManifoldPoints x and y
 # OUTPUT
 * `proxTuple` : resulting two-ManifoldPoint-Tuple of the proximal map
 ---
 ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function proxTVSquared(lambda::Float64,pointTuple::Tuple{ManifoldPoint,ManifoldPoint})::Tuple{ManifoldPoint,ManifoldPoint}
  step = lambda/(1+2*lambda)*distance(pointTuple[1],pointTuple[2])
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
  * `f` an array of `ManifoldPoint`s
  # Output
  * `x` the mean of the values from `f`
  # Optional Parameters
  * `initialValue` (`[]`) start the algorithm with a special initialisation of
  `x`, if not specified, the first value `f[1]` is unsigned
  * `MaxIterations` (`500`) maximal number of iterations
  * `MinimalChange` (`5*10.0^(-7)`) minimal change for the algorithm to stop
  * `Weights` (`[]`) cimpute a weigthed mean, if not specified (`[]`),
  all are choren equal, i.e. `1/n*ones(n)` for `n=length(f)`.
  ---
  ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function mean{T <: ManifoldPoint}(f::Vector{T}; kwargs...)::ManifoldPoint
  # collect optional values
  kwargs_dict = Dict(kwargs);
  x = get(kwargs_dict, "initialValue", f[1])
  Weights = get(kwargs_dict, "Weights", 1/length(f)*ones(length(f)))
  MaxIterations = get(kwargs_dict, "MaxIterations", 50)
  MinimalChange = get(kwargs_dict, "MinimalChange", 5*10.0^(-7))
  iter=0
  xold = x
  while (  ( (distance(x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
    xold = x
    x = exp(x, sum(Weights.*[log(x,fi) for fi in f]))
    iter += 1
  end
  return x
end
"""
    variance(f)
  returns the variance of the set of pints on a maniofold.
"""
function variance{T<:ManifoldPoint}(f::Vector{T})
  meanF = mean(f);
  return 1/( (length(f)-1)*manifoldDimension(f[1]) ) * sum( [ dist(meanF,fi)^2 for fi in f])
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
  * `f` an array of `ManifoldPoint`s
  # Output
  * `x` the mean of the values from `f`
  # Optional Parameters
  * `initialValue` (`[]`) start the algorithm with a special initialisation of
  `x`, if not specified, the first value `f[1]` is unsigned
  * `MaxIterations` (`500`) maximal number of iterations
  * `MinimalChange` (`5*10.0^(-7)`) minimal change for the algorithm to stop
  * `StepSize` (`1`)
  * `Weights` (`[]`) cimpute a weigthed mean, if not specified (`[]`),
  all are choren equal, i.e. `1/n*ones(n)` for `n=length(f)`.
  ---
  ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function median{T <: ManifoldPoint}(f::Vector{T}; kwargs...)::ManifoldPoint
  # collect optional values
  kwargs_dict = Dict(kwargs);
  x = get(kwargs_dict, "initialValue", f[1])
  MaxIterations = get(kwargs_dict, "MaxIterations", 50)
  MinimalChange = get(kwargs_dict, "MinimalChange", 5*10.0^(-7))
  StepSize = get(kwargs_dict, "StepSize",1)
  Weights = get(kwargs_dict, "Weights", 1/length(f)*ones(length(f)))
  iter=0
  xold = x
  while (  ( (distance(x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
    xold = x
    sumDistances = sum( Weights.*[distance(x,fi) for fi in f] )
    x = exp(x, StepSize/sumDistances * sum(Weights.* [ 1/( (distance(x,fi)==0)?1:distance(x,fi) )*log(x,fi) for fi in f]))
    iter += 1
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
 * `p`,`q` : two `ManifoldPoint`s
 # Output
 * `m` : resulting mid point
"""
function midPoint(p::ManifoldPoint, q::ManifoldPoint)::ManifoldPoint
  return exp(p,0.5*log(p,q))
end
"""
    geodesic(p,q)
  return a function to evaluate the geodesic connecting p and q
"""
function geodesic{T <: ManifoldPoint}(p::T,q::T)::Function
  return (t -> exp(p,t*log(p,q)))
end
"""
    geodesic(p,q,n)
  returns vector containing the equispaced n sample-values along the geodesic
"""
function geodesic{T <: ManifoldPoint}(p::T,q::T,n::Integer)::Vector{T}
  geo = geodesic(p,q);
  return [geo(t) for t in linspace(0,1,n)]
end
"""
    geodesic(p,q,t)
  returns the point along the geodesic from `p`to `q` given by the `Float64` `t` (0,1).
"""
geodesic{T <: ManifoldPoint}(p::T,q::T,t::Float64)::T = geodesic(p,q)(t)
"""
    geodesic(p,q,T)
  returns vector containing the points along the geodesic from `p`to `q` given
  by vector of `Float64` `T`.
"""
function geodesic{T <: ManifoldPoint}(p::T,q::T,v::Vector{Float64})::Vector{T}
  geo = geodesic(p,q);
  return [geo(t) for t in v]
end
#
# fallback functions for not yet implemented cases
function distance(p::ManifoldPoint,q::ManifoldPoint)::Float64
       sig1 = string( typeof(p) ); sig2 = string( typeof(q) )
       throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
function dot(xi::ManifoldTangentialPoint,nu::ManifoldTangentialPoint)::Float64
  sig1 = string( typeof(xi) ); sig2 = string( typeof(nu) )
  throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
function exp(p::ManifoldPoint,xi::ManifoldTangentialPoint)::ManifoldPoint
  sig1 = string( typeof(p) ); sig2 = string( typeof(xi) )
  throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
function log(p::ManifoldPoint,q::ManifoldPoint)::ManifoldTangentialPoint
  sig1 = string( typeof(p) ); sig2 = string( typeof(q) )
  throw( ErrorException(" Not Implemented for types $sig1 and $sig2 " ) )
end
function manifoldDimension(p::ManifoldPoint)::Integer
  sig1 = string( typeof(p) );
  throw( ErrorException(" Not Implemented for types $sig1 " ) )
end
function norm(xi::ManifoldTangentialPoint)::Float64
  sig1 = string( typeof(xi) );
  throw( ErrorException(" Not Implemented for types $sig1 " ) )
end
