#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, ManifoldPoint
#  * A point in an tangential space ManifoldTangentialPoint
#
import Base.LinAlg.norm, Base.LinAlg.dot, Base.exp, Base.log, Base.+,Base.-,Base.*
# introcude new types
export ManifoldPoint, ManifoldTangentialPoint
# introduce new functions
export distance, exp, log, norm, dot, manifoldDimension
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
---
ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
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
# CPPA _TV
"""
    TV_Regularization_CPPA(f,alpha, lambda) - compute the TV regularization model of
given data array f and paramater alpha and internal operator start lambda.

# Arguments
* `f` an d-dimensional array of `ManifoldPoint`s
* `alpha` parameter of the model
* `lambda` internal parameter of the cyclic proxximal point algorithm
# Output
* `x` the regulraized array
# Optional Parameters
* `MinimalChange` (`10.0^(-5)`) minimal change for the algorithm to stop
* `MaxIterations` (`500`) maximal number of iterations
---
 ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function TV_Regularization_CPPA(lambda::Float64, alpha::Float64, f::Array{ManifoldPoint};
           MinimalChange=10.0^(-5), MaxIterations=500)::Array{ManifoldPoint}
  x = deepcopy(f)
  xold = deepcopy(x)
  iter = 1
  while ( (sum( [ distance(xi,xoldi) for (xi,xoldi) in zip(x,xold) ] ) > MinimalChange)
    && (iter < MaxIterations) )
    # First term: d(f,x)^2
    for i in eachindex(x)
      x[i] = proxDistanceSquared(f[i],lambda/i,x[i])
    end
    # TV term
    for d in 1:ndims(f)
      for i in eachindex(f)
        # neighbor index
        i2 = i; i2[d] += 1;
        if ( all(i2 <=size(A)) )
          (x[i], x[i2]) = proxTV(alpha*lambda/i,(x[i], x[i2]))
        end
      end
    end
    iter++
    xold = deepcopy(x)
  end
end
#
# other auxilgary functions
"""
midPoint(x,z) - Compute the (geodesic) mid point of x and z.
# Arguments
* `x`,`z` : two `ManifoldPoint`s
# Output
* `m` : resulting mid point
---
ManifoldValuedImageProcessing, R. Bergmann ~ 2015-11-25
"""
function midPoint(p::ManifoldPoint, q::ManifoldPoint)::ManifoldPoint
  return exp(p,0.5*log(p,q))
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
