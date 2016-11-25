"""
      Manifold -- a manifold defined via its data types:
  * A point on the manifold, ManifoldPoint
  * A point in an tangential space ManifoldTangentialPoint
"""
module Manifold
# extend existing methods:
import Base.LinAlg.norm, Base.LinAlg.dot, Base.exp, Base.log, Base.+,Base.-,Base.*
# introcude new types
export ManifoldPoint, ManifoldTangentialPoint
# introduce new functions
export manifoldDimension, distance
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
    proxTuple = proxTV(lambda,pointTuple)
  Compute the proximal map prox_f(x) for f(x) = dist(x,y) with parameter
  lambda
  # INPUT
    * lambda : a real value, parameter of the proximal map
    * pointTuple : a tuple of size 2 containing two ManifoldPoints
  # OUTPUT
    * proxTuple : resulting two-ManifoldPoint-Tuple of the proximal map
---
    ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function proxTV(lambda::Float64,pointTuple::Tuple{ManifoldPoint,ManifoldPoint})::Tuple{ManifoldPoint,ManifoldPoint}
  step = min(0.5, lambda/distance(pointTuple[1],pointTuple[2]))
  return (  exp(pointTuple[1], step*log(pointTuple[1],pointTuple[2])),
            exp(pointTuple[2], step*log(pointTuple[2],pointTuple[1])) )
end
#
# fallback functions for not yet implemented cases
function distance(p::ManifoldPoint,q::ManifoldPoint)::Float64
  error("The distance is not yet available for the manifold you\'re unsing")
end
function dot(xi::ManifoldTangentialPoint,nu::ManifoldTangentialPoint)::Float64
  error("The dot product og two tangential vectors is not yet available for the manifold you\'re unsing")
end
function exp(p::ManifoldPoint,xi::ManifoldTangentialPoint)::ManifoldPoint
  error("The exponential map is not yet available for the manifold you\'re unsing")
end
function log(p::ManifoldPoint,q::ManifoldPoint)::ManifoldTangentialPoint
  error("This logarithmic map is not yet available for the manifold you\'re unsing")
end
function manifoldDimension(p::ManifoldPoint)::Integer
  error("The Dimension of the Manifold is not yet available for the manifold you\'re unsing")
end
function norm(xi::ManifoldTangentialPoint)::Float64
  error("The norm of a tangential vector is not yet vailable for the manifold you\'re unsing")
end
end #module Manifold
