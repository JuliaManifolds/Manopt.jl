"""
      Manifold -- a manifold defined via its data types:
  * A point on the manifold, ManifoldPoint
  * A point in an tangential space ManifoldTangentialPoint
"""
module Manifold

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
# fallback functions for not yet implemented cases
function distance(p::ManifoldPoint,q::ManifoldPoint)
  error("The exponential map is not yet available for the manifold you\'re unsing")
end
function exp(p::ManifoldPoint,xi::ManifoldTangentialPoint)
  error("The exponential map is not yet available for the manifold you\'re unsing")
end
function log(p::ManifoldPoint,q::ManifoldPoint)
  error("This logarithmic map is not yet available for the manifold you\'re unsing")
end
function manifoldDimension(p::ManifoldPoint)
  error("The Dimension of the Manifold is not yet available for the manifold you\'re unsing")
end
end #module Manifold
