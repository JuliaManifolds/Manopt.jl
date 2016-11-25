"""
      Sn - The manifold of the n-dimensional sphere
  Point is a Point on the n-dimensional sphere.
"""
module Sn
using Manifold: ManifoldPoint, ManifoldTangentialPoint

export SnPoint, SnTangentialPoint, exp, log, manifoldDimension

#
# TODO: It would be nice to have a fixed dimension here Sn here, however
#   they need N+1-dimensional vectors
#

immutable SnPoint <: ManifoldPoint
  value::Vector
  SnPoint(value::Vector) = new(value)
end

immutable SnTangentialPoint <: ManifoldTangentialPoint
  value::Vector
  base::Vector
  SnTangentialPoint(value::Vector) = new(value)
  SnTangentialPoint(value::Vector,base::Vector) = new(value,base)
end

function exp(p::SnPoint,xi::SnTangentialPoint,t=1.0)::SnPoint
  len = norm(xi.value)
  if len < eps(Float64)
    return p
  else
    return SnPoint(cos(t*len)*p.value + sin(t*len)/len*xi.value)
  end
end

function log(p::SnPoint,q::SnPoint)::SnTangentialPoint
  scp = dot(p.value,q.value)
  xivalue = q.value-scp*p.value
  xivnorm = norm(xivalue)
  if (xivnorm > eps(Float64))
    return SnTangentialPoint(xivalue*acos(scp)/xivnorm)
  else
    return SnTangentialPoint(zeros(p.value))
  end
end
"""
  manifoldDimension - dimension of the manifold this point belongs to
  # Input
    p : an SnPoint
  # Output
    d : dimension of the manifold (sphere) this point belongs to
"""
function manifoldDimension(p::SnPoint)::Integer
  return length(p.value)-1
end

end  # module Sn
