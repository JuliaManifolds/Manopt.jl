#
#      Sn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
export SnPoint, SnTangentialPoint
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
  base::Nullable{SnPoint}
  SnTangentialPoint(value::Vector) = new(value,Nullable{SnPoint}())
  SnTangentialPoint(value::Vector,base::SnPoint) = new(value,base)
  SnTangentialPoint(value::Vector,base::Nullable{SnPoint}) = new(value,base)
end

function distance(p::SnPoint,q::SnPoint)::Number
  return acos(dot(p.value,q.value))
end

function dot(ξ::SnTangentialPoint, ν::SnTangentialPoint)::Number
  if sameBase(ξ,ν)
    return dot(ξ.value,ν.value)
  else
    throw(ErrorException("Can't compute dot product of two tangential vectors belonging to
      different tangential spaces."))
  end
end

function exp(p::SnPoint,ξ::SnTangentialPoint,t=1.0)::SnPoint
  len = norm(ξ.value)
  if len < eps(Float64)
    return p
  else
    return SnPoint(cos(t*len)*p.value + sin(t*len)/len*ξ.value)
  end
end

function log(p::SnPoint,q::SnPoint,includeBase=false)::SnTangentialPoint
  scp = dot(p.value,q.value)
  ξvalue = q.value-scp*p.value
  ξvnorm = norm(ξvalue)
  if (ξvnorm > eps(Float64))
    value = ξvalue*acos(scp)/ξvnorm;
  else
    value = zeros(p.value)
  end
  if includeBase
    return SnTangentialPoint(value,p)
  else
    return SnTangentialPoint(value)
  end
end
function manifoldDimension(p::SnPoint)::Integer
  return length(p.value)-1
end
function norm(ξ::SnTangentialPoint)::Number
  return norm(ξ.value)
end

function show(io::IO, m::SnPoint)
    print(io, "Sn($(m.value))")
end
function show(io::IO, m::SnTangentialPoint)
  if !isnull(m.base)
    print(io, "SnT_$(m.base.value)($(m.value))")
  else
    print(io, "SnT($(m.value))")
  end
end
