#
#      Sn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
export Sphere, SnPoint, SnTVector

struct Sphere <: MatrixManifold
  name::String
  dimension::Int
  abbreviation::String
  Sphere(dimension::Int) = new("$dimension-Sphere",dimension,"S$dimension")
end

struct SnPoint <: MMPoint
  value::Vector
  SnPoint(value::Vector) = new(value)
end

struct SnTVector <: MMTVector
  value::Vector
  base::Nullable{SnPoint}
  SnTVector(value::Vector) = new(value,Nullable{SnPoint}())
  SnTVector(value::Vector,base::SnPoint) = new(value,base)
  SnTVector(value::Vector,base::Nullable{SnPoint}) = new(value,base)
end

function distance(M::Sphere,p::SnPoint,q::SnPoint)::Number
  return acos(dot(p.value,q.value))
end

function dot(M::Sphere,ξ::SnTVector, ν::SnTVector)::Number
  if checkBase(ξ,ν)
  	return dot(ξ.value,ν.value)
  end
end

function exp(M::Sphere,p::SnPoint,ξ::SnTVector,t=1.0)::SnPoint
	if checkBase(p,ξ)
  	len = norm(ξ.value)
  	if len < eps(Float64)
    	return p
  	else
    	return SnPoint(cos(t*len)*p.value + sin(t*len)/len*ξ.value)
  	end
	end
end

function log(M::Sphere,p::SnPoint,q::SnPoint,includeBase::Bool=false)::SnTVector
  scp = dot(p.value,q.value)
  ξvalue = q.value-scp*p.value
  ξvnorm = norm(ξvalue)
  if (ξvnorm > eps(Float64))
    value = ξvalue*acos(scp)/ξvnorm;
  else
    value = zeros(p.value)
  end
  if includeBase
    return SnTVector(value,p)
  else
    return SnTVector(value)
  end
end
function manifoldDimension(p::SnPoint)::Integer
  return length(p.value)-1
end
function manifoldDimension(M::Sphere)::Integer
  return M.dimension
end
function norm(M::Sphere,ξ::SnTVector)::Number
  return norm(ξ.value)
end
#
#
# --- Display functions for the objects/types
function show(io::IO, M::Sphere)
    print(io, "The Manifold $(M.name).")
  end
function show(io::IO, m::SnPoint)
    print(io, "Sn($(m.value))")
end
function show(io::IO, m::SnTVector)
  if !isnull(m.base)
    print(io, "SnT_$(m.base.value)($(m.value))")
  else
    print(io, "SnT($(m.value))")
  end
end
