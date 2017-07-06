#
#      Rn - The manifold of the n-dimensional (real valued) Euclidean space
#
export Euclideam, RnPoint, RnTVector

import Base: exp, log, +, -, *, /, ==, show
# introduce new functions
export distance, exp, log, norm, dot, manifoldDimension, show

struct Euclidean <: MatrixManifold
  name::String
  dimension::Int
  abbreviation::String
  Sphere(dimension::Int) = new("$dimension-dimensional Euclidean space",dimension,"R$dimension")
end

struct RnPoint <: MMPoint
  value::Vector
  SnPoint(value::Vector) = new(value)
end

struct RnTVector <: MMTVector
  value::Vector
  base::Nullable{RnPoint}
  SnTVector(value::Vector) = new(value,Nullable{RnPoint}())
  SnTVector(value::Vector,base::RnPoint) = new(value,base)
  SnTVector(value::Vector,base::Nullable{RnPoint}) = new(value,base)
end

function distance(M::Euclidean,p::RnPoint,q::RnPoint)::Number
  return norm(p.value-q.value)
end

function dot(M::Euclidean,ξ::RnTVector, ν::RnTVector)::Number
  if checkBase(ξ,ν)
    return dot(ξ.value,ν.value)
  else
    throw(ErrorException("Can't compute dot product of two tangential vectors belonging to
      different tangential spaces."))
  end
end

function exp(M::Euclidean,p::RnPoint,ξ::RnTVector,t=1.0)::SnPoint
  return RnPoint(p.value + ξ.value)
end

function log(M::Euclidean,p::RnPoint,q::RnPoint,includeBase=false)::RnTVector
	return RnTVector(p.value - q.value)
end
function manifoldDimension(p::RnPoint)::Integer
  return length(p.value)
end
function manifoldDimension(M::Euclidean)::Integer
  return M.dimension
end
function norm(M::Euclidean,ξ::RnTVector)::Number
  return norm(ξ.value)
end
#
#
# --- Display functions for the objects/types
function show(io::IO, M::Euclidean)
    print(io, "The Manifold $(M.name).")
  end
function show(io::IO, m::RnPoint)
    print(io, "Rn($(m.value))")
end
function show(io::IO, m::RnTVector)
  if !isnull(m.base)
    print(io, "RnT_$(m.base.value)($(m.value))")
  else
    print(io, "RnT($(m.value))")
  end
end
