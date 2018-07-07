#
#      Rn - The manifold of the n-dimensional (real valued) Euclidean space
#
# Manopt.jl, R. Bergmann, 2018-06-26
export Euclideam, RnPoint, RnTVector

import Base: exp, log, +, -, *, /, ==, show
# introduce new functions
export distance, exp, log, norm, dot, manifoldDimension, show

#
# Types
#
struct Euclidean <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Sphere(dimension::Int) = new("$dimension-dimensional Euclidean space",dimension,"R$dimension")
end

struct RnPoint <: MPoint
  value::Vector
  SnPoint(value::Vector) = new(value)
end

struct RnTVector <: TVector
  value::Vector
  SnTVector(value::Vector) = new(value)
end
#
# Traits
#
# - Rn is a MatrixManifold
@traitimpl IsMatrixM{Euclidean}
@traitimpl IsMatrixP{RnPoint}
@traitimpl IsMatrixV{RnTVector}

#
# Functions
#
function distance(M::Euclidean,p::RnPoint,q::RnPoint)::Number
  return norm(p.value-q.value)
end

function dot(M::Euclidean,ξ::RnTVector, ν::RnTVector)::Number
    return dot(ξ.value,ν.value)
end

function exp(M::Euclidean,p::RnPoint,ξ::RnTVector,t=1.0)::SnPoint
  return RnPoint(p.value + ξ.value)
end

function log(M::Euclidean,p::RnPoint,q::RnPoint)::RnTVector
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
show(io::IO, M::Euclidean) = print(io, "The Manifold $(M.name).");
show(io::IO, m::RnPoint) = print(io, "Rn($(m.value))");
show(io::IO, m::RnTVector) = print(io, "RnT($(m.value))");
