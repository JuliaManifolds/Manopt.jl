#
#      Sn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
export S1Point, S1TVector, S1Manifold
export symRem
#
# TODO: It would be nice to have an arbitrary real type here
#
struct Sphere <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Sphere() = new("Sphere",1,"S1")
end
immutable S1Point <: MPoint
  value::Float64
  S1Point(value::Float64) = new(value)
end

immutable S1TVector <: MTVector
  value::Float64
  base::Nullable{S1Point}
  S1TVector(value::Float64) = new(value,Nullable{S1Point}())
  S1TVector(value::Float64,base::S1Point) = new(value,base)
  S1TVector(value::Float64,base::Nullable{S1Point}) = new(value,base)
end

function addNoise(p::S1Point,σ::Real)::S1Point
  return S1Point(mod(p.value-pi+σ*randn(),2*pi)+pi)
end


function distance(p::S1Point,q::S1Point)::Float64
  return abs( symRem(p.value-q.value) )
end

function dot(ξ::S1TVector, ν::S1TVector)::Float64
  if sameBase(ξ,ν)
    return ξ.value*ν.value
  else
    throw(ErrorException("Can't compute dot product of two tangential vectors belonging to
      different tangential spaces."))
  end
end

function exp(p::S1Point,ξ::S1TVector,t::Number=1.0)::S1Point
  return S1Point(symRem(p.value+t*ξ.value))
end

function log(p::S1Point,q::S1Point,includeBase=false)::S1TVector
  if includeBase
    return S1TVector(symRem(q.value-p.value),p)
  else
    return S1TVector(symRem(q.value-p.value))
  end
end

function manifoldDimension(p::S1Point)::Int64
  return 1
end

function norm(ξ::S1TVector)::Float64
  return abs(ξ.value)
end

function show(io::IO, m::S1Point)
    print(io, "S1($(m.value))")
end
function show(io::IO, ξ::S1TVector)
  if !isnull(ξ.base)
    print(io, "S1T_$(ξ.base.value)($(ξ.value))")
  else
    print(io, "S1T($(ξ.value))")
  end
end
#
#
# little Helpers
"""
    symRem(x,y,T=pi)
  symmetric remainder with respect to the interall 2*T
"""
function symRem(x::Float64, T::Float64=Float64(pi))::Float64
  return (x+T)%(2*T) - T
end
