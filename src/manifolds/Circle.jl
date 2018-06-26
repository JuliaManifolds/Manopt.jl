#
#      S1 - The manifold of the 1-dimensional sphere represented by angles
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show

export Circle, S1Point, S1TVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export show

export symRem
struct Circle <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Circle() = new("1-Sphere as angles",1,"S1")
end
struct S1Point <: MPoint
  value::Float64
  S1Point(value::Float64) = new(value)
end

struct S1TVector <: MTVector
  value::Float64
  base::Nullable{S1Point}
  S1TVector(value::Float64) = new(value,Nullable{S1Point}())
  S1TVector(value::Float64,base::S1Point) = new(value,base)
  S1TVector(value::Float64,base::Nullable{S1Point}) = new(value,base)
end

function addNoise(M::Circle, p::S1Point,σ::Real)::S1Point
  return S1Point(mod(p.value-pi+σ*randn(),2*pi)+pi)
end


function distance(M::Circle, p::S1Point,q::S1Point)::Float64
  return abs( symRem(p.value-q.value) )
end

function dot(M::Circle, p::S1Point, ξ::S1TVector, ν::S1TVector)::Float64
  if checkBase(ξ,ν)
    return ξ.value*ν.value
  else
    throw(ErrorException("Can't compute dot product of two tangential vectors
		belonging to different tangential spaces."))
  end
end

function exp(M::Circle, p::S1Point,ξ::S1TVector,t::Number=1.0)::S1Point
  return S1Point(symRem(p.value+t*ξ.value))
end

function log(M::Circle, p::S1Point,q::S1Point,includeBase::Bool=false)::S1TVector
  if includeBase
    return S1TVector(symRem(q.value-p.value),p)
  else
    return S1TVector(symRem(q.value-p.value))
  end
end

function manifoldDimension(p::S1Point)::Int
  return 1
end
function manifoldDimension(M::Circle)::Int
  return 1
end
function norm(M::Circle, ξ::S1TVector)::Float64
  return abs(ξ.value)
end
#
#
# Display functions for the structs
function show(io::IO, M::Circle)
  print(io, "The Manifold S1 consisting of angles")
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
