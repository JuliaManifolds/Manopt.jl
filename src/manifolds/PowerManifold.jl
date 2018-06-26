#
#      Powermanifold – an array of points of _one_ manifold
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show

export PowerManifold, PowMPoint, PowMTVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export show

struct PowerManifold{M<:Manifold} <: Manifold
  name::String
  manifold::M
  dims::Array{Int,1}
  dimension::Int
  abbreviation::String
  PowerManifold{M}(mv::M,dims::Array{Int,1}) where {M<:Manifold} = new(string("A Power Manifold of ",mv.name,"."),
    mv,dims,prod(dims)*manifoldDimension(mv),string("Prod(",m.abbreviation,",",repr(dims),")") )
end
struct PowMPoint <: MPoint
  value::Array{T,N} where N where T<:MPoint
  PowMPoint(v::Array{T,N} where N where T<:MPoint) = new(v)
end

struct PowMTVector <: MTVector
  value::Array{T,N} where N where T <: MTVector
  base::Nullable{PowMPoint}
  PowMTVector(value::Array{T,N} where N where T <: MTVector) = new(value,Nullable{Pow-MPoint}())
  PowMTVector(value::Array{T,N} where N where T <: MTVector,base::PowMPoint) = new(value,base)
  PowMTVector(value::Array{T where T <: MTVector, N} where N,base::Nullable{PowMPoint}) = new(value,base)
end

function addNoise(M::PowerManifold, p::PowMPoint,σ)::PowMPoint
  return PowMPoint([addNoise.(M.manifold,p.value,σ)])
end


function distance(M::PowerManifold, p::PowMPoint,q::PowMPoint)::Float64
  return sqrt(sum( distance.(manifold,p.value,q.value).^2 ))
end

function dot(M::PowerManifold, ξ::PowMTVector, ν::PowMTVector)::Float64
  if checkBase(ξ,ν)
    return sum(dot.(M.manifold,ξ.value,ν.value))
  else
    throw(ErrorException("Can't compute dot product of two tangential vectors
		belonging to different tangential spaces."))
  end
end

function exp(M::PowerManifold, p::PowMPoint,ξ::PowMTVector,t::Number=1.0)::PowMPoint
  return PowMPoint( exp.(M.manifold,p.value,ξ.value) )
end

function log(M::PowerManifold, p::PowMPoint,q::PowMPoint,includeBase::Bool=false)::PowMTVector
  if includeBase
    return PowMTVector(log.(M.manifold,p.value,q.value),p)
  else
    return ProdMTVector(log.(M.manifold, p.value,q.value))
  end
end

function manifoldDimension(p::PowMPoint)::Int
  return prod(manifoldDimension.(p.value) )
end
function manifoldDimension(M::PowerManifold)::Int
  return prod(M.dims*manifoldDimension(M.manifold) )
end
function norm(M::PowerManifold, ξ::PowMTVector)::Float64
  return sqrt( dot(M,ξ,ξ) )
end
#
#
# Display functions for the structs
function show(io::IO, M::PowerManifold)
  print(io,string("The Power Manifold of ",repr(M.manifold),
    " of size ",repr(M.dims),".") )
end
function show(io::IO, p::PowMPoint)
    print(io,string("PowM[",join(repr.(p.value),", "),"]"))
end
function show(io::IO, ξ::PowMTVector)
  if !isnull(ξ.base)
    print(io,String("ProdMT_(",repr(ξ.base),")[", join(repr.(ξ.value),", "),"]"))
  else
    print(io,String("ProdMT[", join(repr.(ξ.value),", "),"]"))
  end
end
