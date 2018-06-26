#
#      Sn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
import Base.LinAlg: norm, dot
import Base: exp, log, show

export Sphere, SnPoint, SnTVector,show
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport


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

function dot(M::Sphere, p::SnPoint, ξ::SnTVector, ν::SnTVector)::Number
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

manifoldDimension(p::SnPoint)::Integer = length(p.value)-1

manifoldDimension(M::Sphere)::Integer = M.dimension

norm(M::Sphere, ξ::SnTVector)::Number = norm(ξ.value)

function parallelTransport(M::Sphere, p::SnPoint, q::SnPoint, ξ::SnTVector)
	if checkBase(p,ξ)
		ν::SnTVector = log(M,p,q)
		νL::Float64 = norm(M,ν)
		if νL > 0
			ν = ν/νL
    	if isnull(ξ.base)
				# remove p-coponent, add q-component (which is also by substraction - work on value to not have basechecks)
				return SnTVector(ξ.value - dot(M,ν,ξ)*(ν.value + log(M,q,p).value/νL));
			else # add base
				return SnTVector(ξ.value - dot(M,ν,ξ)*(ν + log(M,q,p).value/νL),q);
			end
		else
			# if length of ν is 0, we have p=q and hence ξ is unchanged
			return ξ
		end
	end
end
#
#
# --- Display functions for the objects/types
show(io::IO, M::Sphere) = print(io, "The Manifold $(M.name).")
show(io::IO, p::SnPoint) = print(io, "Sn($(p.value))")
show(io::IO, ξ::SnTVector) = isnull(ξ.base) ? print(io, "SnT($(ξ.value))") : print(io, "SnT_$(ξ.base.value)($(ξ.value))")
