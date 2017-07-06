#
# Manopt.jl – The manifold of symmetric positive definite matrices
# with affine metric.
#
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import Base.LinAlg: svd, norm, dot
import Base: exp, log, show

export SymmetricPositiveDefinite, SPDPoint, SPDTVector
export distance, exp, log, norm, dot, manifoldDimension, show

struct SymmetricPositiveDefinite <: MatrixManifold
  name::String
  dimension::Int
  abbreviation::String
  SymmetricPositiveDefinite(dimension::Int) = new("$dimension-by-$dimension symmetric positive definite matrices",(dimension*(dimension+1)/2),"SPD($dimension) affine")
end

struct SPDPoint <: MMPoint
	value::Matrix{Float64}
	decomposition::Array{Matrix{Float64},1}
	SPDPoint(v::Matrix{Float64}) = new(v,[])
	SPDPoint(v::Matrix{Float64}, decomp::Array{Matrix{Float64},1}) = new(v,decomp)
	SPDPoint(p::SPDPoint) = new(copy(p.value),copy(p.decomp))
end

struct SPDTVector <: MMTVector
	value::Matrix{Float64}
  base::Nullable{SPDPoint}
  SPDTVector(value::Matrix{Float64}) = new(value,Nullable{SPDPoint}())
  SPDTVector(value::Matrix{Float64},base::SPDPoint) = new(value,base)
  SPDTVector(value::Matrix{Float64},base::Nullable{SPDPoint}) = new(value,base)
end
"""
	decomp(p::SPDPoint,cache=true)
"""
function decomp!(p::SPDPoint,cache::Bool=true)::Array{Matrix{Float64},1}
	svdResult = svd(p.value)
	if cache # cache result?
		if length(p.decomposition)==0
			# not cached before?
			push!(p.decomposition,svdResult[1])
			push!(p.decomposition,diagm(svdResult[2]))
			push!(p.decomposition,svdResult[3])
			# note that since p.value is immutable the cache never expires.
		end
		return p.decomposition
	end
end
function exp(M::SymmetricPositiveDefinite,p::SPDPoint,ξ::SPDTVector,t::Float64=1.0, cache::Bool=true)
	if checkBase(p,ξ)
	  #if norm(M,p,ξ) == 0
	  #	return p
	  #else
		  svd1 = decomp!(p,cache)
		  U = svd1[1]
		  S = copy(svd1[2])
		  Ssqrt = sqrt.(diag(S))
		  SsqrtInv = 1./Ssqrt
	  	pSqrt = U*diagm(Ssqrt)*U.'
  		T = U*S*U.'*(t.*ξ.value)*U*S*U.';
		  svd2 = svd(T)
		  Se = diagm(exp.(svd2[2]))
		  Ue = svd2[1]
		  return SPDPoint(pSqrt*Ue*Se*Ue.'*pSqrt)
		#end
	end
end

function show(io::IO, M::SymmetricPositiveDefinite)
    print(io, "The Manifold $(M.name).")
  end
function show(io::IO, m::SPDPoint)
    print(io, "SPD($(m.value))")
end
function show(io::IO, m::SPDTVector)
  if !isnull(m.base)
    print(io, "SPDT_$(m.base.value)($(m.value))")
  else
    print(io, "SPDT($(m.value))")
  end
end
