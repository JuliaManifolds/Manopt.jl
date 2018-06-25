#
# Manopt.jl – The manifold of symmetric positive definite matrices
# with affine metric.
#
#
# ---
# Manopt.jl - Ronny Bergmann - 2017-07-06

import Base.LinAlg: svd, norm, dot
import Base: exp, log, show

export SymmetricPositiveDefinite, SPDPoint, SPDTVector, shoq
export distance, dot, exp, norm, dot, manifoldDimension, parallelTransport

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
			# note that since p.value is struct the cache never expires.
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
		  SsqrtInv = diagm(1./Ssqrt)
	  	pSqrt = U*diagm(Ssqrt)*U.'
  		T = U*SsqrtInv*U.'*(t.*ξ.value)*U*SsqrtInv*U.';
		  svd2 = svd(T)
		  Se = diagm(exp.(svd2[2]))
		  Ue = svd2[1]
		  return SPDPoint(pSqrt*Ue*Se*Ue.'*pSqrt)
		#end
	end
end

function log(M::SymmetricPositiveDefinite,p::SPDPoint,q::SPDPoint, includeBase::Bool=false,cache::Bool=true)
	svd1 = decomp!(p)
	U = svd1[1]
	S = copy(svd1[2])
	Ssqrt = sqrt.(diag(S))
	SsqrtInv = diagm(1./Ssqrt)
	Ssqrt = diagm(Ssqrt)
  pSqrt = U*Ssqrt*U.'
	T = U*SsqrtInv*U.'*q.value*U*SsqrtInv*U.'
	svd2 = svd(T)
	Se = diagm(log.(svd2[2]))
	Ue = svd2[1]
	ξ = pSqrt*Ue*Se*Ue.'*pSqrt
	if includeBase
		return SPDTVector(ξ,p)
	else
		return SPDTVector(ξ)
	end
end

# shortest way to compute
# sqrt(trace(log(sqrt(p)^-1 q sqrt(p)^-1)^2)
# is to determine the generalized eigenvalus of XDV=Y we can
# compute from these sqrt(sum(abs(log(D)).^2))
function distance(M::SymmetricPositiveDefinite,p::SPDPoint,q::SPDPoint)::Float64
  return sqrt(sum(log.(abs.(eig(p.value,q.value)[1])).^2))
end
function dot(M::SymmetricPositiveDefinite, p::SPDPoint, ξ::SPDTVector, ν::SPDTVector, cache::Bool=true)
	svd1 = decomp!(p,cache)
	U = svd1[1]
	S = copy(svd1[2])
	SInv = diagm(1./diag(S))
	return trace(ξ.value*U*SInv*U.'*ν.value*U*Sing*U.')
end

function parallelTransport(M::SymmetricPositiveDefinite,p::SPDPoint,q::SPDPoint,ξ::SPDTVector,cache::Bool=true)::SPDTVector
	svd1 = decomp!(p,cache)
	U = svd1[1]
	S = copy(svd1[2])
	Ssqrt = sqrt.(diag(S))
	SsqrtInv = diagm(1./Ssqrt)
	Ssqrt = diagm(Ssqrt)
	pSqrt = U*Ssqrt*U.'
  pSqrtInv = U*SsqrtInv*U.'
	tξ = pSqrtInv*ξ.value*pSqrtInv
	tQ = pSqrtInv*q.value*pSqrtInv
	svd2 = svd(tQ)
	Se = diagm(log.(svd2[2]))
	Ue = svd2[1]
	tQ2 = Ue*Se*Ue.'
	eig1 = eig(0.5*tQ2)
	Sf = diagm(exp.(eig1[1]))
	Uf = eig1[2]
	if isnull(ξ.base)
		return SPDTVector(pSqrt*Uf*Sf*Uf.'*(0.5*(tξ+tξ.'))*Uf*Sf*Uf.'*pSqrt)
	else
		return SPDTVector(pSqrt*Uf*Sf*Uf.'*(0.5*(tξ+tξ.'))*Uf*Sf*Uf.'*pSqrt,q)
	end
end

show(io::IO, M::SymmetricPositiveDefinite) = print(io, "The Manifold $(M.name).")
show(io::IO, p::SPDPoint) = print(io, "SPD($(p.value))")
show(io::IO, ξ::SPDTVector) = isnull(ξ.base) ? print(io, "SPDT($(ξ.value))") : print(io, "SPDT_$(ξ.base.value)($(ξ.value))")
