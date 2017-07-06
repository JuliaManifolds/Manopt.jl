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
  Sphere(dimension::Int) = new("The manifold of $dimension-by$dimension symmetric positive definite manifold",dimension,"SPD($dimension) affine")
end

struct SPDPoint <: MMPoint
	value::Array{Float64,2}
	decomposition::Array{Array{Float64,2},1}
	SPDPoint(v::Array{Float64,2}) = new(v,[])
	SPDPoint(v::Array{Float64,2}, decomp::Array{Array{Float64,2},1}) = new(v,decomp)
	SPDPoint(p::SPDPoint) = new(copy(p.value),copy(p.decomp))
end

struct SPDTVector <: MMTVector
	value::Array{Float64,2}
end
"""
	decomp(p::SPDPoint,cache=true)
"""
function decomp!(p::SPDPoint,cache::Bool=true)::Array{Array{Float64,2},1}
	SvdResult = svd(p.value)
	if cache # cache result?
		if length(p.decomposition)==0
			# not cached before?
			append(p.decomposition,svdResult.U)
			append(p.decomposition,svdResult.S)
			append(p.decomposition,svdResult.Vt)
			# note that since p.value is immutable the cache never expires.
		end
		return p
	end
end
function exp(M::SymmetricPositiveDefinite,p::SPDPoint,ξ::SPDTVector,t::Float64=1.0 cache::Bool=true)
	if norm(M,p,ξ) == 0
		return p
	else
		svd = decomp!(p,cache)
		U = svd[1]
		S = svd[2].copy()
		Ssqrt = sqrt.(S)
		SsqrtInv = 1./Ssqrt
		pSqrt = U*diagm(Sqrt)*U.'
		T = U*diagm(S)*U.'*(t.*ξ)*U*S*U.';
		svd2 = svd(T)
		Se = exp.(svd2.S)
		return pSqrt*svd2.U*Se*svd2.U.'*pSqrt;
end
