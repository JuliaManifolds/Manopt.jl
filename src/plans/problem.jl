#
# Define a global problem and ist constructors
#
# ---
import Random: randperm
export getGradient, getCost, getProximalMap
export Problem, HessianProblem

"""
    Problem
Specify properties (values) and related functions for computing
a certain optimization problem.
"""
abstract type Problem end
#
# 1) Function defaults / Fallbacks
#
"""
    getCost(p,x)

evaluate the cost function `F` stored within a [`Problem`](@ref) at the [`MPoint`](@ref) `x`.
"""
function getCost(p::P,x::MP) where {P <: Problem, MP <: MPoint}
  return p.costFunction(x)
end
getGradient(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(ErrorException("no gradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))
getProximalMap(p::Pr,λ,x::P,i) where {Pr <: Problem, P <: MPoint} =
    throw(ErrorException("No proximal map No. $(i) found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))
getSubGradient(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
        throw(ErrorException("no subgradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))
getHessian(p::Pr,x::P,ξ::T) where {Pr <: Problem, P <: MPoint, T <: TVector} =
    throw(ErrorException("no hessian found in $(typeof(p)) to evaluate at point $(typeof(x)) and tangent vector $(typeof(ξ))."))