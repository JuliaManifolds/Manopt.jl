#
# Define a global problem and ist constructors
#
# ---
import Random: randperm
export getGradient, getCost, getHessian, getProximalMap, getProximalMaps
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
getCost(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(ErrorException("no costFunction found in $(typeof(p)) to evaluate for a $(typeof(x))."))
getGradient(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(ErrorException("no gradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))
getHessian(p::Pr,x::P,η::T) where {Pr <: Problem, P <: MPoint, T <: TVector} =
    throw(ErrorException("no Hessian found in $(typeof(p)) to evaluate for a $(typeof(x)) with tangent vector $(typeof(η))."))
getProximalMaps(p::Pr,λ,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(ErrorException("no proximal maps found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))
getProximalMap(p::Pr,λ,x::P,i) where {Pr <: Problem, P <: MPoint} =
    throw(ErrorException("no $(i)th proximal map found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))
getSubGradient(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
        throw(ErrorException("no sub gradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))

"""
    HessianProblem <: Problem
For now this is just a dummy problem to carry information about a Problem also providing a Hessian
"""
mutable struct HessianProblem{mT <: Manifold} <: Problem
    M::mT
    costFunction::Function
    Heassian::Function
end
@doc doc"""
    getHessian(p,x)
evakuate the Hessian of a [`HessianProblem`](@ref)` p` at the [`MPoint`](@ref) `x`.
"""
function getHessian(p::P,x::MP) where {P <: HessianProblem{M} where M <: Manifold, MP <: MPoint }
    return p.Hessian(x)
end
