#
# Define a global problem and ist constructors
#
# ---

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
    get_cost(p,x)

evaluate the cost function `F` stored within a [`Problem`](@ref) at the point `x`.
"""
function get_cost(p::P,x) where {P <: Problem}
  return p.cost(x)
end
getGradient(p::Pr,x) where {Pr <: Problem} =
    throw(ErrorException("no gradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))
getProximalMap(p::Pr,λ,x,i) where {Pr <: Problem} =
    throw(ErrorException("No proximal map No. $(i) found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))
get_subgradient(p::Pr,x) where {Pr <: Problem} =
    throw(ErrorException("no subgradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))
getHessian(p::Pr,x,ξ) where {Pr <: Problem} =
    throw(ErrorException("no hessian found in $(typeof(p)) to evaluate at point $(typeof(x)) and tangent vector $(typeof(ξ))."))