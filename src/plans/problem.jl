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
function getGradient(p::Problem,x)
    throw(ErrorException(
        "no gradient found in $(typeof(p)) to evaluate for a $(typeof(x))."
    ))
end
function getProximalMap(p::Problem,λ,x,i)
    throw(ErrorException("No proximal map No. $(i) found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))
end
function get_subgradient(p::Problem,x)
    throw(ErrorException("no subgradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))
end
function getHessian(p::Problem,x,ξ)
    throw(ErrorException("no hessian found in $(typeof(p)) to evaluate at point $(typeof(x)) and tangent vector $(typeof(ξ))."))
end