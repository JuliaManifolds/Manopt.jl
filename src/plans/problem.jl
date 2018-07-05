#
# Define a global problem and ist constructors
#
# ---
export getGradient
export Problem, GradientProblem #, HessianProblem, ProximalProblem

"""
    Problem

Specify properties (values) and related functions for computing
a certain optimization problem.
    FIELDS
    manifold          – a manifold
    costFunction      - a function F:M -> R
"""
abstract type Problem end

# """
#     DescentProblem <: Problem
#       specify a problem for gradient descent type algorithms.
#
#     ADDITIONAL FIELDS
#       gradient          – the gradient of F
# """
mutable struct GradientProblem{mT <: Manifold} <: Problem
  M::mT
  costFunction::Function
  gradient::Function
end

# TODO: divide into problems and options
# mutable struct HessianProblem{M} <: Problem{M}
#   hessian::Function
#   stoppingCriterion::Function
#   initX::T where T <: MPoint
#   useCache::Bool
#   verbosity::Int
# end

# TODO: divide into problems and options
# mutable struct ProximalProblem{M} <: Problem{M}
#   proximalMaps::Array{Function,N} where N
#   stoppingCriterion::Function
#   initX::T where T <: MPoint
#   useCache::Bool
#   verbosity::Int
# end
#
# Access functions for Gradient problem.
# ---
"""
    getGradient(p,x)

evaluate the gradient of a problem at x, where x is either a MPoint
or an array of MPoints
"""
function getGradient{P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}(p::P,x::MP)
  return p.gradient(x)
end
# TODO for a subtype GradientProblem might store gradients internally
