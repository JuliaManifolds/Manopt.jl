#
# Define a global problem and ist constructors
#
# ---
export setDefaults, getGradient, getStepSize, evaluateStoppingCriterion
export Problem, DescentProblem, LineSearchProblem, HessianProblem, ProximalProblem
"""
    Problem

Specify properties (values) and related functions for computing
a certain optimization problem.
    FIELDS
    manifold          – a manifold
    costFunction      - a function F:M -> R
"""
mutable struct Problem{mT <: Manifold}
  M::mT
  costFunction::Function
end

"""
    DescentProblem <: Problem
      specify a problem for gradient descent type algorithms.

    ADDITIONAL FIELDS
      gradient          – the gradient of F
"""
mutable struct GradientProblem{M} <: Problem{M}
  gradient::Function
end
# TODO: divide into problems and options
mutable struct HessianProblem{M} <: Problem{M}
  hessian::Function
  stoppingCriterion::Function
  initX::T where T <: MPoint
  useCache::Bool
  verbosity::Int
end

# TODO: divide into problems and options
mutable struct ProximalProblem{M} <: Problem{M}
  costFunction::Function
  proximalMaps::Array{Function,N} where N
  stoppingCriterion::Function
  initX::T where T <: MPoint
  useCache::Bool
  verbosity::Int
end
# set verbosity and cache to something standard when not present
ProximalProblem(f,h,s,x) = ProximalProblem(f,g,s,x,false,0)
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
"""
    getStepsize(p,x,ξ)

evaluates the stopping criterion of problem with respect to the current
iteration iter, the descent direction ξ, and two (successive) iterates x1, x2
of the algorithm.
"""
function getStepsize{P <: GradientProblem{M} where M <: Manifold, MP <: MPoint, MT <: TVector}(p::P,
                              x::MP,gradF::MT,descentDir::MT,s::Float64)
  p.lineSearchProblem.initialStepsize = s
  return p.lineSearch(p.lineSearchProblem,x,gradF,descentDir)
end
function getStepsize{P <: GradientProblem{M} where M <: Manifold, MP <: MPoint, MT <: TVector}(p::P,
                              x::MP, gradF::MT, descentDir::MT)
  return getStepSize(p,x,gradF,descentDir,p.lineSearchProblem.initialStepsize)
end
function getStepsize{P <: GradientProblem{M} where M <: Manifold, MP <: MPoint, MT <: TVector}(p::P,
                              x::MP,gradF::MT,s::Float64)
  return getStepsize(p,x,gradF,-gradF,s)
end
function getStepsize{P <: GradientProblem{M} where M <: Manifold, MP <: MPoint, MT <: TVector}(p::P,
                              x::MP, gradF::MT)
  return p.lineSearch(pL,x,gradF,-gradF)
end
