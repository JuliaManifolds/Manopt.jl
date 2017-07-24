#
# Define a global problem and ist constructors
#
# ---
abstract type Problem end
"""
    GradientProblem <: Problem

specify a problem for gradient descent type algorithms, i.e. we require
a `costFuncion(x)`, a `gradient(x)`, a `stoppingCriterion(iter,x1,x2)`
and an initial value `xInit`. The `lineSearch` function to find the optimal
length of the gradient (e.g. by Amijo rule) is deactivated if not specified.
"""
type GradientProblem <: Problem
  costFunction::Function
  gradient::Function
  stoppingCriterion::Function
  initX::Union({MPoint,Array{MPoint,N}})
  lineSearch::Function
  useCache::Bool
  verbosity::Int
end
# set verbosity and cache to something standard when not present
GradientProblem(f,g,s,x,l) = GradientProblem(f,g,s,x,l,false,0)
# deactivate line search to keep the gradient vector as before, whten not specified
GradientProblem{MV::MTVector}(f,g,s,x) = GradientProblem(f,g,s,x,@(ξ::MV)::FloaT64 = 1.0)

type HessianProblem <: Problem
  costFunction::Function
  hessian::Function
  stoppingCriterion::Function
  initX::Union({MPoint,Array{MPoint,N}})
  useCache::Bool
  verbosity::Int
end
# set verbosity and cache to something standard when not present
HessianProblem(f,h,s,x) = GradientProblem(f,g,s,x,false,0)

type ProximalProblem <: Problem
  costFunction::Function
  proximalMaps::Array{Function,N}
  stoppingCriterion::Function
  initX::Union({MPoint,Array{MPoint,N}})
  useCache::Bool
  verbosity::Int
end
# set verbosity and cache to something standard when not present
ProximalProblem(f,h,s,x) = GradientProblem(f,g,s,x,false,0)

"""
    getGradient(p,x)

evaluate the gradient of a problem at x, where x is either a ManifoldPoint
or an array of ManifoldPoints
"""
function getGradient{P <: GradientProblem,MP <: ManifoldPoint}(p::P,x::Union(MP,Array{MP,N}))
  p.gradient(x)
end
"""
    evaluateStoppingCriterion(problem,iter,x1,x2)

evaluates the stoppinc criterion of problem with respect to the current
iteration and two (successive) values of the algorithm
"""
function evaluateStoppingCriterion{P<:Problem, MP <: ManifoldPoint,I<:Integer}(p::P,
                          iter::I,
                          x1::Union(MP,Array{MP,N}),
                          x2::Union(MP,Array{MP,N}))
  p.stoppingCriterion(iter,x1,x2)
end
"""
    getVerbosity(problem)

returns the verbosity of the problem.
"""
function getVerbosity{P<:Problem}(p::P)
  p.verbosity
end
