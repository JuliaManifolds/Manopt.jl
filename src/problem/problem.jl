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
"""
abstract type Problem end
#
# Types of problems for corresponding algorithms
# ---
"""
    LineSearchProblem <: Problem

Collects data on the line search problem
"""
mutable struct LineSearchProblem{M <: Manifold} <: Problem
    manifold::M
    costFunction::Function
    initialStepsize::Float64
    rho::Float64
    c::Float64
end
# Without step size
LineSearchProblem{M <: Manifold}(Mf::M,F::Function,rho::Float64,c::Float64) = LineSearchProblem(Mf,F,1.0,rho,c)
LineSearchProblem{M <: Manifold}(Mf::M,F::Function) = setDefaults(LineSearchProblem(Mf,F,0.0,0.0,0.0))

function setDefaults(p::LineSearchProblem)::LineSearchProblem
  p.initialStepsize = 1.0
  p.rho = 0.5
  p.c = 0.001
  return p
end

"""
    DescentProblem <: Problem

specify a problem for gradient descent type algorithms, i.e. we require
a `costFuncion(x)`, a `gradient(x)`, a `stoppingCriterion(iter,x1,x2)`
and an initial value `xInit`. The `lineSearch` function to find the optimal
length of the gradient (e.g. by Amijo rule) is deactivated if not specified.
"""
mutable struct DescentProblem <: Problem
  manifold::Manifold
  costFunction::Function
  gradient::Function
  stoppingCriterion::Function
  initX::T where T <: MPoint
  lineSearch::Function
  lineSearchProblem::LineSearchProblem
  useCache::Bool
  verbosity::Int
  # Optional: debug.
  debugFunction::Nullable{Function}
  debugSettings::Nullable{Dict{String,<:Any}}
end
# set verbosity and cache to something standard when not present
DescentProblem(M,f,g,s,x,l,lp,u,v) = DescentProblem(M,f,g,s,x,l,lp,u,v,Nullable{Function}(),Nullable{Dict{String,Any}}())

DescentProblem(M,f,g,s,x,l,lp) = DescentProblem(M,f,g,s,x,l,lp,false,0)
# deactivate line search to keep the gradient vector as before, whten not specified
DescentProblem(M,f,g,s,x) = DescentProblem(M,f,g,s,x, ξ -> 1.0)


mutable struct HessianProblem <: Problem
  costFunction::Function
  hessian::Function
  stoppingCriterion::Function
  initX::T where T <: MPoint
  useCache::Bool
  verbosity::Int
end
# set verbosity and cache to something standard when not present
HessianProblem(f,h,s,x) = GradientProblem(f,g,s,x,false,0)

mutable struct ProximalProblem <: Problem
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
function getGradient{P <: DescentProblem, MP <: MPoint}(p::P,x::MP)
  return p.gradient(x)
end
"""
    getStepsize(p,x,ξ)

evaluates the stopping criterion of problem with respect to the current
iteration iter, the descent direction ξ, and two (successive) iterates x1, x2
of the algorithm.
"""
function getStepsize{P <: DescentProblem, MP <: MPoint, MT <: MTVector}(p::P,
                              x::MP,gradF::MT,descentDir::MT,s::Float64)
  p.lineSearchProblem.initialStepsize = s
  return p.lineSearch(p.lineSearchProblem,x,gradF,descentDir)
end
function getStepsize{P<:DescentProblem, MP <: MPoint, MT <: MTVector}(p::P,
                              x::MP, gradF::MT, descentDir::MT)
  return getStepSize(p,x,gradF,descentDir,p.lineSearchProblem.initialStepsize)
end
function getStepsize{P<:DescentProblem, MP <: MPoint, MT <: MTVector}(p::P,
                              x::MP,gradF::MT,s::Float64)
  return getStepsize(p,x,gradF,-gradF,s)
end
function getStepsize{P<:DescentProblem, MP <: MPoint, MT <: MTVector}(p::P,
                              x::MP, gradF::MT)
  return p.lineSearch(pL,x,gradF,-gradF)
end

"""
    evaluateStoppingCriterion(problem,iter,x1,x2)

evaluates the stoppinc criterion of problem with respect to the current
iteration and two (successive) values of the algorithm
"""
function evaluateStoppingCriterion{P<:Problem, MP <: MPoint,I<:Integer}(p::P,
                          iter::I,x1::MP,x2::MP)
  p.stoppingCriterion(iter,ξ,x1,x2)
end
"""
    evaluateStoppingCriterion(problem,iter,x1,x2)

evaluates the stopping criterion of problem with respect to the current
iteration iter, the descent direction ξ, and two (successive) iterates x1, x2
of the algorithm.
"""
function evaluateStoppingCriterion{P<:DescentProblem, MP <: MPoint, MT <: MTVector, I<:Integer}(p::P,
                          iter::I,ξ::MT, x1::MP, x2::MP)
  p.stoppingCriterion(iter,ξ,x1,x2)
end
"""
    getVerbosity(problem)

returns the verbosity of the problem.
"""
function getVerbosity{P<:Problem}(p::P)
  p.verbosity
end
