#
# Gradient Plan
#
export GradientProblem
export getGradient, getCost
export GradientDescentOptions
export evaluateStoppingCriterion
#
# Problem
@doc doc"""
    GradientProblem <: Problem
specify a problem for gradient based algorithms.

# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `gradient`     : the gradient $\nabla F\colon\mathcal M
  \to \mathcal T\mathcal M$ of the cost function $F$

# See also
[`steepestDescent`](@ref), [`conjugateGradientDescent`](@ref),
[`GradientDescentOptions`](@ref), [`ConjugateGradientOptions`](@ref)

# """
mutable struct GradientProblem{mT <: Manifold} <: Problem
  M::mT
  costFunction::Function
  gradient::Function
end
"""
    getGradient(p,x)

evaluate the gradient of a [`GradientProblem`](@ref)`p` at the [`MPoint`](@ref)` x`.
"""
function getGradient(p::P,x::MP) where {P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}
  return p.gradient(x)
end
"""
    getCost(p,x)

evaluate the cost function `F` stored within a [`GradientProblem`](@ref) at the [`MPoint`](@ref)` x`.
"""
function getCost(p::P,x::MP) where {P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}
  return p.costFunction(x)
end

#
# Options
"""
    GradientDescentOptions{P,L} <: Options
Describes a Gradient based descent algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x0` : an [`MPoint`](@ref) as starting point
* `stoppingCriterion` : a function s,r = @(o,iter,ξ,x,xnew) returning a stop
    indicator and a reason based on an iteration number, the gradient and the last and
    current iterates
* `retraction` : (exp) the rectraction to use
* `lineSearch` : a function performing the lineSearch, returning a step size
* `lineSearchOptions` : options the linesearch is called with.

# See also
[`steepestDescent`](@ref)
"""
mutable struct GradientDescentOptions <: Options
    x0::P where {P <: MPoint}
    stoppingCriterion::Function
    retraction::Function
    lineSearch::Function
    lineSearchOptions::L where {L <: LineSearchOptions}
    # fallback do exp
    GradientDescentOptions(x0::P where {P<:MPoint},sC::Function,lS::Function,lSO::L where {L <: LineSearchOptions},retr::Function=exp) = new(x0,sC,retr,lS,lSO)
end
function evaluateStoppingCriterion(o::O,iter::I,ξ::MT, x::P, xnew::P) where {O<:GradientDescentOptions, P <: MPoint, MT <: TVector, I<:Integer}
  o.stoppingCriterion(iter,ξ,x,xnew)
end
"""
    getStepsize(p,o)

evaluates the stopping criterion of problem with respect to the current
iteration iter, the descent direction ξ, and two (successive) iterates x1, x2
of the algorithm.
"""
function getStepsize(p::P,o::O,vars...) where {P <: GradientProblem{M} where M <: Manifold, O <: GradientDescentOptions}
    return getStepsize(p,o.lineSearchOptions,o.lineSearch,vars...)
end
# for gradientLineSearch: Update initS and x and start
function getStepsize(p::gP,o::O, f::Function, x::P, s::Float64) where {gP <: GradientProblem{M} where M <: Manifold, O <: ArmijoLineSearchOptions, P <: MPoint}
    o.initialStepsize = s
    o.x = x;
    return getStepsize(p,o,f)
end
# for generalLineSearch - update DescentDir and continue
function getStepsize(p::gP,o::O, f::Function, x::P, s::Float64) where {gP <: GradientProblem{M} where M <: Manifold, P <: MPoint, O <: LineSearchOptions}
  o.initialStepsize = s;
  o.x = x;
  updateDescentDir!(o,x)
  return getStepsize(p,o,f)
end
# (finally) call lineSearchProcedure
function getStepsize(p::gP, o::O, f::Function) where {gP <: GradientProblem{M} where M <: Manifold, O <: Union{ArmijoLineSearchOptions,LineSearchOptions}}
  return f(p,o)
end
