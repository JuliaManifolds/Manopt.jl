#
# Gradient Plan
#
export GradientProblem, GradientDescentOptions
export getGradient, getCost, getStepsize, getInitialStepsize

export DirectionUpdateOptions, HessianDirectionUpdateOptions
#
# Problem
#
@doc doc"""
    GradientProblem <: Problem
specify a problem for gradient based algorithms.

# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `gradient`     : the gradient $\nabla F\colon\mathcal M
  \to \mathcal T\mathcal M$ of the cost function $F$

# See also
[`steepestDescent`](@ref)
[`GradientDescentOptions`](@ref)

# """
mutable struct GradientProblem{mT <: Manifold} <: Problem
  M::mT
  costFunction::Function
  gradient::Function
end

#
# Options for subproblems
#

abstract type DirectionUpdateOptions end
"""
    SimpleDirectionUpdateOptions <: DirectionUpdateOptions
A simple update rule requires no information
"""
struct SimpleDirectionUpdateOptions <: DirectionUpdateOptions
end
"""
    HessianDirectionUpdateOptions
An update rule that keeps information about the Hessian or optains these
informations from the corresponding [`Options`](@ref)
"""
struct HessianDirectionUpdateOptions <: DirectionUpdateOptions
end

"""
    getGradient(p,x)

evaluate the gradient of a [`GradientProblem`](@ref)`p` at the [`MPoint`](@ref) `x`.
"""
function getGradient(p::P,x::MP) where {P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}
  return p.gradient(x)
end
"""
    getCost(p,x)

evaluate the cost function `F` stored within a [`GradientProblem`](@ref) at the [`MPoint`](@ref) `x`.
"""
function getCost(p::P,x::MP) where {P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}
  return p.costFunction(x)
end
#
# Options
#
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
* `stepsize` : a `Function` to compute the next step size)
* `StepsizeOptions` : options the linesearch is called with.

# See also
[`steepestDescent`](@ref)
"""
mutable struct GradientDescentOptions{P <: MPoint, Q <: TVector, S <: StepsizeOptions} <: Options
    x::P where {P <: MPoint}
    xOld::P where {P <: MPoint}
    ∇::Q where {Q <: TVector}
    ∇Old::Q where {Q <: TVector}
    stepsize::Float64
    stepsizeOld::Float64
    stoppingCriterion::Function
    retraction::Function
    stepsizeFunction::Function
    stepsizeOptions::S
    GradientDescentOptions{P,Q,S}(
        initialX::P,
        stoppingCriterion::Function,
        stepsizeF::Function,
        stepsizeO::S,
        retraction::Function=exp
    ) where {P <: MPoint, Q <: TVector, S <: StepsizeOptions} = (
        o = new{P,typeofTVector(P),S}();
        o.x = initialX;
        o.stoppingCriterion = stoppingCriterion;
        o.retraction = retraction;
        o.stepsizeFunction = stepsizeF;
        o.stepsizeOptions = stepsizeO;
        return o
    )
end
GradientDescentOptions(x::P,sC::Function,sF::Function,sO::S,retraction::Function=exp) where {P <: MPoint, S <:StepsizeOptions} = GradientDescentOptions{P,typeofTVector(P),S}(x,sC,sF,sO,retraction)
#
# Access functions
#

"""
    getInitialStepsize(p,o)

for a [`GradientProblem`](@ref)` p` and some [`Options`](@ref)` o` return the
initial step size from within the [`StepsizeOptions`](@ref) within `o`.
"""
function getInitialStepsize(p::P,o::O) where {P <: GradientProblem{M} where M <: Manifold, O <: GradientDescentOptions}
    return getInitialStepsize(p,o,o.stepsizeOptions)
end
# default just do a line search as init.
function getInitialStepsize(p::P,o::O, sO::S) where {P <: GradientProblem{M} where M <: Manifold, O <: GradientDescentOptions, S <: StepsizeOptions}
    return o.stepsizeFunction(p,o,sO)
end
"""
    getStepsize(p,o,lo,vars...)

calculate a step size using the internal line search of the
[`GradientDescentOptions`](@ref)` o` belonging to the [`GradientProblem`](@ref),
where `vars...` might contain additional information
"""
function getStepsize(p::P,o::O,vars...) where {P <: GradientProblem{M} where M <: Manifold, O <: GradientDescentOptions}
    return o.stepsizeFunction(p,o,o.stepsizeOptions,vars...)
end
