#
# SubGradient Plan
#
export SubGradientMethodOptions, SubGradientProblem
export getCost, getSubGradient
export getStepSize, evaluateStoppingCriterion
#
# Problem
@doc doc"""
    SubGradientProblem <: Problem
A structure to store information about a subgradient based optimization problem

# Fields
* `M` – a [`Manifold`](@ref)
* `costFunction` – the function $F$ to be minimized
* `subGradient` – a function returning a subgradient $\partial F$ of $F$
"""
mutable struct SubGradientProblem{mT <: Manifold} <: Problem
    M::mT
    costFunction::Function
    subGradient::Function
end
"""
    getSubGradient(p,x)

evaluate the gradient of a [`SubGradientProblem`](@ref)` p` at the [`MPoint`](@ref)` x`.
"""
getSubGradient(p::P,x::MP) where {P <: SubGradientProblem{M} where M <: Manifold, MP <: MPoint} = p.subGradient(x)
"""
    getCost(p,x)

evaluate the cost function `F` stored within a [`GradientProblem`](@ref) at the [`MPoint`](@ref)` x`.
"""
getCost(p::P,x::MP) where {P <: SubGradientProblem{M} where M <: Manifold, MP <: MPoint} = p.costFunction(x)

#
# SubGradientMethod Options
"""
    SubGradientMethodOptions <: Options
stories option values for a [`subGradientMethod`](@ref) solver

# Fields
* `retraction` – the retration to use within
* `stepSize` – a function returning the step size.
* `stoppingCriterion` – stopping criterion for the algorithm
* `x0` – Initial value the algorithm starts
"""
mutable struct SubGradientMethodOptions <: Options
    retraction::Function
    stepSize::Function
    stoppingCriterion::Function
    x0::P where P <: MPoint
    SubGradientMethodOptions(x,sC,retr,stepS) = new(retr,stepS,sC,x)
end
"""
    getStepSize(p,o,vars...)
evaluate the step size function from the plan of a subgradient plan given by the
`SubGradientMethodOptions o` and the `SubGradientProblem p` with respect to the
variables `vars...`.
"""
function getStepsize(p::P,o::O,vars...) where {P <: SubGradientProblem{M} where M <: Manifold, O <: SubGradientMethodOptions}
    return o.stepSize(vars...)
end
"""
    evaluateStoppingCriterion(o,iter,ξ,x,xnew)
evaluate the stopping criterion stored within the `SubGradientMethodOptions o`
with respect to the current iteration `iter`, the last and current iterates
`x` and `xnew` as well as the current subgradient `ξ`.
"""
function evaluateStoppingCriterion(o::O,iter::I,ξ::MT, x::P, xnew::P) where {O<:SubGradientMethodOptions, P <: MPoint, MT <: TVector, I<:Integer}
  o.stoppingCriterion(iter,ξ,x,xnew)
end
