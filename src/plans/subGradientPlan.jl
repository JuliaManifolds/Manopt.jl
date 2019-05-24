#
# SubGradient Plan
#
export SubGradientMethodOptions, SubGradientProblem
export getCost, getSubGradient
export getStepsize
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

evaluate the gradient of a [`SubGradientProblem`](@ref)` p` at the [`MPoint`](@ref) `x`.
"""
getSubGradient(p::P,x::MP) where {P <: SubGradientProblem{M} where M <: Manifold, MP <: MPoint} = p.subGradient(x)
"""
    getCost(p,x)

evaluate the cost function `F` stored within a [`GradientProblem`](@ref) at the [`MPoint`](@ref) `x`.
"""
getCost(p::P,x::MP) where {P <: SubGradientProblem{M} where M <: Manifold, MP <: MPoint} = p.costFunction(x)

"""
    SubGradientMethodOptions <: Options
stories option values for a [`subGradientMethod`](@ref) solver

# Fields
* `retraction` – the retration to use within
* `stepsize´ – see [`Stepsize`](@ref)
* `stoppingCriterion` – [`StoppingCriterion`](@ref)
* `x` – (initial or current) value the algorithm is at
* `optimalX` – optimal value
"""
mutable struct SubGradientMethodOptions{P,T} <: Options where {P <: MPoint, T <: TVector}
    retraction::Function
    stepsize::Stepsize
    stoppingCriterion::StoppingCriterion
    x::P
    xOld::P
    xOptimal::P
    subGradient::T
    SubGradientMethodOptions{P,T}(x::P,sC::StoppingCriterion,s::Stepsize,retr::Function=exp) where {P <: MPoint, T <: TVector} = (
        o = new{P,T}(); o.x = x; o.xLast = x; o.xOptimal = x;
        o.stepsize = s; o.retraction = retr;
        o.stoppingCriterion = sC;
        return o
    )
end
SubGradientMethodOptions(x::P,sC::StoppingCriterion,stepsize::Stepsize,retr::Function=exp) where {P <: MPoint} = SubGradientMethodOptions{P,typeofTVector(x)}(x,sC,s,retr)

getStepsize(p::P,o::O,vars...) where {P <: SubGradientProblem{M} where M <: Manifold, O <: SubGradientMethodOptions} = o.stepsze(p,o,vars...)
