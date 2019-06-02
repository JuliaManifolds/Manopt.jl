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
    SubGradientMethodOptions <: Options
stories option values for a [`subGradientMethod`](@ref) solver

# Fields
* `retraction` – the retration to use within
* `stepsize` – a [`Stepsize`](@ref)
* `stop` – a [`StoppingCriterion`](@ref)
* `x` – (initial or current) value the algorithm is at
* `optimalX` – optimal value
"""
mutable struct SubGradientMethodOptions{P,T} <: Options where {P <: MPoint, T <: TVector}
    retraction::Function
    stepsize::Stepsize
    stop::StoppingCriterion
    x::P
    xOptimal::P
    ∂::T
    SubGradientMethodOptions{P,T}(x::P,sC::StoppingCriterion,s::Stepsize,retr::Function=exp) where {P <: MPoint, T <: TVector} = (
        o = new{P,T}(); o.x = x; o.xOptimal = x;
        o.stepsize = s; o.retraction = retr;
        o.stop = sC;
        return o
    )
end
SubGradientMethodOptions(x::P,sC::StoppingCriterion,s::Stepsize,retr::Function=exp) where {P <: MPoint} = SubGradientMethodOptions{P,typeofTVector(x)}(x,sC,s,retr)