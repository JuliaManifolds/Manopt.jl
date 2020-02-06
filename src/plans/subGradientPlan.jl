#
# SubGradient Plan
#
export SubGradientMethodOptions, SubGradientProblem
export getCost, getSubGradient
export getStepsize
#
# Problem
@doc raw"""
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

evaluate the gradient of a [`SubGradientProblem`](@ref)` p` at the point `x`.
"""
getSubGradient(p::P,x) where {P <: SubGradientProblem{M} where M <: Manifold} = p.subGradient(x)
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
mutable struct SubGradientMethodOptions{P,T} <: Options where {P,T}
    retraction::Function
    stepsize::Stepsize
    stop::StoppingCriterion
    x::P
    xOptimal::P
    ∂::T
    function SubGradientMethodOptions{P,T}(
        x::P,
        sC::StoppingCriterion,
        s::Stepsize,
        retr::Function=exp
        ) where {P,T}
        o = new{P,T}(); o.x = x; o.xOptimal = x;
        o.stepsize = s; o.retraction = retr;
        o.stop = sC;
        return o
    end
end
function SubGradientMethodOptions(
    x::P,
    sC::StoppingCriterion,
    s::Stepsize,
    retr::Function=exp
    ) where {P}
    return SubGradientMethodOptions{P,typeofTVector(x)}(x,sC,s,retr)
end