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

#
# SubGradientMethod Options
"""
    SubGradientMethodOptions <: Options
stories option values for a [`subGradientMethod`](@ref) solver

# Fields
* `retraction` – the retration to use within
* `stepsizeFunction` – a function (p,o,sO[,i]) -> s to determine the next step size.
* `stepsizeOptions` – options for the current state of the step size determination, see [`StepsizeOptions`](@ref).
* `stoppingCriterion` – stopping criterion for the algorithm
* `x` – (initial or current) value the algorithm is at
* `optimalX` – optimal value
"""
mutable struct SubGradientMethodOptions{P,T,S} <: Options where {P <: MPoint, T <: TVector, S <: StepsizeOptions}
    retraction::Function
    stepsizeFunction::Function
    stepsizeOptions::S
    stoppingCriterion::Function
    x::P
    xLast::P
    xOptimal::P
    subGradient::T
    SubGradientMethodOptions{P,T,S}(x::P,sC::Function,sF::Function,sO::S,retr::Function=exp) where {P <: MPoint, T <: TVector, S <: StepsizeOptions} = (
        o = new{P,T,S}(); o.x = x; o.xLast = x; o.xOptimal = x;
        o.stepsizeFunction = sF; o.stepsizeOptions = sO; o.retraction = retr;
        o.stoppingCriterion = sC;
        return o
    )
end
SubGradientMethodOptions(x::P,sC::Function,stepsize::Function,sF::Function,sO::S,retr::Function=exp) where {P <: MPoint, S <: StepsizeOptions} = SubGradientMethodOptions{P,typeofTVector(x),L}(x,sC,stepsize,sF,sO,retr)
"""
    getStepsize(p,o,lo,vars...)

calculate a step size using the internal line search of the
[`SubGradientMethodOptions`](@ref)` o` belonging to the [`GradientProblem`](@ref),
where `vars...` might contain additional information
"""
function getStepsize(p::P,o::O,vars...) where {P <: GradientProblem{M} where M <: Manifold, O <: SubGradientMethodOptions}
    return o.stepsizeFunction(p,o,o.stepsizeOptions,vars...)
end
