#
# SubGradient Plan
#

@doc raw"""
    SubGradientProblem <: Problem

A structure to store information about a subgradient based optimization problem

# Fields
* `manifold` – a [`Manifold`](@ref)
* `cost` – the function $F$ to be minimized
* `subgradient` – a function returning a subgradient $\partial F$ of $F$

# Constructor

    SubGradientProblem(M, f, ∂f)

Generate the [`Problem`] for a subgradient problem, i.e. a function `f` on the
manifold `M` and a function `∂f` that returns an element from the subdifferential
at a point.
"""
mutable struct SubGradientProblem{mT <: Manifold} <: Problem
    M::mT
    cost::Function
    subgradient::Function
end
"""
    get_subgradient(p,x)

Evaluate the (sub)gradient of a [`SubGradientProblem`](@ref)` p` at the point `x`.
"""
function get_subgradient(p::P, x) where {P <: SubGradientProblem{M} where M <: Manifold}
    return p.subgradient(x)
end
"""
    SubGradientMethodOptions <: Options
stories option values for a [`subgradient_method`](@ref) solver

# Fields
* `retraction` – the retration to use within
* `stepsize` – a [`Stepsize`](@ref)
* `stop` – a [`StoppingCriterion`](@ref)
* `x` – (initial or current) value the algorithm is at
* `xOptimal` – optimal value
* `∂` the current element from the possivle subgradients at `x` that is used
"""
mutable struct SubGradientMethodOptions{P,T} <: Options where {P,T}
    retract!::Function
    stepsize::Stepsize
    stop::StoppingCriterion
    x::P
    xOptimal::P
    ∂::T
    function SubGradientMethodOptions{P}(
        M::TM,
        x::P,
        sC::StoppingCriterion,
        s::Stepsize,
        retr!::Function=exp!
        ) where {TM <: Manifold, P}
        o = new{P,typeof(zero_tangent_vector(M,x))}();
        o.x = x;
        o.xOptimal = x;
        o.stepsize = s; o.retract! = retr!;
        o.stop = sC;
        return o
    end
end
function SubGradientMethodOptions(
    M::Manifold,
    x::P,
    sC::StoppingCriterion,
    s::Stepsize,
    retr!::Function=exp!
) where {P}
    return SubGradientMethodOptions{P}(M, x, sC, s, retr!)
end