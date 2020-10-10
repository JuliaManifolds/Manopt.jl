@doc raw"""
    SubGradientProblem <: Problem

A structure to store information about a subgradient based optimization problem

# Fields
* `manifold` – a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
* `cost` – the function $F$ to be minimized
* `subgradient` – a function returning a subgradient $\partial F$ of $F$

# Constructor

    SubGradientProblem(M, f, ∂f)

Generate the [`Problem`] for a subgradient problem, i.e. a function `f` on the
manifold `M` and a function `∂f` that returns an element from the subdifferential
at a point.
"""
struct SubGradientProblem{mT<:Manifold,TCost,TSubgradient} <: Problem
    M::mT
    cost::TCost
    subgradient::TSubgradient
end
"""
    get_subgradient(p,x)

Evaluate the (sub)gradient of a [`SubGradientProblem`](@ref)` p` at the point `x`.
"""
function get_subgradient(p::SubGradientProblem, x)
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
mutable struct SubGradientMethodOptions{TRetract<:AbstractRetractionMethod,TStepsize,P,T} <:
               Options where {P,T}
    retraction_method::TRetract
    stepsize::TStepsize
    stop::StoppingCriterion
    x::P
    xOptimal::P
    ∂::T
    function SubGradientMethodOptions(
        M::TM,
        x::P,
        sC::StoppingCriterion,
        s::Stepsize,
        retraction_method=ExponentialRetraction(),
    ) where {TM<:Manifold,P}
        return new{typeof(retraction_method),typeof(s),P,typeof(zero_tangent_vector(M, x))}(
            retraction_method, s, sC, x, x
        )
    end
end
