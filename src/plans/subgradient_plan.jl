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
struct SubGradientProblem{T,mT<:Manifold,C,S} <: Problem{T}
    M::mT
    cost::C
    subgradient!!::S
    function SubGradientProblem(
        M::mT,
        cost::C,
        subgrad::S;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {mT<:Manifold,C,S}
        return new{typeof(evaluation),mT,C,S}(M, cost, subgrad)
    end
end
"""
    get_subgradient(p, q)
    get_subgradient!(p, X, q)

Evaluate the (sub)gradient of a [`SubGradientProblem`](@ref)` p` at the point `q`.

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`MutatingEvaluation`](@ref)
memory for the result is allocated.
"""
function get_subgradient(p::SubGradientProblem{AllocatingEvaluation}, q)
    return p.subgradient!!(p.M, q)
end
function get_subgradient(p::SubGradientProblem{MutatingEvaluation}, q)
    X = zero_tangent_vector(p.M, q)
    return p.subgradient!!(p.M, X, q)
end
function get_subgradient!(p::SubGradientProblem{AllocatingEvaluation}, X, q)
    return copyto!(p.M, X, p.subgradient!!(p.M, q))
end
function get_subgradient!(p::SubGradientProblem{MutatingEvaluation}, X, q)
    return p.subgradient!!(p.M, X, q)
end

"""
    SubGradientMethodOptions <: Options
stories option values for a [`subgradient_method`](@ref) solver

# Fields
* `retraction_method` – the retration to use within
* `stepsize` – a [`Stepsize`](@ref)
* `stop` – a [`StoppingCriterion`](@ref)
* `x` – (initial or current) value the algorithm is at
* `x_optimal` – optimal value
* `∂` the current element from the possivle subgradients at `x` that is used
"""
mutable struct SubGradientMethodOptions{TRetract<:AbstractRetractionMethod,TStepsize,P,T} <:
               Options where {P,T}
    retraction_method::TRetract
    stepsize::TStepsize
    stop::StoppingCriterion
    x::P
    x_optimal::P
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
