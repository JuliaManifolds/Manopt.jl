@doc raw"""
    SubGradientProblem <:AbstractManoptProblem

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
struct SubGradientProblem{T<:AbstractEvaluationType,mT<:AbstractManifold,C,S} <:
       AbstractManoptProblem{mT}
    M::mT
    cost::C
    subgradient!!::S
    function SubGradientProblem(
        M::mT,
        cost::C,
        subgrad::S;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {mT<:AbstractManifold,C,S}
        return new{typeof(evaluation),mT,C,S}(M, cost, subgrad)
    end
end
"""
    get_subgradient(p, q)
    get_subgradient!(p, X, q)

Evaluate the (sub)gradient of a [`SubGradientProblem`](@ref)` p` at the point `q`.

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`InplaceEvaluation`](@ref)
memory for the result is allocated.
"""
function get_subgradient(p::SubGradientProblem{AllocatingEvaluation}, q)
    return p.subgradient!!(p.M, q)
end
function get_subgradient(p::SubGradientProblem{InplaceEvaluation}, q)
    X = zero_vector(p.M, q)
    return p.subgradient!!(p.M, X, q)
end
function get_subgradient!(p::SubGradientProblem{AllocatingEvaluation}, X, q)
    return copyto!(p.M, X, p.subgradient!!(p.M, q))
end
function get_subgradient!(p::SubGradientProblem{InplaceEvaluation}, X, q)
    return p.subgradient!!(p.M, X, q)
end

"""
    SubGradientMethodState <: AbstractManoptSolverState
stories option values for a [`subgradient_method`](@ref) solver

# Fields
* `retraction_method` – the retration to use within
* `stepsize` – a [`Stepsize`](@ref)
* `stop` – a [`StoppingCriterion`](@ref)
* `x` – (initial or current) value the algorithm is at
* `x_optimal` – optimal value
* `∂` the current element from the possible subgradients at `x` that is used
"""
mutable struct SubGradientMethodState{
    TR<:AbstractRetractionMethod,TS<:Stepsize,TSC<:StoppingCriterion,P,T
} <: AbstractManoptSolverState where {P,T}
    retraction_method::TR
    stepsize::TS
    stop::TSC
    x::P
    x_optimal::P
    ∂::T
    function SubGradientMethodState(
        M::TM,
        x::P;
        stopping_criterion::SC=StopAfterIteration(5000),
        stepsize::S=ConstantStepsize(M),
        subgrad::T=zero_vector(M, x),
        retraction_method::TR=default_retraction_method(M),
    ) where {
        TM<:AbstractManifold,
        P,
        T,
        SC<:StoppingCriterion,
        S<:Stepsize,
        TR<:AbstractRetractionMethod,
    }
        return new{TR,S,SC,P,T}(
            retraction_method, stepsize, stopping_criterion, x, deepcopy(x), subgrad
        )
    end
end
get_iterate(o::SubGradientMethodOptions) = o.x
