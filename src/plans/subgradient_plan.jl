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
struct ManifoldSubgradientObjective{T<:AbstractEvaluationType,C,S} <:
       AbstractManifoldCostObjective{T}
    cost::C
    subgradient!!::S
    function SubGradientProblem(
        cost::C,
        subgrad::S;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {mT<:AbstractManifold,C,S}
        return new{typeof(evaluation),C,S}(cost, subgrad)
    end
end

@doc raw"""
    get_subgradient(mp::AbstractManoptProblem, p)
    get_subgradient!(mp::AbstractManoptProblem, X, p)

evaluate the subgradient of an [`AbstractManoptProblem`](@ref) `mp` at `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(mp::AbstractManoptProblem, p)
    return get_subgradient(get_manifold(mp), get_objective(mp), p)
end
function get_subgradient!(mp::AbstractManoptProblem, X, p)
    return get_subgradient!(get_manifold(mp), X, get_objective(mp), p)
end

"""
    X = get_subgradient(M;;AbstractManifold, sgo::ManifoldSubgradientObjective, p)
    get_subgradient!(M;;AbstractManifold, X, sgo::ManifoldSubgradientObjective, p)

Evaluate the (sub)gradient of a [`ManifoldSubgradientObjective`](@ref) `sgo`
at the point `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(M::AbstractManifold, sgo::ManifoldSubgradientObjective{AllocatingEvaluation}, p)
    return sgo.subgradient!!(M, q)
end
function get_subgradient(M::AbstractManifold, sgo::ManifoldSubgradientObjective{InplaceEvaluation}, p)
    X = zero_vector(M, q)
    return sgo.subgradient!!(M, X, q)
end
function get_subgradient!(M::AbstractManifold, sgo::ManifoldSubgradientObjective{AllocatingEvaluation}, X, p)
    copyto!(M, X, sgo.subgradient!!(M, q))
    return X
end
function get_subgradient!(M::AbstractManifold, sgo::SubGradientProblem{InplaceEvaluation}, X, p)
    sgo.subgradient!!(p.M, X, q)
    return X
end

"""
    SubGradientMethodState <: AbstractManoptSolverState

stories option values for a [`subgradient_method`](@ref) solver

# Fields
* `retraction_method` – the retration to use within
* `stepsize` – ([`ConstantStepsize`](@ref)`(M)`) a [`Stepsize`](@ref)
* `stop` – ([`StopAfterIteration`](@ref)`(5000)``)a [`StoppingCriterion`](@ref)
* `p` – (initial or current) value the algorithm is at
* `p_star` – optimal value (initialized to a copy of `p`.)
* `X` ([`zero_vector`](@ref)`(M, p)`) the current element from the possible subgradients at
   `p` that was last evaluated.

# Constructor

SubGradientMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields above besides `p_star` which obtains the same type as `p`.
You can use e.g. `X=` to specify the type of tangent vector to use

"""
mutable struct SubGradientMethodState{
    TR<:AbstractRetractionMethod,TS<:Stepsize,TSC<:StoppingCriterion,P,T
} <: AbstractManoptSolverState where {P,T}
    p::P
    p_star::P
    retraction_method::TR
    stepsize::TS
    stop::TSC
    X::T
    function SubGradientMethodState(
        M::TM,
        p::P;
        stopping_criterion::SC=StopAfterIteration(5000),
        stepsize::S=ConstantStepsize(M),
        X::T=zero_vector(M, p),
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
            p, copy(M, p), retraction_method, stepsize, stopping_criterion, subgrad
        )
    end
end
get_iterate(o::SubGradientMethodState) = o.p
get_subgradient(o::SubGradientMethodState) = o.X
