@doc raw"""
    BundleProblem <: Problem

A structure to store information about a bundle based optimization problem

# Fields
* `manifold` – a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
* `cost` – the function $F$ to be minimized
* `subgradient` – a function returning a subgradient $\partial F$ of $F$

# Constructor

    BundleProblem(M, f, ∂f)

Generate the [`Problem`] for a bundle problem, i.e. a function `f` on the
manifold `M` and a function `∂f` that returns an element from the subdifferential
at a point.
"""
struct BundleProblem{T<:AbstractEvaluationType,mT<:AbstractManifold,C,S} <: Problem{T}
    M::mT
    cost::C
    subgradient!!::S
    function BundleProblem(
        M::mT,
        cost::C,
        subgrad::S;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {mT<:AbstractManifold,C,S}
        return new{typeof(evaluation),mT,C,S}(M, cost, subgrad)
    end
end
"""
    get_subgradient(prb, q)
    get_subgradient!(prb, X, q)

Evaluate the (sub)gradient of a [`SubGradientProblem`](@ref) `prb` at the point `q`.

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`MutatingEvaluation`](@ref)
memory for the result is allocated.
"""
function get_subgradient(prb::BundleProblem{AllocatingEvaluation}, q)
    return prb.subgradient!!(prb.M, q)
end
function get_subgradient(prb::BundleProblem{MutatingEvaluation}, q)
    X = zero_vector(prb.M, q)
    return prb.subgradient!!(prb.M, X, q)
end
function get_subgradient!(prb::BundleProblem{AllocatingEvaluation}, X, q)
    return copyto!(prb.M, X, prb.subgradient!!(prb.M, q))
end
function get_subgradient!(prb::BundleProblem{MutatingEvaluation}, X, q)
    return prb.subgradient!!(prb.M, X, q)
end

"""
    BundleMethodOptions <: Options
stores option values for a [`bundle_method`](@ref) solver

# Fields

* `L` - the bundle that comprises each iterate `p`` with the computed subgradient `∂` at the iterate
* `p` - current iterate
* `p_last_serious` - last serious iterate
* `retraction_method` – the retration to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `vector_transport_method` - the vector transport method to use within
* `∂` the current element from the possible subgradients at `p` that is used
"""
mutable struct BundleMethodOptions{
    P,TR<:AbstractRetractionMethod,TSC<:StoppingCriterion,VT<:AbstractVectorTransportMethod
} <: Options where {P,T}
    p::P
    p_last_serious::P
    retraction_method::TR
    stop::TSC
    vector_transport_method::VT
    ∂::T
    function BundleMethodOptions(
        M::TM,
        p::P;
        retraction_method::TR=default_retraction_method(M),
        stopping_criterion::SC=StopAfterIteration(5000),
        vector_transport_method::VT=default_vector_transport_method(M),
    ) where {TM<:AbstractManifold,P,TR<:AbstractRetractionMethod,SC<:StoppingCriterion,VT<:AbstractVectorTransportMethod}
        return new{P,TR,SC,VT,T}(
            p,
            deepcopy(p),
            retraction_method,
            stopping_criterion,
            vector_transport_method,
            subgrad,
        )
    end
end
get_iterate(o::BundleMethodOptions) = o.p
