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
function get_bundle_subgradient(prb::BundleProblem{AllocatingEvaluation}, q)
    return prb.subgradient!!(prb.M, q)
end
function get_bundle_subgradient(prb::BundleProblem{MutatingEvaluation}, q)
    X = zero_vector(prb.M, q)
    return prb.subgradient!!(prb.M, X, q)
end
function get_bundle_subgradient!(prb::BundleProblem{AllocatingEvaluation}, X, q)
    return copyto!(prb.M, X, prb.subgradient!!(prb.M, q))
end
function get_bundle_subgradient!(prb::BundleProblem{MutatingEvaluation}, X, q)
    return prb.subgradient!!(prb.M, X, q)
end

"""
    BundleMethodOptions <: Options
stores option values for a [`bundle_method`](@ref) solver

# Fields

* `index_set` - the index set that keeps track of the strictly positive convex coefficients of the subproblem
* `bundle_points` - collects each iterate `p` with the computed subgradient `∂` at the iterate
* `lin_errors` - linearization errors at the last serious step
* `m` - the parameter to test the decrease of the cost
* `p` - current iterate
* `p_last_serious` - last serious iterate
* `retraction_method` – the retration to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `tol` - the tolerance parameter
* `vector_transport_method` - the vector transport method to use within
* `X` the current element from the possible subgradients at `p` that is used
"""
mutable struct BundleMethodOptions{
    IR<:AbstractInverseRetractionMethod,
    L<:Array,
    P,
    T,
    TR<:AbstractRetractionMethod,
    TSC<:StoppingCriterion,
    S<:Set,
    VT<:AbstractVectorTransportMethod,
} <: Options where {P,T}
    bundle_points::AbstractVector{Tuple{P,T}}
    inverse_retraction_method::IR
    lin_errors::L
    p::P
    p_last_serious::P
    X::T
    retraction_method::TR
    stop::TSC
    index_set::S
    vector_transport_method::VT
    m::Real
    tol::Real
    function BundleMethodOptions(
        M::TM,
        p::P;
        m::Real=0.0125,
        #lin_errors::L=[0],
        #index_set::S=Set(1),
        inverse_retraction_method::IR=default_inverse_retraction_method(M),
        retraction_method::TR=default_retraction_method(M),
        stopping_criterion::SC=StopAfterIteration(5000),
        subgrad::T=zero_vector(M, p),
        #bundle_points::A=[p, subgrad],
        tol::Real=1e-8,
        vector_transport_method::VT=default_vector_transport_method(M),
    ) where {
        #A<:Array,
        IR<:AbstractInverseRetractionMethod,
        #L<:Array,
        P,
        T,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        #S<:Set,
        VT<:AbstractVectorTransportMethod,
    }
        index_set = Set(1)
        bundle_points = [(p, subgrad)]
        lin_errors = [0.0]
        #Float64[]
        return new{IR,typeof(lin_errors),P,T,TR,SC,typeof(index_set),VT}(
            bundle_points,
            inverse_retraction_method,
            lin_errors,
            p,
            deepcopy(p),
            subgrad,
            retraction_method,
            stopping_criterion,
            index_set,
            vector_transport_method,
            m,
            tol,
        )
    end
end
get_iterate(o::BundleMethodOptions) = o.p_last_serious
function BundleMethodSubsolver(M::AbstractManifold, o::BundleMethodOptions, X::T) where {T}
    d = length(o.index_set)
    λ = Variable(d)
    problem = minimize(0.5 * norm(M, o.p_last_serious, sum(λ .* X))^2 + sum(λ .* o.lin_errors))
    problem.constraints +=  [i >= 0 for i in λ]
    problem.constraints += [sum(λ) == 1]
    solve!(problem, SCS.Optimizer; silent_solver=true)
    return evaluate(λ)
end
