@doc raw"""
    ProxBundleMethodState <: AbstractManoptSolverState
stores option values for a [`prox_bundle_method`](@ref) solver

# Fields

* `bundle` - bundle that collects each iterate with the computed subgradient at the iterate
* `index_set` - the index set that keeps track of the strictly positive convex coefficients of the subproblem
* `inverse_retraction_method` - the inverse retraction to use within
* `lin_errors` - linearization errors at the last serious step
* `m` - the parameter to test the decrease of the cost
* `p` - current iterate
* `p_last_serious` - last serious iterate
* `prox_bundle` - bundle that collects each iterate with its computed proximal subgradient
* `retraction_method` – the retraction to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `size` - the maximal size of the bundle
* `vector_transport_method` - the vector transport method to use within
* `X` - (`zero_vector(M, p)`) the current element from the possible subgradients at
    `p` that was last evaluated.

# Constructor

ProxBundleMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields above besides `p_last_serious` which obtains the same type as `p`.
You can use e.g. `X=` to specify the type of tangent vector to use

"""
mutable struct ProxBundleMethodState{
    IR<:AbstractInverseRetractionMethod,
    P,
    T,
    TR<:AbstractRetractionMethod,
    TSC<:StoppingCriterion,
    VT<:AbstractVectorTransportMethod,
    R<:Real,
} <: AbstractManoptSolverState where {P,T}
    approx_error::AbstractVector{R}
    bundle::AbstractVector{Tuple{P,T}}
    d::T
    inverse_retraction_method::IR
    lin_errors::AbstractVector{R}
    m::R
    p::P
    p_last_serious::P
    retraction_method::TR
    size::Integer
    stop::TSC
    transported_subgradients::AbstractVector{T}
    vector_transport_method::VT
    X::T
    α::R
    α₀::R
    ε::R
    η::R
    μ::R
    ν::R
    function ProxBundleMethodState(
        M::TM,
        p::P;
        m::R=0.0125,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenProxBundleLess(1e-8),
        size::Integer=50,
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
        X::T=zero_vector(M, p),
        α₀::R=1.2,
        ε::R=1e-2,
        μ::R=0.5,
    ) where {
        IR<:AbstractInverseRetractionMethod,
        P,
        T,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        VT<:AbstractVectorTransportMethod,
        R<:Real,
    }
        # Initialize indes set, bundle points, linearization errors, and stopping parameter
        approx_errors = [0.0]
        bundle = [(copy(M, p), copy(M, p, X))]
        d = copy(M, p, X)
        lin_errors = [0.0]
        transported_subgradients = [copy(M, p, X)]
        α = 0.0
        η = 0.0
        ν = 0.0
        return new{IR,P,T,TR,SC,VT,R}(
            approx_errors,
            bundle,
            d,
            inverse_retraction_method,
            lin_errors,
            m,
            p,
            copy(M, p),
            retraction_method,
            size,
            stopping_criterion,
            transported_subgradients,
            vector_transport_method,
            X,
            α,
            α₀,
            ε,
            η,
            μ,
            ν,
        )
    end
end
get_iterate(bms::ProxBundleMethodState) = bms.p_last_serious
get_subgradient(bms::ProxBundleMethodState) = bms.d

@doc raw"""
    prox_bundle_method(M, f, ∂f, p)

perform a proximal bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)``,

where ``g_k = \sum_{j\in J_k} λ_j^k \mathrm{P}_{p_k←q_j}X_{q_j}``,

with ``X_{q_j}\in∂f(q_j)``, and

where ``\mathrm{retr}`` is a retraction and ``p_k`` is the last serious iterate.
Though the subgradient might be set valued, the argument `∂f` should always
return _one_ element from the subgradient, but not necessarily deterministic.

# Input
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `∂f`– the (sub)gradient ``\partial f: \mathcal M→ T\mathcal M`` of f
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

# Optional
* `m` - a real number that controls the decrease of the cost function
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by
   allocation (default) form `∂f(M, q)` or [`MutatingEvaluation`](@ref) in place, i.e. is
   of the form `∂f!(M, X, p)`.
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction method to use
* `retraction` – (`default_retraction_method(M, typeof(p))`) a `retraction(M, p, X)` to use.
* `stopping_criterion` – ([`StopWhenProxBundleLess`](@ref)`(1e-8)`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
* `vector_transport_method` - (`default_vector_transport_method(M, typeof(p))`) a vector transport method to use
...
and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
function prox_bundle_method(
    M::AbstractManifold, f::TF, ∂f::TdF, p; kwargs...
) where {TF,TdF}
    p_star = copy(M, p)
    return prox_bundle_method!(M, f, ∂f, p_star; kwargs...)
end
@doc raw"""
    prox_bundle_method!(M, f, ∂f, p)

perform a proximal bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)`` in place of `p`

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`- the (sub)gradient ``\partial f:\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

for more details and all optional parameters, see [`prox_bundle_method`](@ref).
"""
function prox_bundle_method!(
    M::AbstractManifold,
    f::TF,
    ∂f!!::TdF,
    p;
    m=0.0125,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    size=50,
    stopping_criterion::StoppingCriterion=StopWhenProxBundleLess(1e-8),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    α₀=1.2,
    ε=1e-2,
    μ=0.5,
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    bms = ProxBundleMethodState(
        M,
        p;
        m=m,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        size=size,
        stopping_criterion=stopping_criterion,
        vector_transport_method=vector_transport_method,
        α₀=α₀,
        ε=ε,
        μ=μ,
    )
    bms = decorate_state!(bms; kwargs...)
    return get_solver_return(solve!(mp, bms))
end
function initialize_solver!(mp::AbstractManoptProblem, bms::ProxBundleMethodState)
    M = get_manifold(mp)
    copyto!(M, bms.p_last_serious, bms.p)
    get_subgradient!(mp, bms.X, bms.p)
    copyto!(M, bms.d, bms.p_last_serious, bms.X)
    bms.bundle = [(copy(M, bms.p), copy(M, bms.p, bms.X))]
    return bms
end
function prox_bundle_method_sub_solver(::Any, ::Any)
    throw(
        ErrorException("""Both packages "QuadraticModels" and "RipQP" need to be loaded.""")
    )
end
function step_solver!(mp::AbstractManoptProblem, bms::ProxBundleMethodState, i)
    M = get_manifold(mp)
    v = maximum([
        -2 * ej /
        norm(
            M,
            bms.p_last_serious,
            inverse_retract(
                M, bms.p_last_serious, qj, bms.inverse_retraction_method
            ),
        )^2 for (ej, (qj, Xj)) in zip(bms.lin_errors, bms.bundle) if
        !(qj ≈ bms.p_last_serious)
    ])
    if !isempty(v)
        bms.η =
            bms.α₀ + max(
                bms.α₀,
                bms.α,
                v,
                )
    else
        bms.η =
        bms.α₀ + max(
            bms.α₀,
            bms.α,
            )
    end
    bms.transported_subgradients = [
        vector_transport_to(M, qj, Xj, bms.p_last_serious, bms.vector_transport_method) +
        bms.η * inverse_retract(M, bms.p_last_serious, qj, bms.inverse_retraction_method)
        for (qj, Xj) in bms.bundle
    ]
    bms.d = prox_prox_bundle_method_sub_solver(mp, bms)
    bms.ν = maximum([
        -c + inner(M, bms.p_last_serious, bms.d, Xj) for
        (c, Xj) in zip(bms.approx_error, bms.transported_subgradients)
    ])
    if norm(M, bms.p_last_serious, bms.d) ≤ bms.ε
        retract!(M, bms.p, bms.p_last_serious, bms.d, bms.retraction_method)
        get_subgradient!(mp, bms.X, bms.p)
        bms.α = 0
    else
        retract!(
            M,
            bms.p,
            bms.p_last_serious,
            bms.ε * bms.d / norm(M, bms.p_last_serious, bms.d),
            bms.retraction_method,
        )
        get_subgradient!(mp, bms.X, bms.p)
        bms.α =
            -inner(
                M,
                bms.p_last_serious,
                bms.d,
                vector_transport_to(
                    M, bms.p, bms.X, bms.p_last_serious, bms.vector_transport_method
                ),
            ) / (bms.ε * norm(M, bms.p_last_serious, bms.d))
    end
    if get_cost(mp, bms.p) ≤ (get_cost(mp, bms.p_last_serious) + bms.m * bms.ν)
        copyto!(M, bms.p_last_serious, bms.p)
    end
    l = length(bms.bundle)
    push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
    if l == bms.size
        deleteat!(bms.bundle, l - bms.size)
    end
    bms.lin_errors = [
        get_cost(mp, bms.p_last_serious) - get_cost(mp, qj) - inner(
            M,
            qj,
            Xj,
            inverse_retract(M, qj, bms.p_last_serious, bms.inverse_retraction_method),
        ) for (qj, Xj) in bms.bundle
    ]
    bms.approx_error =
        bms.lin_errors + [
            bms.η / 2 *
            norm(
                M,
                bms.p_last_serious,
                inverse_retract(M, bms.p_last_serious, qj, bms.inverse_retraction_method),
            )^2 for (qj, Xj) in bms.bundle
        ]
    return bms
end
get_solver_result(bms::ProxBundleMethodState) = bms.p_last_serious

"""
    StopWhenProxBundleLess <: StoppingCriterion

Stopping criterion for [`prox_bundle_method`](@ref) to indicate to stop when

* the parameter -ν = -max{−c^k_j +  (ξ^k_j,d) }.

is less than a given tolerance tol.

# Constructor

    StopWhenProxBundleLess(tol=1e-8)

"""
mutable struct StopWhenProxBundleLess{R} <: StoppingCriterion
    tol::R
    reason::String
    at_iteration::Int
    function StopWhenProxBundleLess(tol=1e-8)
        return new{typeof(tol)}(tol, "", 0)
    end
end
function (b::StopWhenProxBundleLess)(
    mp::AbstractManoptProblem, bms::ProxBundleMethodState, i::Int
)
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    M = get_manifold(mp)
    if -bms.ν ≤ b.tol && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter -ν = $(-bms.ν) is less than $(b.tol).\n"
        b.at_iteration = i
        return true
    end
    return false
end
function status_summary(b::StopWhenProxBundleLess)
    has_stopped = length(b.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Stopping parameter: -ν ≤ $(b.tol):\t$s"
end
function show(io::IO, b::StopWhenProxBundleLess)
    return print(io, "StopWhenProxBundleLess($(b.tol)\n    $(status_summary(b))")
end
