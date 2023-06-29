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
* `bundle_size` - the maximal bundle_size of the bundle
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
    approx_errors::AbstractVector{R}
    bundle::AbstractVector{Tuple{P,T}}
    c::R
    d::T
    inverse_retraction_method::IR
    lin_errors::AbstractVector{R}
    m::R
    p::P
    p_last_serious::P
    retraction_method::TR
    bundle_size::Integer
    stop::TSC
    transported_subgradients::AbstractVector{T}
    vector_transport_method::VT
    X::T
    α::R
    α₀::R
    ε::R
    δ::R
    η::R
    λ::AbstractVector{R}
    μ::R
    ν::R
    function ProxBundleMethodState(
        M::TM,
        p::P;
        m::R=0.0125,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenProxBundleLess(1e-8),
        bundle_size::Integer=50,
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
        X::T=zero_vector(M, p),
        α₀::R=1.2,
        ε::R=1e-2,
        δ::R=1.0,
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
        c = 0.0
        d = copy(M, p, X)
        lin_errors = [0.0]
        transported_subgradients = [copy(M, p, X)]
        α = 0.0
        λ = [0.0]
        η = 0.0
        ν = 0.0
        return new{IR,P,T,TR,SC,VT,R}(
            approx_errors,
            bundle,
            c,
            d,
            inverse_retraction_method,
            lin_errors,
            m,
            p,
            copy(M, p),
            retraction_method,
            bundle_size,
            stopping_criterion,
            transported_subgradients,
            vector_transport_method,
            X,
            α,
            α₀,
            ε,
            δ,
            η,
            λ,
            μ,
            ν,
        )
    end
end
get_iterate(pbms::ProxBundleMethodState) = pbms.p_last_serious
get_subgradient(pbms::ProxBundleMethodState) = pbms.d

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
    bundle_size=50,
    stopping_criterion::StoppingCriterion=StopWhenProxBundleLess(1e-8),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    α₀=1.2,
    ε=1e-2,
    δ=1.0,
    μ=0.5,
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    pbms = ProxBundleMethodState(
        M,
        p;
        m=m,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        bundle_size=bundle_size,
        stopping_criterion=stopping_criterion,
        vector_transport_method=vector_transport_method,
        α₀=α₀,
        ε=ε,
        δ=δ,
        μ=μ,
    )
    pbms = decorate_state!(pbms; kwargs...)
    return get_solver_return(solve!(mp, pbms))
end
function initialize_solver!(mp::AbstractManoptProblem, pbms::ProxBundleMethodState)
    M = get_manifold(mp)
    copyto!(M, pbms.p_last_serious, pbms.p)
    get_subgradient!(mp, pbms.X, pbms.p)
    copyto!(M, pbms.d, pbms.p_last_serious, pbms.X)
    pbms.bundle = [(copy(M, pbms.p), copy(M, pbms.p, pbms.X))]
    return pbms
end
function step_solver!(mp::AbstractManoptProblem, pbms::ProxBundleMethodState, i)
    M = get_manifold(mp)
    v = [
        -2 * ej /
        norm(
            M,
            pbms.p_last_serious,
            inverse_retract(M, pbms.p_last_serious, qj, pbms.inverse_retraction_method),
        )^2 for
        (ej, (qj, Xj)) in zip(pbms.lin_errors, pbms.bundle) if !(qj ≈ pbms.p_last_serious)
    ]
    if !isempty(v)
        pbms.η = pbms.α₀ + max(pbms.α₀, pbms.α, maximum(v))
    else
        pbms.η = pbms.α₀ + max(pbms.α₀, pbms.α)
    end
    pbms.transported_subgradients = [
        vector_transport_to(M, qj, Xj, pbms.p_last_serious, pbms.vector_transport_method) +
        pbms.η * inverse_retract(M, pbms.p_last_serious, qj, pbms.inverse_retraction_method)
        for (qj, Xj) in pbms.bundle
    ]
    pbms.λ = bundle_method_sub_solver(M, pbms)
    pbms.c = sum(pbms.λ .* pbms.approx_errors)
    pbms.d .= -1 / pbms.μ * sum(pbms.λ .* pbms.transported_subgradients)
    nd = norm(M, pbms.p_last_serious, pbms.d)
    pbms.ν = -nd^2 - pbms.c
    if nd ≤ pbms.ε
        retract!(M, pbms.p, pbms.p_last_serious, pbms.d, pbms.retraction_method)
        get_subgradient!(mp, pbms.X, pbms.p)
        pbms.α = 0.0
    else
        retract!(M, pbms.p, pbms.p_last_serious, pbms.ε * pbms.d / nd, pbms.retraction_method)
        get_subgradient!(mp, pbms.X, pbms.p)
        pbms.α =
            -inner(
                M,
                pbms.p_last_serious,
                pbms.d,
                vector_transport_to(
                    M, pbms.p, pbms.X, pbms.p_last_serious, pbms.vector_transport_method
                ),
            ) / (pbms.ε * nd)
    end
    if get_cost(mp, pbms.p) ≤ (get_cost(mp, pbms.p_last_serious) + pbms.m * pbms.ν)
        # pbms.μ = pbms.μ * 
        #     norm(
        #         M,
        #         pbms.p,
        #         pbms.X - vector_transport_to(
        #             M,
        #             pbms.p_last_serious,
        #             get_subgradient(mp, pbms.p_last_serious),
        #             pbms.p,
        #             pbms.vector_transport_method,
        #         ),
        #     )^2 / (
        #         norm(
        #             M,
        #             pbms.p,
        #             pbms.X - vector_transport_to(
        #                 M,
        #                 pbms.p_last_serious,
        #                 get_subgradient(mp, pbms.p_last_serious),
        #                 pbms.p,
        #                 pbms.vector_transport_method,
        #             ),
        #         )^2 -
        #         pbms.μ * inner(
        #             M,
        #             pbms.p,
        #             inverse_retract(
        #                 M, pbms.p, pbms.p_last_serious, pbms.inverse_retraction_method
        #             ),
        #             pbms.X - vector_transport_to(
        #                 M,
        #                 pbms.p_last_serious,
        #                 get_subgradient(mp, pbms.p_last_serious),
        #                 pbms.p,
        #                 pbms.vector_transport_method,
        #             ),
        #         )
        #     )
        # pbms.μ = norm(
        #     M,
        #     pbms.p,
        #     pbms.X - vector_transport_to(
        #         M,
        #         pbms.p_last_serious,
        #         get_subgradient(mp, pbms.p_last_serious),
        #         pbms.p,
        #         pbms.vector_transport_method,
        #     ),
        # )^2 / 
        # inner(
        #             M,
        #             pbms.p,
        #             inverse_retract(
        #                 M, pbms.p, pbms.p_last_serious, pbms.inverse_retraction_method
        #             ),
        #             pbms.X - vector_transport_to(
        #                 M,
        #                 pbms.p_last_serious,
        #                 get_subgradient(mp, pbms.p_last_serious),
        #                 pbms.p,
        #                 pbms.vector_transport_method,
        #             ),
        #         )
        pbms.μ += pbms.δ * pbms.μ
        # (length(pbms.bundle) < pbms.bundle_size) && (pbms.μ = (π / (4 * i))^-1)
        copyto!(M, pbms.p_last_serious, pbms.p)
    end
    l = length(pbms.bundle)
    push!(pbms.bundle, (copy(M, pbms.p), copy(M, pbms.p, pbms.X)))
    if l == pbms.bundle_size
        deleteat!(pbms.bundle, l - pbms.bundle_size + 1)
    end
    pbms.lin_errors = [
        get_cost(mp, pbms.p_last_serious) - get_cost(mp, qj) + inner(
            M,
            qj,
            vector_transport_to(M, qj, Xj, pbms.p_last_serious, pbms.vector_transport_method),
            inverse_retract(M, pbms.p_last_serious, qj, pbms.inverse_retraction_method),
        ) for (qj, Xj) in pbms.bundle
    ]
    pbms.approx_errors =
        pbms.lin_errors + [
            pbms.η / 2 *
            norm(
                M,
                pbms.p_last_serious,
                inverse_retract(M, pbms.p_last_serious, qj, pbms.inverse_retraction_method),
            )^2 for (qj, Xj) in pbms.bundle
        ]
    return pbms
end
get_solver_result(pbms::ProxBundleMethodState) = pbms.p_last_serious

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
    mp::AbstractManoptProblem, pbms::ProxBundleMethodState, i::Int
)
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    M = get_manifold(mp)
    if -pbms.ν ≤ b.tol && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter -ν = $(-pbms.ν) is less than $(b.tol).\n"
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
