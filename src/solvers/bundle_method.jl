@doc raw"""
    BundleMethodState <: AbstractManoptSolverState
stores option values for a [`bundle_method`](@ref) solver

# Fields

* `bundle` - bundle that collects each iterate with the computed subgradient at the iterate
* `index_set` - the index set that keeps track of the strictly positive convex coefficients of the subproblem
* `inverse_retraction_method` - the inverse retraction to use within
* `lin_errors` - linearization errors at the last serious step
* `m` - the parameter to test the decrease of the cost
* `p` - current iterate
* `p_last_serious` - last serious iterate
* `retraction_method` – the retraction to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `vector_transport_method` - the vector transport method to use within
* `X` - (`zero_vector(M, p)`) the current element from the possible subgradients at
    `p` that was last evaluated.
* `ξ` - the stopping parameter given by ξ = -\norm{g}^2 - ε

# Constructor

BundleMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields above besides `p_last_serious` which obtains the same type as `p`.
You can use e.g. `X=` to specify the type of tangent vector to use

"""
mutable struct BundleMethodState{
    IR<:AbstractInverseRetractionMethod,
    L<:Array,
    P,
    T,
    TR<:AbstractRetractionMethod,
    TSC<:StoppingCriterion,
    VT<:AbstractVectorTransportMethod,
    R<:Real,
} <: AbstractManoptSolverState where {P,T}
    bundle::AbstractVector{Tuple{P,T}}
    inverse_retraction_method::IR
    lin_errors::L
    p::P
    p_last_serious::P
    X::T
    retraction_method::TR
    stop::TSC
    vector_transport_method::VT
    m::R
    ξ::R
    diam::R
    λ::AbstractVector{R}
    g::T
    ε::R
    transported_subgradients::AbstractVector{T}
    function BundleMethodState(
        M::TM,
        p::P;
        m::R=0.0125,
        diam::R=1.0,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenBundleLess(1e-8),
        X::T=zero_vector(M, p),
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
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
        bundle = [(copy(M, p), copy(M, p, X))]
        lin_errors = [0.0]
        ξ = 0.0
        λ = [1.0]
        g = copy(M, p, X)
        ε = 0.0
        transported_subgradients = [copy(M, p, X)]
        return new{IR,typeof(lin_errors),P,T,TR,SC,VT,R}(
            bundle,
            inverse_retraction_method,
            lin_errors,
            p,
            copy(M, p),
            X,
            retraction_method,
            stopping_criterion,
            vector_transport_method,
            m,
            ξ,
            diam,
            λ,
            g,
            ε,
            transported_subgradients,
        )
    end
end
get_iterate(bms::BundleMethodState) = bms.p_last_serious
get_subgradient(bms::BundleMethodState) = bms.g

@doc raw"""
    bundle_method(M, f, ∂f, p)

perform a bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)``,

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
* `retraction` – (`default_retraction_method(M, typeof(p))`) a `retraction(M,p,X)` to use.
* `stopping_criterion` – ([`StopWhenBundleLess`](@ref)`(1e-8)`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
* `vector_transport_method` - (`default_vector_transport_method(M, typeof(p))`) a vector transport method to use
...
and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
function bundle_method(M::AbstractManifold, f::TF, ∂f::TdF, p; kwargs...) where {TF,TdF}
    p_star = copy(M, p)
    return bundle_method!(M, f, ∂f, p_star; kwargs...)
end
@doc raw"""
    bundle_method!(M, f, ∂f, p)

perform a bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)`` in place of `p`

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`- the (sub)gradient ``\partial f:\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

for more details and all optional parameters, see [`bundle_method`](@ref).
"""
function bundle_method!(
    M::AbstractManifold,
    f::TF,
    ∂f!!::TdF,
    p;
    m=0.0125,
    diam=1.0,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopWhenBundleLess(1e-8),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    bms = BundleMethodState(
        M,
        p;
        m=m,
        diam=diam,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        vector_transport_method=vector_transport_method,
    )
    bms = decorate_state!(bms; kwargs...)
    return get_solver_return(solve!(mp, bms))
end
function initialize_solver!(mp::AbstractManoptProblem, bms::BundleMethodState)
    M = get_manifold(mp)
    copyto!(M, bms.p_last_serious, bms.p)
    bms.X = get_subgradient(mp, bms.p)
    bms.g = copy(M, bms.p_last_serious, bms.X)
    bms.bundle = [(copy(M, bms.p), copy(M, bms.p, bms.X))]
    return bms
end
function bundle_method_sub_solver(::Any, ::Any, ::Any)
    throw(
        ErrorException("""Both packages "QuadraticModels" and "RipQP" need to be loaded.""")
    )
end
function step_solver!(mp::AbstractManoptProblem, bms::BundleMethodState, i)
    M = get_manifold(mp)
    bms.transported_subgradients = [
        vector_transport_to(M, qj, Xj, bms.p_last_serious, bms.vector_transport_method) for
        (qj, Xj) in bms.bundle
    ]
    bms.λ = bundle_method_sub_solver(M, bms)
    bms.g .= sum(bms.λ .* bms.transported_subgradients)
    bms.ε = sum(bms.λ .* bms.lin_errors)
    bms.ξ = -norm(M, bms.p_last_serious, bms.g)^2 - bms.ε
    retract!(M, bms.p, bms.p_last_serious, -bms.g, bms.retraction_method)
    get_subgradient!(mp, bms.X, bms.p)
    if get_cost(mp, bms.p) ≤ (get_cost(mp, bms.p_last_serious) + bms.m * bms.ξ)
        copyto!(M, bms.p_last_serious, bms.p)
    end
    push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
    deleteat!(bms.bundle, findall(λj -> λj ≤ √eps(Float64), bms.λ))
    bms.lin_errors = [
        get_cost(mp, bms.p_last_serious) - get_cost(mp, qj) - inner(
            M,
            qj,
            Xj,
            inverse_retract(M, qj, bms.p_last_serious, bms.inverse_retraction_method),
        ) +
        bms.diam *
        sqrt(
            2 * norm(
                M,
                qj,
                inverse_retract(M, qj, bms.p_last_serious, bms.inverse_retraction_method),
            ),
        ) *
        norm(M, qj, Xj) for (qj, Xj) in bms.bundle
    ]
    for j in 1:length(bms.lin_errors)
    #     if bms.lin_errors[j] ≤ -√eps(Float64)
    #         #@warn "($i) Entry $j in the linearization error vector is out of bounds: $(bms.lin_errors[j])"
    #     end
        if √eps(Float64) ≥ bms.lin_errors[j] ≥ -√eps(Float64)
            bms.lin_errors[j] = 0.
        end
        # if bms.lin_errors[j] < 0.
        #     bms.lin_errors[j] = -bms.lin_errors[j]
        # end
        # if √eps(Float64) ≥ bms.lin_errors[j]
        #     bms.lin_errors[j] = 0.
        # end
    end
    # bms.lin_errors = max.(bms.lin_errors, Ref(0.))
    return bms
end
get_solver_result(bms::BundleMethodState) = bms.p_last_serious

"""
    StopWhenBundleLess <: StoppingCriterion

A stopping criterion for [`bundle_method`](@ref) to indicate to stop when

* the parameter ξ = -|g|² - ε

is less than a given tolerance tol.

# Constructor

    StopWhenBundleLess(tol::Real=1e-8)

"""
mutable struct StopWhenBundleLess{T<:Real} <: StoppingCriterion
    tol::T
    reason::String
    at_iteration::Int
    function StopWhenBundleLess(tol::Real=1e-8)
        return new{typeof(tol)}(tol, "", 0)
    end
end
function (b::StopWhenBundleLess)(mp::AbstractManoptProblem, bms::BundleMethodState, i::Int)
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    if -bms.ξ ≤ b.tol && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter -ξ = $(-bms.ξ) is less than $(b.tol).\n"
        b.at_iteration = i
        return true
    end
    return false
end
function status_summary(b::StopWhenBundleLess)
    has_stopped = length(b.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Stopping parameter: -ξ ≤ $(b.tol):\t$s"
end
function show(io::IO, b::StopWhenBundleLess)
    return print(io, "StopWhenBundleLess($(b.tol)\n    $(status_summary(b))")
end
