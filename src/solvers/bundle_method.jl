@doc raw"""
    BundleMethodState <: AbstractManoptSolverState
stores option values for a [`bundle_method`](@ref) solver

# Fields

* `bundle_points` - collects each iterate `p` with the computed subgradient `∂` at the iterate
* `index_set` - the index set that keeps track of the strictly positive convex coefficients of the subproblem
* `inverse_retraction_method` - the inverse retraction to use within
* `lin_errors` - linearization errors at the last serious step
* `m` - the parameter to test the decrease of the cost
* `p` - current iterate
* `p_last_serious` - last serious iterate
* `retraction_method` – the retraction to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `tol` - the tolerance parameter
* `vector_transport_method` - the vector transport method to use within
* `X` - (`zero_vector(M, p)`) the current element from the possible subgradients at
    `p` that was last evaluated.

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
    S<:Set,
    VT<:AbstractVectorTransportMethod,
} <: AbstractManoptSolverState where {P,T}
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
    function BundleMethodState(
        M::TM,
        p::P;
        m::Real=0.0125,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopAfterIteration(5000),
        X::T=zero_vector(M, p),
        tol::Real=1e-8,
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
    ) where {
        IR<:AbstractInverseRetractionMethod,
        P,
        T,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        VT<:AbstractVectorTransportMethod,
    }
        # Initialize indes set, bundle points, and linearization errors
        index_set = Set(1)
        bundle_points = [(p, X)]
        lin_errors = [0.0]
        return new{IR,typeof(lin_errors),P,T,TR,SC,typeof(index_set),VT}(
            bundle_points,
            inverse_retraction_method,
            lin_errors,
            p,
            deepcopy(p),
            X,
            retraction_method,
            stopping_criterion,
            index_set,
            vector_transport_method,
            m,
            tol,
        )
    end
end
get_iterate(bms::BundleMethodState) = bms.p_last_serious
get_subgradient(bms::BundleMethodState) = bms.X
function set_iterate!(bms::BundleMethodState, M, p)
    copyto!(M, bms.p, p)
    return bms
end

@doc raw"""
    bundle_method(M, f, ∂f, p)

perform a bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)``, 

where ``g_k = \sum_{j\in J_k} λ_j^k \mathrm{P}_{p_k←q_j}X_{q_j}``,

with ``X_{q_j}\in∂f(q_j)``, and

where ``\mathrm{retr}`` is a retraction and `p_k` is the last serious iterate. 
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
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by
   allocation (default) form `∂f(M, q)` or [`MutatingEvaluation`](@ref) in place, i.e. is
   of the form `∂f!(M, X, p)`.
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction method to use
* `retraction` – (`default_retraction_method(M, typeof(p))`) a `retraction(M,p,X)` to use.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(5000)`)
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
    m::Real=0.0125,
    tol::Real=1e-8,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopAfterIteration(5000),
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
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        tol=tol,
        vector_transport_method=vector_transport_method,
    )
    bms = decorate_state!(bms; kwargs...)
    return get_solver_return(solve!(mp, bms))
end
function initialize_solver!(mp::AbstractManoptProblem, bms::BundleMethodState)
    M = get_manifold(mp)
    copyto!(M, bms.p_last_serious, bms.p)
    bms.X = zero_vector(M, bms.p)
    return bms
end
function bundle_method_sub_solver(::Any, ::Any, ::Any)
    throw(
        ErrorException("""Both packages "QuadraticModels" and "RipQP" need to be loaded.""")
    )
end
function step_solver!(mp::AbstractManoptProblem, bms::BundleMethodState, i)
    M = get_manifold(mp)
    transported_subgradients = [
        vector_transport_to(
            M,
            bms.bundle_points[j][1],
            get_subgradient!(mp, bms.bundle_points[j][2], bms.bundle_points[j][1]),
            bms.p_last_serious,
            bms.vector_transport_method,
        ) for j in 1:length(bms.index_set)
    ]
    λ = bundle_method_sub_solver(M, bms, transported_subgradients)
    g = sum(λ .* transported_subgradients)
    ε = sum(λ .* bms.lin_errors)
    # Check transported subgradients ε-inequality
    r = rand(M)
    if (
        get_cost(mp, r) <
        get_cost(mp, bms.p_last_serious) +
        inner(M, bms.p_last_serious, g, log(M, bms.p_last_serious, r)) - ε
    )
        println("No")
        println(r)
        println(bms.p)
        println(bms.p_last_serious)
    end
    δ = -norm(M, bms.p_last_serious, g)^2 - ε
    (δ == 0 || -δ <= bms.tol) && (return bms)
    q = retract(M, bms.p_last_serious, -g, bms.retraction_method)
    X_q = get_subgradient(mp, q)
    if get_cost(mp, q) <= (get_cost(mp, bms.p_last_serious) + bms.m * δ)
        bms.p_last_serious = q
        push!(bms.bundle_points, (bms.p_last_serious, X_q))
    else
        push!(bms.bundle_points, (q, X_q))
    end
    positive_indices = intersect(bms.index_set, Set(findall(j -> j > 0, λ)))
    bms.index_set = union(positive_indices, i + 1)
    bms.lin_errors = [
        get_cost(mp, bms.p_last_serious) - get_cost(mp, bms.bundle_points[j][1]) - inner(
            M,
            bms.bundle_points[j][1],
            bms.bundle_points[j][2],
            inverse_retract(
                M,
                bms.bundle_points[j][1],
                bms.p_last_serious,
                bms.inverse_retraction_method,
            ),
        ) for j in 1:length(bms.index_set)
    ]
    return bms
end
get_solver_result(bms::BundleMethodState) = bms.p_last_serious
