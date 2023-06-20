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
* `retraction` – (`default_retraction_method(M, typeof(p))`) a `retraction(M, p, X)` to use.
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
    bundle_size=50,
    m=1e-2,
    diam=1.0,
    filter1=eps(),
    filter2=eps(),
    δ=1.0,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopWhenBundleLess(1e-4),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    bms = BundleMethodState(
        M,
        p;
        bundle_size=bundle_size,
        m=m,
        diam=diam,
        filter1=filter1,
        filter2=filter2,
        δ=δ,
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
    get_subgradient!(mp, bms.X, bms.p)
    copyto!(M, bms.g, bms.p_last_serious, bms.X)
    bms.bundle[1] = (copy(M, bms.p), copy(M, bms.p, bms.X))
    return bms
end
function bundle_method_sub_solver(::Any, ::Any)
    throw(
        ErrorException("""Both packages "QuadraticModels" and "RipQP" need to be loaded.""")
    )
end
function step_solver!(mp::AbstractManoptProblem, bms::BundleMethodState, i)
    M = get_manifold(mp)
    bms.transported_subgradients = [
        vector_transport_to(
            M,
            bms.bundle[l][1],
            bms.bundle[l][2],
            bms.p_last_serious,
            bms.vector_transport_method,
        ) for l in bms.indices if l != 0
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
    bms.j = mod1(i, bms.bundle_size)
    bms.p0 .= bms.bundle[bms.indices[1]][1]
    if i ≤ bms.bundle_size
        bms.indices[i] = i
        bms.indices_ref[i] = i
    else
        circshift!(bms.indices, bms.indices_ref, -bms.j)
    end
    copyto!(M, bms.bundle[bms.j][1], bms.p)
    copyto!(M, bms.bundle[bms.j][2], bms.p, bms.X)
    if bms.indices[2] != 0
        bms.diam = max(0.0, bms.diam - bms.δ * distance(M, bms.bundle[bms.indices[2]][1], bms.p0))
    end
    # v = findall(λj -> λj ≤ bms.filter1, bms.λ)
    # if !isempty(v)
    #     k = findfirst(x -> x == minimum(bms.indices), bms.indices)
    #     y = copy(M, bms.bundle[k][1])
    #     bms.indices[v] = 0
    #     if k < bms.bundle_size
    #         bms.diam = max(0.0, bms.diam - bms.δ * distance(M, bms.bundle[k + 1][1], y))
    #     else
    #         bms.diam = max(0.0, bms.diam - bms.δ * distance(M, bms.bundle[1][1], y))
    #     end
    # end
    bms.lin_errors = [
        get_cost(mp, bms.p_last_serious) - get_cost(mp, bms.bundle[l][1]) - inner(
            M,
            bms.bundle[l][1],
            bms.bundle[l][2],
            inverse_retract(
                M, bms.bundle[l][1], bms.p_last_serious, bms.inverse_retraction_method
            ),
        ) +
        bms.diam *
        sqrt(
            2 * norm(
                M,
                bms.bundle[l][1],
                inverse_retract(
                    M, bms.bundle[l][1], bms.p_last_serious, bms.inverse_retraction_method
                ),
            ),
        ) *
        norm(M, bms.bundle[l][1], bms.bundle[l][2]) for l in bms.indices if l != 0
    ]
    return bms
end
get_solver_result(bms::BundleMethodState) = bms.p_last_serious

"""
    StopWhenBundleLess <: StoppingCriterion

Two stopping criteria for [`bundle_method`](@ref) to indicate to stop when either

* the parameters ε and |g|

are less than given tolerances tole and tolg respectively, or

* the parameter -ξ = - |g|^2 - ε

is less than a given tolerance tolxi.

# Constructors

    StopWhenBundleLess(tole=1e-4, tolg=1e-2)

    StopWhenBundleLess(tolxi=1e-4)

"""
mutable struct StopWhenBundleLess{T,R} <: StoppingCriterion
    tole::T
    tolg::T
    tolxi::R
    reason::String
    at_iteration::Int
    function StopWhenBundleLess{Real,Nothing}(tole=1e-4, tolg=1e-2)
        return new{typeof(tole),Nothing}(tole, tolg, nothing, "", 0)
    end
    function StopWhenBundleLess(tole::Real, tolg::Real)
        return StopWhenBundleLess{Real,Nothing}(tole, tolg)
    end
    function StopWhenBundleLess{Nothing,Real}(tolxi=1e-4)
        return new{Nothing,typeof(tolxi)}(nothing, nothing, tolxi, "", 0)
    end
    StopWhenBundleLess(tolxi::Real) = StopWhenBundleLess{Nothing,Real}(tolxi)
end
function (b::StopWhenBundleLess)(mp::AbstractManoptProblem, bms::BundleMethodState, i::Int)
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    M = get_manifold(mp)
    if b.tolxi == nothing
        if (bms.ε ≤ b.tole && norm(M, bms.p_last_serious, bms.g) ≤ b.tolg) && i > 0
            b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter ε = $(bms.ε) is less than $(b.tole) and |g| = $(norm(M, bms.p_last_serious, bms.g)) is less than $(b.tolg).\n"
            b.at_iteration = i
            return true
        end
    elseif -bms.ξ ≤ b.tolxi && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter -ξ = $(-bms.ξ) is less than $(b.tolxi).\n"
        b.at_iteration = i
        return true
    end
    return false
end
function status_summary(b::StopWhenBundleLess)
    has_stopped = length(b.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    if b.tolxi == nothing
        return "Stopping parameter: ε ≤ $(b.tole), |g| ≤ $(b.tolg):\t$s"
    else
        return "Stopping parameter: -ξ ≤ $(b.tolxi):\t$s"
    end
end
function show(io::IO, b::StopWhenBundleLess)
    if b.tolxi == nothing
        return print(
            io, "StopWhenBundleLess($(b.tole), $(b.tolg)\n    $(status_summary(b))"
        )
    else
        return print(io, "StopWhenBundleLess($(b.tol)\n    $(status_summary(b))")
    end
end
