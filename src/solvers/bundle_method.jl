@doc raw"""
    BundleMethodState <: AbstractManoptSolverState
stores option values for a [`bundle_method`](@ref) solver

# Fields

* `atol_λ` - tolerance parameter for the convex coefficients in λ
* `atol_errors` - tolerance parameter for the linearization errors
* `bundle` - bundle that collects each iterate with the computed subgradient at the iterate
* `bundle_size` - (25) the size of the bundle
* `diam` - (50.0) estimate for the diameter of the level set of the objective function at the starting point
* `g`- descent direction
* `inverse_retraction_method` - the inverse retraction to use within
* `j` - index to cycle through the bundle given by mod1(iteration, bundle_size)
* `lin_errors` - linearization errors at the last serious step
* `m` - the parameter to test the decrease of the cost
* `p` - current candidate point
* `p_last_serious` - last serious iterate
* `retraction_method` – the retraction to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `transported_subgradients` - subgradients of the bundle that are transported to p_last_serious
* `vector_transport_method` - the vector transport method to use within
* `X` - (`zero_vector(M, p)`) the current element from the possible subgradients at
    `p` that was last evaluated.
* `δ` - update parameter for the diameter
* `ε` - convex combination of the linearization errors
* `λ` - convex coefficients that solve the subproblem
* `ξ` - the stopping parameter given by ξ = -|g|^2 - ε

# Constructor

BundleMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields above besides `p_last_serious` which obtains the same type as `p`.
You can use e.g. `X=` to specify the type of tangent vector to use

"""
mutable struct BundleMethodState{
    R,
    P,
    T,
    A<:AbstractVector{<:R},
    B<:AbstractVector{Tuple{<:P,<:T}},
    C<:AbstractVector{T},
    I,
    D<:AbstractVector{<:I},
    IR<:AbstractInverseRetractionMethod,
    TR<:AbstractRetractionMethod,
    TSC<:StoppingCriterion,
    VT<:AbstractVectorTransportMethod,
} <: AbstractManoptSolverState where {R<:Real,P,T,I<:Int64}
    atol_λ::R
    atol_errors::R
    bundle::B
    bundle_size::I
    diam::R
    g::T
    indices::D
    inverse_retraction_method::IR
    j::I
    lin_errors::A
    m::R
    p::P
    p_last_serious::P
    p0::P
    active_indices::BitVector
    retraction_method::TR
    stop::TSC
    transported_subgradients::C
    vector_transport_method::VT
    X::T
    δ::R
    ε::R
    ξ::R
    λ::A
    function BundleMethodState(
        M::TM,
        p::P;
        atol_λ::R=eps(R),
        atol_errors::R=eps(R),
        bundle_size::Integer=25,
        m::R=1e-2,
        diam::R=1.0,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenBundleLess(1e-8) | StopAfterIteration(5000),
        X::T=zero_vector(M, p),
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
        δ::R=one(R),
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
        bundle = [(copy(M, p), zero_vector(M, p))]
        g = zero_vector(M, p)
        lin_errors = zeros(bundle_size)
        transported_subgradients = [zero_vector(M, p)]
        ε = zero(R)
        λ = zero(R)
        ξ = zero(R)
        return new{
            typeof(m),
            P,
            T,
            typeof(lin_errors),
            typeof(bundle),
            typeof(transported_subgradients),
            typeof(bundle_size),
            typeof(indices),
            IR,
            TR,
            SC,
            VT,
        }(
            atol_λ,
            atol_errors,
            bundle,
            bundle_size,
            diam,
            g,
            inverse_retraction_method,
            j,
            lin_errors,
            m,
            p,
            copy(M, p),
            retraction_method,
            stopping_criterion,
            transported_subgradients,
            vector_transport_method,
            X,
            δ,
            ε,
            ξ,
            λ,
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
Though the subdifferential might be set valued, the argument `∂f` should always
return _one_ element from the subdifferential, but not necessarily deterministic.

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
    atol_λ=eps(),
    atol_errors=eps(),
    bundle_size=25,
    diam=50.0,
    m=1e-3,
    δ=√2,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopWhenBundleLess(1e-8) |
                                          StopAfterIteration(5000),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    bms = BundleMethodState(
        M,
        p;
        atol_λ=atol_λ,
        atol_errors=atol_errors,
        bundle_size=bundle_size,
        diam=diam,
        m=m,
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
    bms.bundle = [(copy(M, bms.p), copy(M, bms.p, bms.X))]
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
        vector_transport_to(M, qj, Xj, bms.p_last_serious, bms.vector_transport_method) for
        (qj, Xj) in bms.bundle
    ]
    bms.λ = bundle_method_sub_solver(M, bms)
    bms.g .= sum(bms.λ .* bms.transported_subgradients)
    bms.ε = sum(bms.λ .* bms.approx_errors)
    bms.ξ = -norm(M, bms.p_last_serious, bms.g)^2 - bms.ε
    retract!(M, bms.p, bms.p_last_serious, -bms.g, bms.retraction_method)
    get_subgradient!(mp, bms.X, bms.p)
    if get_cost(mp, bms.p) ≤ (get_cost(mp, bms.p_last_serious) + bms.m * bms.ξ)
        # bms.diam = max(0.0, bms.diam - bms.δ * distance(M, bms.p_last_serious, bms.p))
        copyto!(M, bms.p_last_serious, bms.p)
    end
    l = length(bms.bundle)
    push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
    v = findall(λj -> λj ≤ bms.atol_λ, bms.λ)
    if !isempty(v)
        y = copy(M, bms.bundle[1][1])
        deleteat!(bms.bundle, v)
        bms.diam = max(0.0, bms.diam - bms.δ * distance(M, bms.bundle[1][1], y))
    end
    if l == bms.bundle_size
        y = copy(M, bms.bundle[1][1])
        deleteat!(bms.bundle, 1)
        bms.diam = max(0.0, bms.diam - bms.δ * distance(M, bms.bundle[1][1], y))
    end
    bms.approx_errors = [
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
    bms.approx_errors = [zero(bms.atol_errors) ≥ x ≥ -bms.atol_errors ? zero(bms.atol_errors) : x for x in bms.approx_errors]
    return bms
end
get_solver_result(bms::BundleMethodState) = bms.p_last_serious

"""
    StopWhenBundleLess <: StoppingCriterion

Two stopping criteria for [`bundle_method`](@ref) to indicate to stop when either

* the parameters ε and |g|

are less than given tolerances tole and tolg respectively, or

* the parameter -ξ = - |g|^2 - ε

is less than a given tolerance tolξ.

# Constructors

    StopWhenBundleLess(tole=1e-6, tolg=1e-3)

    StopWhenBundleLess(tolξ=1e-6)

"""
mutable struct StopWhenBundleLess{T,R} <: StoppingCriterion
    tole::T
    tolg::T
    tolξ::R
    reason::String
    at_iteration::Int
    function StopWhenBundleLess(tole::T, tolg::T) where {T}
        return new{T,Nothing}(tole, tolg, nothing, "", 0)
    end
    function StopWhenBundleLess(tolξ::R=1e-6) where {R}
        return new{Nothing,R}(nothing, nothing, tolξ, "", 0)
    end
end
function (b::StopWhenBundleLess{T,Nothing})(
    mp::AbstractManoptProblem, bms::BundleMethodState, i::Int
) where {T}
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    M = get_manifold(mp)
    if (bms.ε ≤ b.tole && norm(M, bms.p_last_serious, bms.g) ≤ b.tolg) && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter ε = $(bms.ε) is less than $(b.tole) and |g| = $(norm(M, bms.p_last_serious, bms.g)) is less than $(b.tolg).\n"
        b.at_iteration = i
        return true
    end
    return false
end
function (b::StopWhenBundleLess{Nothing,R})(
    mp::AbstractManoptProblem, bms::BundleMethodState, i::Int
) where {R}
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    M = get_manifold(mp)
    if -bms.ξ ≤ b.tolξ && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter -ξ = $(-bms.ξ) is less than $(b.tolξ).\n"
        b.at_iteration = i
        return true
    end
    return false
end
function status_summary(b::StopWhenBundleLess{T,Nothing}) where {T}
    s = length(b.reason) > 0 ? "reached" : "not reached"
    return "Stopping parameter: ε ≤ $(b.tole), |g| ≤ $(b.tolg):\t$s"
end
function status_summary(b::StopWhenBundleLess{Nothing,R}) where {R}
    s = length(b.reason) > 0 ? "reached" : "not reached"
    return "Stopping parameter: -ξ ≤ $(b.tolξ):\t$s"
end
function show(io::IO, b::StopWhenBundleLess{T,Nothing}) where {T}
    return print(io, "StopWhenBundleLess($(b.tole), $(b.tolg))\n    $(status_summary(b))")
end
function show(io::IO, b::StopWhenBundleLess{Nothing,R}) where {R}
    return print(io, "StopWhenBundleLess($(b.tolξ))\n    $(status_summary(b))")
end
