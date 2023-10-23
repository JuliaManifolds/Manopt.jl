@doc raw"""
    ProxBundleMethodState <: AbstractManoptSolverState
stores option values for a [`prox_bundle_method`](@ref) solver

# Fields

* `approx_errors` - approximation of the linearization errors at the last serious step
* `bundle` - bundle that collects each iterate with the computed subgradient at the iterate
* `bundle_size` - (50) the size of the bundle
* `c` - convex combination of the approximation errors
* `d`- descent direction
* `inverse_retraction_method` - the inverse retraction to use within
* `m` - (0.0125) the parameter to test the decrease of the cost
* `p` - current candidate point
* `p_last_serious` - last serious iterate
* `retraction_method` – the retraction to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `transported_subgradients` - subgradients of the bundle that are transported to p_last_serious
* `vector_transport_method` - the vector transport method to use within
* `X` - (`zero_vector(M, p)`) the current element from the possible subgradients at
`p` that was last evaluated.
* `α₀` - (1.2) initalization value for α, used to update η
* `α` - curvature-dependent parameter used to update η
* `ε` - (1e-2) stepsize-like parameter related to the injectivity radius of the manifold
* `δ` - parameter for updating μ: if δ < 0 then μ = log(i + 1), else μ += δ * μ
* `η` - curvature-dependent term for updating the approximation errors
* `λ` - convex coefficients that solve the subproblem
* `μ` - (0.5) (initial) proximal parameter for the subproblem
* `ν` - the stopping parameter given by ν = - μ * |d|^2 - c

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
        stopping_criterion::SC=StopWhenProxBundleLess(1e-8) | StopAfterIteration(5000),
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
        approx_errors = [zero(R)]
        bundle = [(copy(M, p), copy(M, p, X))]
        c = zero(R)
        d = copy(M, p, X)
        lin_errors = [zero(R)]
        transported_subgradients = [copy(M, p, X)]
        α = zero(R)
        λ = [zero(R)]
        η = zero(R)
        ν = zero(R)
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

function show(io::IO, pbms::ProxBundleMethodState)
    i = get_count(pbms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pbms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Convex Bundle Method
    $Iter
    ## Parameters
    * bundle size: $(pbms.bundle_size)
    * inverse retraction: $(pbms.inverse_retraction_method)
    * descent test parameter: $(pbms.m)
    * retraction: $(pbms.retraction_method)
    * vector transport: $(pbms.vector_transport_method)
    * stopping parameter value: $(pbms.ν)
    * curvature-dependent α: $(pbms.α)
    * stepsize-like parameter ε: $(pbms.ε)
    * update parameter for proximal parameter, δ: $(pbms.δ)
    * curvature-dependent η: $(pbms.η)
    * proximal parameter μ: $(pbms.μ)

    ## Stopping Criterion
    $(status_summary(pbms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    prox_bundle_method(M, f, ∂f, p)

perform a proximal bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)``,

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
    stopping_criterion::StoppingCriterion=StopWhenProxBundleLess(1e-8) |
                                          StopAfterIteration(5000),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    α₀=1.2,
    ε=1e-2,
    δ=-1.0,#0.0,
    μ=0.5,#1.0,
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
    pbms.bundle = [(copy(M, pbms.p), copy(M, pbms.p, pbms.X))]
    return pbms
end
function step_solver!(mp::AbstractManoptProblem, pbms::ProxBundleMethodState, i)
    M = get_manifold(mp)
    pbms.transported_subgradients = [
        if qj ≈ pbms.p_last_serious
            Xj
        else
            vector_transport_to(
                M, qj, Xj, pbms.p_last_serious, pbms.vector_transport_method
            ) +
            pbms.η *
            inverse_retract(M, pbms.p_last_serious, qj, pbms.inverse_retraction_method)
        end for (qj, Xj) in pbms.bundle
    ]
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
    pbms.λ = bundle_method_sub_solver(M, pbms)
    pbms.c = sum(pbms.λ .* pbms.approx_errors)
    pbms.d .= -1 / pbms.μ .* sum(pbms.λ .* pbms.transported_subgradients)
    nd = norm(M, pbms.p_last_serious, pbms.d)
    pbms.ν = -pbms.μ * norm(M, pbms.p_last_serious, pbms.d)^2 - pbms.c
    if nd ≤ pbms.ε
        retract!(M, pbms.p, pbms.p_last_serious, pbms.d, pbms.retraction_method)
        get_subgradient!(mp, pbms.X, pbms.p)
        pbms.α = 0.0
    else
        retract!(
            M, pbms.p, pbms.p_last_serious, pbms.ε * pbms.d / nd, pbms.retraction_method
        )
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
        copyto!(M, pbms.p_last_serious, pbms.p)
        if pbms.δ < zero(eltype(pbms.μ))
            pbms.μ = log(i + 1)
        else
            pbms.μ += pbms.δ * pbms.μ
        end
        pbms.bundle = [(copy(M, pbms.p), copy(M, pbms.p, pbms.X))]
        pbms.lin_errors = [0.0]
        pbms.approx_errors = [0.0]
    else
        push!(pbms.bundle, (copy(M, pbms.p), copy(M, pbms.p, pbms.X)))
        push!(
            pbms.lin_errors,
            get_cost(mp, pbms.p_last_serious) - get_cost(mp, pbms.p) + inner(
                M,
                pbms.p_last_serious,
                vector_transport_to(
                    M, pbms.p, pbms.X, pbms.p_last_serious, pbms.vector_transport_method
                ),
                inverse_retract(
                    M, pbms.p_last_serious, pbms.p, pbms.inverse_retraction_method
                ),
            ),
        )
        push!(
            pbms.approx_errors,
            get_cost(mp, pbms.p_last_serious) - get_cost(mp, pbms.p) +
            inner(
                M,
                pbms.p_last_serious,
                vector_transport_to(
                    M, pbms.p, pbms.X, pbms.p_last_serious, pbms.vector_transport_method
                ),
                inverse_retract(
                    M, pbms.p_last_serious, pbms.p, pbms.inverse_retraction_method
                ),
            ) +
            pbms.η / 2 *
            norm(
                M,
                pbms.p_last_serious,
                inverse_retract(
                    M, pbms.p_last_serious, pbms.p, pbms.inverse_retraction_method
                ),
            )^2,
        )
        if length(pbms.bundle) == pbms.bundle_size
            deleteat!(pbms.bundle, 1)
            deleteat!(pbms.lin_errors, 1)
            deleteat!(pbms.approx_errors, 1)
        end
    end
    return pbms
end
get_solver_result(pbms::ProxBundleMethodState) = pbms.p_last_serious

"""
    StopWhenProxBundleLess <: StoppingCriterion

Two stopping criteria for [`prox_bundle_method`](@ref) to indicate to stop when either

    * the parameters c and |d|

    are less than given tolerances tolc and told respectively, or

    * the parameter -ν = -max{−c^k_j +  (ξ^k_j,d) }.

    is less than a given tolerance tolν.

# Constructors

    StopWhenProxBundleLess(tolc=1e-6, told=1e-3)

    StopWhenProxBundleLess(tolν=1e-6)

"""
mutable struct StopWhenProxBundleLess{T,R} <: StoppingCriterion
    tolc::T
    told::T
    tolν::R
    reason::String
    at_iteration::Int
    function StopWhenProxBundleLess(tolc::T, told::T) where {T}
        return new{T,Nothing}(tolc, told, nothing, "", 0)
    end
    function StopWhenProxBundleLess(tolν::R=1e-6) where {R}
        return new{Nothing,R}(nothing, nothing, tolν, "", 0)
    end
end
function (b::StopWhenProxBundleLess{T,Nothing})(
    mp::AbstractManoptProblem, pbms::ProxBundleMethodState, i::Int
) where {T}
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    M = get_manifold(mp)
    if (pbms.c ≤ b.tolc && norm(M, pbms.p_last_serious, pbms.d) ≤ b.told) && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter c = $(pbms.c) is less than $(b.tolc) and |d| = $(norm(M, pbms.p_last_serious, pbms.d)) is less than $(b.told).\n"
        b.at_iteration = i
        return true
    end
    return false
end
function (b::StopWhenProxBundleLess{Nothing,R})(
    mp::AbstractManoptProblem, pbms::ProxBundleMethodState, i::Int
) where {R}
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    M = get_manifold(mp)
    if -pbms.ν ≤ b.tolν && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter -ν = $(-pbms.ν) is less than $(b.tolν).\n"
        b.at_iteration = i
        return true
    end
    return false
end
function status_summary(b::StopWhenProxBundleLess{T,Nothing}) where {T}
    s = length(b.reason) > 0 ? "reached" : "not reached"
    return "Stopping parameter: c ≤ $(b.tolc), |d| ≤ $(b.told):\t$s"
end
function status_summary(b::StopWhenProxBundleLess{Nothing,R}) where {R}
    s = length(b.reason) > 0 ? "reached" : "not reached"
    return "Stopping parameter: -ν ≤ $(b.tolν):\t$s"
end
function show(io::IO, b::StopWhenProxBundleLess{T,Nothing}) where {T}
    return print(
        io, "StopWhenProxBundleLess($(b.tolc), $(b.told))\n    $(status_summary(b))"
    )
end
function show(io::IO, b::StopWhenProxBundleLess{Nothing,R}) where {R}
    return print(io, "StopWhenProxBundleLess($(b.tolν))\n    $(status_summary(b))")
end

function (d::DebugWarnIfStoppingParameterIncreases)(
    p::AbstractManoptProblem, st::ProxBundleMethodState, i::Int
)
    (i < 1) && (return nothing)
    if d.status !== :No
        new_value = -st.ν
        if new_value ≥ d.old_value * d.tol
            @warn """The stopping parameter increased by at least $(d.tol).
            At iteration #$i the stopping parameter -ν increased from $(d.old_value) to $(new_value).\n
            Consider changing either the initial proximal parameter `μ`, its update coefficient `δ`, or
            the stepsize-like parameter `ε` related to the invectivity radius of the manifold in the 
            `prox_bundle_method` call.
            """
            if d.status === :Once
                @warn "Further warnings will be supressed, use DebugWarnIfStoppingParameterIncreases(:Always) to get all warnings."
                d.status = :No
            end
        elseif new_value < zero(number_eltype(st.ν))
            @warn """The stopping parameter is negative.
            At iteration #$i the stopping parameter -ν became negative.\n
            Consider changing either the initial proximal parameter `μ`, its update coefficient `δ`, or
            the stepsize-like parameter `ε` related to the invectivity radius of the manifold in the 
            `prox_bundle_method` call.
            """
        else
            d.old_value = min(d.old_value, new_value)
        end
    end
    return nothing
end
