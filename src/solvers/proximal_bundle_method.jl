@doc raw"""
    ProximalBundleMethodState <: AbstractManoptSolverState

stores option values for a [`proximal_bundle_method`](@ref) solver.

# Fields

* `approx_errors`:            approximation of the linearization errors at the last serious step
* `bundle`:                   bundle that collects each iterate with the computed subgradient at the iterate
* `bundle_size`:              (`50`) the maximal size of the bundle
* `c`:                         convex combination of the approximation errors
* `d`:                         descent direction
* `inverse_retraction_method`: the inverse retraction to use within
* `m`:                         (`0.0125`) the parameter to test the decrease of the cost
* `p`:                         current candidate point
* `p_last_serious`:            last serious iterate
* `retraction_method`:         the retraction to use within
* `stop`:                      a [`StoppingCriterion`](@ref)
* `transported_subgradients`:  subgradients of the bundle that are transported to `p_last_serious`
* `vector_transport_method`:   the vector transport method to use within
* `X`:                         (`zero_vector(M, p)`) the current element from the possible subgradients
  at `p` that was last evaluated.
* `α₀`:                        (`1.2`) initialization value for `α`, used to update `η`
* `α`:                         curvature-dependent parameter used to update `η`
* `ε`:                         (`1e-2`) stepsize-like parameter related to the injectivity radius of the manifold
* `δ`:                         parameter for updating `μ`: if ``δ < 0`` then ``μ = \log(i + 1)``, else ``μ += δ μ``
* `η`:                         curvature-dependent term for updating the approximation errors
* `λ`:                         convex coefficients that solve the subproblem
* `μ`:                         (`0.5`) (initial) proximal parameter for the subproblem
* `ν`:                         the stopping parameter given by ``ν = - μ |d|^2 - c``
* `sub_problem`:               a function evaluating with new allocations that solves the sub problem on `M` given the last serious iterate `p_last_serious`, the linearization errors `linearization_errors`, and the transported subgradients `transported_subgradients`,
* `sub_state`:                 an [`AbstractManoptSolverState`](@ref) for the subsolver

# Constructor

    ProximalBundleMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields from before besides `p_last_serious` which obtains the same type as `p`.
You can use for example `X=` to specify the type of tangent vector to use

"""
mutable struct ProximalBundleMethodState{
    P,
    T,
    Pr,
    St<:AbstractManoptSolverState,
    R<:Real,
    IR<:AbstractInverseRetractionMethod,
    TR<:AbstractRetractionMethod,
    TSC<:StoppingCriterion,
    VT<:AbstractVectorTransportMethod,
} <: AbstractManoptSolverState where {P,T,Pr}
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
    sub_problem::Pr
    sub_state::St
    function ProximalBundleMethodState(
        M::TM,
        p::P;
        m::R=0.0125,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenLagrangeMultiplierLess(1e-8) |
                               StopAfterIteration(5000),
        bundle_size::Integer=50,
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
        X::T=zero_vector(M, p),
        α₀::R=1.2,
        ε::R=1e-2,
        δ::R=1.0,
        μ::R=0.5,
        sub_problem::Pr=proximal_bundle_method_subsolver,
        sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState}=AllocatingEvaluation(),
    ) where {
        P,
        T,
        Pr,
        R<:Real,
        IR<:AbstractInverseRetractionMethod,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        VT<:AbstractVectorTransportMethod,
    }
        sub_state_storage = maybe_wrap_evaluation_type(sub_state)
        # Initialize index set, bundle points, linearization errors, and stopping parameter
        approx_errors = [zero(R)]
        bundle = [(copy(M, p), copy(M, p, X))]
        c = zero(R)
        d = copy(M, p, X)
        lin_errors = [zero(R)]
        transported_subgradients = [copy(M, p, X)]
        sub_state_storage = maybe_wrap_evaluation_type(sub_state)
        α = zero(R)
        λ = [zero(R)]
        η = zero(R)
        ν = zero(R)
        return new{P,T,Pr,typeof(sub_state_storage),R,IR,TR,SC,VT}(
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
            sub_problem,
            sub_state_storage,
        )
    end
end
get_iterate(pbms::ProximalBundleMethodState) = pbms.p_last_serious
function set_iterate!(pbms::ProximalBundleMethodState, M, p)
    copyto!(M, pbms.p_last_serious, p)
    return pbms
end
get_subgradient(pbms::ProximalBundleMethodState) = pbms.d

function show(io::IO, pbms::ProximalBundleMethodState)
    i = get_count(pbms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pbms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Proximal Bundle Method
    $Iter

    ## Parameters

    * bundle size:                                $(pbms.bundle_size)
    * inverse retraction:                         $(pbms.inverse_retraction_method)
    * descent test parameter:                     $(pbms.m)
    * retraction:                                 $(pbms.retraction_method)
    * vector transport:                           $(pbms.vector_transport_method)
    * stopping parameter value:                   $(pbms.ν)
    * curvature-dependent α:                      $(pbms.α)
    * stepsize-like parameter ε:                  $(pbms.ε)
    * update parameter for proximal parameter, δ: $(pbms.δ)
    * curvature-dependent η:                      $(pbms.η)
    * proximal parameter μ:                       $(pbms.μ)

    ## Stopping criterion
    $(status_summary(pbms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    proximal_bundle_method(M, f, ∂f, p)

perform a proximal bundle method ``p_{j+1} = \mathrm{retr}(p_k, -d_k)``, where
```math
d_k = \frac{1}{\mu_l} \sum_{j\in J_k} λ_j^k \mathrm{P}_{p_k←q_j}X_{q_j},
```
where ``X_{q_j}\in∂f(q_j)``, ``\mathrm{retr}`` is a retraction,
``p_k`` is the last serious iterate, ``\mu_l`` is a proximal parameter, and the
``λ_j^k`` are solutions to the quadratic subproblem provided by the
[`proximal_bundle_method_subsolver`](@ref).

Though the subdifferential might be set valued, the argument `∂f` should always
return _one_ element from the subdifferential, but not necessarily deterministic.

For more details see [HoseiniMonjeziNobakhtianPouryayevali:2021](@cite).

# Input

* `M`: a manifold ``\mathcal M``
* `f`: a cost function ``F:\mathcal M → ℝ`` to minimize
* `∂f`: the (sub)gradient ``∂ f: \mathcal M → T\mathcal M`` of f
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p`: an initial value ``p ∈ \mathcal M``

# Optional

* `m`: a real number that controls the decrease of the cost function
* `evaluation`: ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by
   allocation (default) form `∂f(M, q)` or [`InplaceEvaluation`](@ref) in place,
   that is it is of the form `∂f!(M, X, p)`.
* `inverse_retraction_method`: (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction method to use
* `retraction`: (`default_retraction_method(M, typeof(p))`) a `retraction(M, p, X)` to use.
* `stopping_criterion`: ([`StopWhenLagrangeMultiplierLess`](@ref)`(1e-8)`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
* `vector_transport_method`: (`default_vector_transport_method(M, typeof(p))`) a vector transport method to use

and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
function proximal_bundle_method(
    M::AbstractManifold, f::TF, ∂f::TdF, p; kwargs...
) where {TF,TdF}
    p_star = copy(M, p)
    return proximal_bundle_method!(M, f, ∂f, p_star; kwargs...)
end
@doc raw"""
    proximal_bundle_method!(M, f, ∂f, p)

perform a proximal bundle method ``p_{j+1} = \mathrm{retr}(p_k, -d_k)`` in place of `p`

# Input

* `M`:  a manifold ``\mathcal M``
* `f`:  a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`: the (sub)gradient ``\partial f:\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p`:  an initial value ``p_0=p ∈ \mathcal M``

for more details and all optional parameters, see [`proximal_bundle_method`](@ref).
"""
function proximal_bundle_method!(
    M::AbstractManifold,
    f::TF,
    ∂f!!::TdF,
    p;
    m=0.0125,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    bundle_size=50,
    stopping_criterion::StoppingCriterion=StopWhenLagrangeMultiplierLess(
        1e-8; names=["-ν"]
    ) | StopAfterIteration(5000),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    α₀=1.2,
    ε=1e-2,
    δ=-1.0,#0.0,
    μ=0.5,#1.0,
    sub_problem=proximal_bundle_method_subsolver,
    sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState}=evaluation,
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    sub_state_storage = maybe_wrap_evaluation_type(sub_state)
    pbms = ProximalBundleMethodState(
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
        sub_problem=sub_problem,
        sub_state=sub_state_storage,
    )
    pbms = decorate_state!(pbms; kwargs...)
    return get_solver_return(solve!(mp, pbms))
end
function initialize_solver!(
    mp::AbstractManoptProblem, pbms::ProximalBundleMethodState{P,T,Pr,St,R}
) where {P,T,Pr,St<:AbstractManoptSolverState,R<:Real}
    M = get_manifold(mp)
    copyto!(M, pbms.p_last_serious, pbms.p)
    get_subgradient!(mp, pbms.X, pbms.p)
    pbms.bundle = [(copy(M, pbms.p), copy(M, pbms.p, pbms.X))]
    empty!(pbms.λ)
    push!(pbms.λ, zero(R))
    return pbms
end
function step_solver!(mp::AbstractManoptProblem, pbms::ProximalBundleMethodState, i)
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
    _proximal_bundle_subsolver!(M, pbms)
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
get_solver_result(pbms::ProximalBundleMethodState) = pbms.p_last_serious

#
#
# Dispatching on different types of subsolvers
# (a) closed form allocating
function _proximal_bundle_subsolver!(
    M, pbms::ProximalBundleMethodState{P,T,F,ClosedFormSubSolverState{AllocatingEvaluation}}
) where {P,T,F}
    pbms.λ = pbms.sub_problem(
        M, pbms.p_last_serious, pbms.μ, pbms.approx_errors, pbms.transported_subgradients
    )
    return pbms
end
# (b) closed form in-place
function _proximal_bundle_subsolver!(
    M, pbms::ProximalBundleMethodState{P,T,F,ClosedFormSubSolverState{InplaceEvaluation}}
) where {P,T,F}
    pbms.sub_problem(
        M,
        pbms.λ,
        pbms.p_last_serious,
        pbms.μ,
        pbms.approx_errors,
        pbms.transported_subgradients,
    )
    return pbms
end

function (sc::StopWhenLagrangeMultiplierLess)(
    mp::AbstractManoptProblem, pbms::ProximalBundleMethodState, i::Int
)
    if i == 0 # reset on init
        sc.at_iteration = -1
    end
    M = get_manifold(mp)
    if (sc.mode == :estimate) && (-pbms.ν ≤ sc.tolerances[1]) && (i > 0)
        sc.values[1] = -pbms.ν
        sc.at_iteration = i
        return true
    end
    nd = norm(M, pbms.p_last_serious, pbms.d)
    if (sc.mode == :both) &&
        (pbms.c ≤ sc.tolerances[1]) &&
        (nd ≤ sc.tolerances[2]) &&
        (i > 0)
        sc.values[1] = pbms.c
        sc.values[2] = nd
        sc.at_iteration = i
        return true
    end
    return false
end

@doc raw"""
    DebugWarnIfLagrangeMultiplierIncreases <: DebugAction

print a warning if the stopping parameter of the bundle method increases.

# Constructor
    DebugWarnIfLagrangeMultiplierIncreases(warn=:Once; tol=1e2)

Initialize the warning to warning level (`:Once`) and introduce a tolerance for the test of `1e2`.

The `warn` level can be set to `:Once` to only warn the first time the cost increases,
to `:Always` to report an increase every time it happens, and it can be set to `:No`
to deactivate the warning, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always:`
"""
function (d::DebugWarnIfLagrangeMultiplierIncreases)(
    ::AbstractManoptProblem, st::ProximalBundleMethodState, i::Int
)
    (i < 1) && (return nothing)
    if d.status !== :No
        new_value = -st.ν
        if new_value ≥ d.old_value * d.tol
            @warn """The stopping parameter increased by at least $(d.tol).
            At iteration #$i the stopping parameter -ν increased from $(d.old_value) to $(new_value).\n
            Consider changing either the initial proximal parameter `μ`, its update coefficient `δ`, or
            the stepsize-like parameter `ε` related to the invectivity radius of the manifold in the
            `proximal_bundle_method` call.
            """
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfLagrangeMultiplierIncreases(:Always) to get all warnings."
                d.status = :No
            end
        elseif new_value < zero(number_eltype(st.ν))
            @warn """The stopping parameter is negative.
            At iteration #$i the stopping parameter -ν became negative.\n
            Consider changing either the initial proximal parameter `μ`, its update coefficient `δ`, or
            the stepsize-like parameter `ε` related to the invectivity radius of the manifold in the
            `proximal_bundle_method` call.
            """
        else
            d.old_value = min(d.old_value, new_value)
        end
    end
    return nothing
end
