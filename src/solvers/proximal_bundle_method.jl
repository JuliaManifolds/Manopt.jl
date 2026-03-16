@doc """
    ProximalBundleMethodState <: AbstractManoptSolverState

stores option values for a [`proximal_bundle_method`](@ref) solver.

# Fields

* `α`:                        curvature-dependent parameter used to update `η`
* `α₀`:                       initialization value for `α`, used to update `η`
* `approx_errors`:            approximation of the linearization errors at the last serious step
* `bundle`:                   bundle that collects each iterate with the computed subgradient at the iterate
* `bundle_size`:              the maximal size of the bundle
* `c`:                        convex combination of the approximation errors
* `d`:                        descent direction
* `δ`:                        parameter for updating `μ`: if ``δ < 0`` then ``μ = \\log(i + 1)``, else ``μ += δ μ``
* `ε`:                        stepsize-like parameter related to the injectivity radius of the manifold
* `η`:                        curvature-dependent term for updating the approximation errors
$(_fields(:inverse_retraction_method))
* `λ`:                        convex coefficients that solve the subproblem
* `m`:                        the parameter to test the decrease of the cost
* `μ`:                        (initial) proximal parameter for the subproblem
* `ν`:                        the stopping parameter given by ``ν = - μ |d|^2 - c``
$(_fields(:p; add_properties = [:as_Iterate]))
* `p_last_serious`:           last serious iterate
$(_fields(:retraction_method))
$(_fields(:stopping_criterion; name = "stop"))
* `transported_subgradients`: subgradients of the bundle that are transported to `p_last_serious`
$(_fields(:vector_transport_method))
$(_fields(:X; add_properties = [:as_Subgradient]))
$(_fields([:sub_problem, :sub_state]))

# Constructor

    ProximalBundleMethodState(M::AbstractManifold, sub_problem, sub_state; kwargs...)
    ProximalBundleMethodState(M::AbstractManifold, sub_problem=proximal_bundle_method_subsolver; evaluation=AllocatingEvaluation(), kwargs...)

Generate the state for the [`proximal_bundle_method`](@ref) on the manifold `M`

# Keyword arguments

* `α₀=1.2`
* `bundle_size=50`
* `δ=1.0`
* `ε=1e-2`
* `μ=0.5`
* `m=0.0125`
$(_kwargs([:retraction_method, :inverse_retraction_method]))
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:stopping_criterion; default = "`[`StopWhenLagrangeMultiplierLess`](@ref)`(1e-8)`$(_sc(:Any))[`StopAfterIteration`](@ref)`(5000)"))
$(_kwargs(:sub_problem; default = "`[`proximal_bundle_method_subsolver`](@ref)` "))
$(_kwargs(:sub_state; default = "`[`AllocatingEvaluation`](@ref)` "))
$(_kwargs(:vector_transport_method))
$(_kwargs(:X)) to specify the type of tangent vector to use.
"""
mutable struct ProximalBundleMethodState{
        P,
        T,
        Pr,
        St <: AbstractManoptSolverState,
        R <: Real,
        IR <: AbstractInverseRetractionMethod,
        TR <: AbstractRetractionMethod,
        TSC <: StoppingCriterion,
        VT <: AbstractVectorTransportMethod,
    } <: AbstractManoptSolverState where {P, T, Pr}
    approx_errors::AbstractVector{R}
    bundle::AbstractVector{Tuple{P, T}}
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
            sub_problem::Pr,
            sub_state::St;
            p::P = rand(M),
            m::R = 0.0125,
            inverse_retraction_method::IR = default_inverse_retraction_method(M, typeof(p)),
            retraction_method::TR = default_retraction_method(M, typeof(p)),
            stopping_criterion::SC = StopWhenLagrangeMultiplierLess(1.0e-8) |
                StopAfterIteration(5000),
            bundle_size::Integer = 50,
            vector_transport_method::VT = default_vector_transport_method(M, typeof(p)),
            X::T = zero_vector(M, p),
            α₀::R = 1.2,
            ε::R = 1.0e-2,
            δ::R = 1.0,
            μ::R = 0.5,
        ) where {
            P,
            T,
            Pr <: Union{AbstractManoptProblem, F} where {F},
            St <: AbstractManoptSolverState,
            R <: Real,
            IR <: AbstractInverseRetractionMethod,
            TM <: AbstractManifold,
            TR <: AbstractRetractionMethod,
            SC <: StoppingCriterion,
            VT <: AbstractVectorTransportMethod,
        }
        sub_state_storage = maybe_wrap_evaluation_type(sub_state)
        # Initialize index set, bundle points, linearization errors, and stopping parameter
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
        return new{P, T, Pr, St, R, IR, TR, SC, VT}(
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
            sub_state,
        )
    end
end
function ProximalBundleMethodState(
        M::AbstractManifold,
        sub_problem = proximal_bundle_method_subsolver;
        evaluation::E = AllocatingEvaluation(),
        kwargs...,
    ) where {E <: AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation = evaluation)
    return ProximalBundleMethodState(M, sub_problem, cfs; kwargs...)
end

get_iterate(pbms::ProximalBundleMethodState) = pbms.p_last_serious
function set_iterate!(pbms::ProximalBundleMethodState, M, p)
    copyto!(M, pbms.p_last_serious, p)
    return pbms
end
get_subgradient(pbms::ProximalBundleMethodState) = pbms.d

function status_summary(pbms::ProximalBundleMethodState; context::Symbol = :default)
    (context === :short) && return repr(pbms)
    i = get_count(pbms, :Iterations)
    conv_inl = (i > 0) ? (indicates_convergence(pbms.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the proximal bundle method$(conv_inl)"
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
    $(status_summary(pbms.stop; context = context))
    This indicates convergence: $Conv"""
    return s
end

_doc_PBM_dk = raw"""
```math
d_k = \frac{1}{\mu_k} \sum_{j\in J_k} λ_j^k \mathrm{P}_{p_k←q_j}X_{q_j},
```

with ``X_{q_j} ∈ ∂f(q_j)``, ``p_k`` the last serious iterate,
``\mu_k`` a proximal parameter, and the
``λ_j^k`` as solutions to the quadratic subproblem provided by the
sub solver, see for example the [`proximal_bundle_method_subsolver`](@ref).
"""
_doc_PBM = """
    proximal_bundle_method(M, f, ∂f, p=rand(M), kwargs...)
    proximal_bundle_method!(M, f, ∂f, p, kwargs...)

perform a proximal bundle method ``p^{(k+1)} = $(_tex(:retr))_{p^{(k)}}(-d_k)``,
where ``$(_tex(:retr))`` is a retraction and

$(_doc_PBM_dk)

Though the subdifferential might be set valued, the argument `∂f` should always
return _one_ element from the subdifferential, but not necessarily deterministic.

For more details see [HoseiniMonjeziNobakhtianPouryayevali:2021](@cite).

# Input

$(_args([:M, :f, :subgrad_f, :p]))

# Keyword arguments

* `α₀=1.2`:          initialization value for `α`, used to update `η`
* `bundle_size=50`:  the maximal size of the bundle
* `δ=1.0`:           parameter for updating `μ`: if ``δ < 0`` then ``μ = \\log(i + 1)``, else ``μ += δ μ``
* `ε=1e-2`:          stepsize-like parameter related to the injectivity radius of the manifold
$(_kwargs([:evaluation, :inverse_retraction_method]))
* `m=0.0125`:        a real number that controls the decrease of the cost function
* `μ=0.5`:           initial proximal parameter for the subproblem
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopWhenLagrangeMultiplierLess`](@ref)`(1e-8)`$(_sc(:Any))[`StopAfterIteration`](@ref)`(5000)`"))
$(_kwargs(:sub_problem; default = "`[`proximal_bundle_method_subsolver`](@ref)`"))
$(_kwargs(:sub_state; default = "`[`AllocatingEvaluation`](@ref)` "))
$(_kwargs(:vector_transport_method))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_PBM)"
function proximal_bundle_method(
        M::AbstractManifold, f::TF, ∂f::TdF, p = rand(M); kwargs...
    ) where {TF, TdF}
    keywords_accepted(proximal_bundle_method; kwargs...)
    p_star = copy(M, p)
    return proximal_bundle_method!(M, f, ∂f, p_star; kwargs...)
end
calls_with_kwargs(::typeof(proximal_bundle_method)) = (proximal_bundle_method!,)

@doc "$(_doc_PBM)"
function proximal_bundle_method!(
        M::AbstractManifold,
        f::TF,
        ∂f!!::TdF,
        p;
        m = 0.0125,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        inverse_retraction_method::IR = default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TRetr = default_retraction_method(M, typeof(p)),
        bundle_size = 50,
        stopping_criterion::StoppingCriterion = StopWhenLagrangeMultiplierLess(
            1.0e-8; names = ["-ν"]
        ) | StopAfterIteration(5000),
        vector_transport_method::VTransp = default_vector_transport_method(M, typeof(p)),
        α₀ = 1.2,
        ε = 1.0e-2,
        δ = -1.0, #0.0,
        μ = 0.5, #1.0,
        sub_problem = proximal_bundle_method_subsolver,
        sub_state::Union{AbstractEvaluationType, AbstractManoptSolverState} = evaluation,
        kwargs..., #especially may contain debug
    ) where {TF, TdF, TRetr, IR, VTransp}
    keywords_accepted(proximal_bundle_method!; kwargs...)
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation = evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    sub_state_storage = maybe_wrap_evaluation_type(sub_state)
    pbms = ProximalBundleMethodState(
        M,
        sub_problem,
        maybe_wrap_evaluation_type(sub_state);
        p = p,
        m = m,
        inverse_retraction_method = inverse_retraction_method,
        retraction_method = retraction_method,
        bundle_size = bundle_size,
        stopping_criterion = stopping_criterion,
        vector_transport_method = vector_transport_method,
        α₀ = α₀,
        ε = ε,
        δ = δ,
        μ = μ,
    )
    pbms = decorate_state!(pbms; kwargs...)
    return get_solver_return(solve!(mp, pbms))
end
calls_with_kwargs(::typeof(proximal_bundle_method!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(
        mp::AbstractManoptProblem, pbms::ProximalBundleMethodState{P, T, Pr, St, R}
    ) where {P, T, Pr, St <: AbstractManoptSolverState, R <: Real}
    M = get_manifold(mp)
    copyto!(M, pbms.p_last_serious, pbms.p)
    get_subgradient!(mp, pbms.X, pbms.p)
    pbms.bundle = [(copy(M, pbms.p), copy(M, pbms.p, pbms.X))]
    empty!(pbms.λ)
    push!(pbms.λ, zero(R))
    return pbms
end
function step_solver!(mp::AbstractManoptProblem, pbms::ProximalBundleMethodState, k)
    M = get_manifold(mp)
    pbms.transported_subgradients = [
        (qj ≈ pbms.p_last_serious) ? Xj : vector_transport_to(
                M, qj, Xj, pbms.p_last_serious, pbms.vector_transport_method
            ) + pbms.η * inverse_retract(M, pbms.p_last_serious, qj, pbms.inverse_retraction_method)
            for (qj, Xj) in pbms.bundle
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
    norm_d = norm(M, pbms.p_last_serious, pbms.d)
    pbms.ν = -pbms.μ * norm(M, pbms.p_last_serious, pbms.d)^2 - pbms.c
    if norm_d ≤ pbms.ε
        retract!(M, pbms.p, pbms.p_last_serious, pbms.d, pbms.retraction_method)
        get_subgradient!(mp, pbms.X, pbms.p)
        pbms.α = 0.0
    else
        retract!(
            M, pbms.p, pbms.p_last_serious, pbms.ε * pbms.d / norm_d, pbms.retraction_method
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
        ) / (pbms.ε * norm_d)
    end
    if get_cost(mp, pbms.p) ≤ (get_cost(mp, pbms.p_last_serious) + pbms.m * pbms.ν)
        copyto!(M, pbms.p_last_serious, pbms.p)
        if pbms.δ < zero(eltype(pbms.μ))
            pbms.μ = log(k + 1)
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
# Dispatching on different types of sub solvers
# (a) closed form allocating
function _proximal_bundle_subsolver!(
        M, pbms::ProximalBundleMethodState{P, T, F, ClosedFormSubSolverState{AllocatingEvaluation}}
    ) where {P, T, F}
    pbms.λ = pbms.sub_problem(
        M, pbms.p_last_serious, pbms.μ, pbms.approx_errors, pbms.transported_subgradients
    )
    return pbms
end
# (b) closed form in-place
function _proximal_bundle_subsolver!(
        M, pbms::ProximalBundleMethodState{P, T, F, ClosedFormSubSolverState{InplaceEvaluation}}
    ) where {P, T, F}
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
        mp::AbstractManoptProblem, pbms::ProximalBundleMethodState, k::Int
    )
    if k == 0 # reset on init
        sc.at_iteration = -1
    end
    M = get_manifold(mp)
    if (sc.mode == :estimate) && (-pbms.ν ≤ sc.tolerances[1]) && (k > 0)
        sc.values[1] = -pbms.ν
        sc.at_iteration = k
        return true
    end
    norm_d = norm(M, pbms.p_last_serious, pbms.d)
    if (sc.mode == :both) &&
            (pbms.c ≤ sc.tolerances[1]) &&
            (norm_d ≤ sc.tolerances[2]) &&
            (k > 0)
        sc.values[1] = pbms.c
        sc.values[2] = norm_d
        sc.at_iteration = k
        return true
    end
    return false
end

@doc """
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
        ::AbstractManoptProblem, st::ProximalBundleMethodState, k::Int
    )
    (k < 1) && (return nothing)
    if d.status !== :No
        new_value = -st.ν
        if new_value ≥ d.old_value * d.tol
            @warn """The stopping parameter increased by at least $(d.tol).
            At iteration #$k the stopping parameter -ν increased from $(d.old_value) to $(new_value).\n
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
            At iteration #$k the stopping parameter -ν became negative.\n
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
