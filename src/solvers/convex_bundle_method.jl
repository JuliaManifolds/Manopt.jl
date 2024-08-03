@doc """
    ConvexBundleMethodState <: AbstractManoptSolverState

Stores option values for a [`convex_bundle_method`](@ref) solver.

# Fields

THe following fields require a (real) number type `R`, as well as
point type `P` and a tangent vector type `T``

* `atol_λ::R`:                 tolerance parameter for the convex coefficients in λ
* `atol_errors::R:             tolerance parameter for the linearization errors
* `bundle<:AbstractVector{Tuple{<:P,<:T}}`: bundle that collects each iterate with the computed subgradient at the iterate
* `bundle_cap::Int`: the maximal number of elements the bundle is allowed to remember
* `diameter::R`: estimate for the diameter of the level set of the objective function at the starting point
* `domain: the domain of ``f`` as a function `(M,p) -> b`that evaluates to true when the current candidate is in the domain of `f`, and false otherwise,
* `g::T`:                      descent direction
* $(_field_inv_retr)
* `k_max::R`:                  upper bound on the sectional curvature of the manifold
* `linearization_errors<:AbstractVector{<:R}`: linearization errors at the last serious step
* `m::R`:                      the parameter to test the decrease of the cost: ``f(q_{k+1}) ≤ f(p_k) + m ξ``.
* $(_field_iterate)
* `p_last_serious::P`:         last serious iterate
* $(_field_retr)
* $(_field_stop)
* `transported_subgradients`:  subgradients of the bundle that are transported to `p_last_serious`
* $(_field_vector_transp)
* $(_field_subgradient)
* $(_field_step)
* `ε::R`:                      convex combination of the linearization errors
* `λ:::AbstractVector{<:R}`:   convex coefficients from the slution of the subproblem
* `ξ`:                         the stopping parameter given by ``ξ = -\\lVert g\\rvert^2 – ε``
* $(_field_sub_problem)
* $(_field_sub_state)

# Constructor

    ConvexBundleMethodState(M::AbstractManifold, p=rand(M); kwargs...)

Generate the state for the [`convex_bundle_method`](@ref) on the manifodl `M`
with initial point `p`.

# Keyword arguments

Most of the following keyword arguments set default values for the fields mentioned before.

* `atol_λ=eps()`
* `atol_errors=eps()`
* `bundle_cap=25``
* `m=1e-2`
* `diameter=50.0`
* `domain=(M, p) -> isfinite(f(M, p))`
* `k_max=0`
* `stepsize=default_stepsize(M, ConvexBundleMethodState)`, which defaults to [`ConstantStepsize`](@ref)`(M)`.
* $(_kw_inverse_retraction_method_default): $(_kw_inverse_retraction_method)
* $(_kw_retraction_method_default): $(_kw_retraction_method)
* `stopping_criterion=`[`StopWhenLagrangeMultiplierLess`](@ref)`(1e-8)`$(_sc_any)[`StopAfterIteration`](@ref)`(5000)`
* `X=`$(_link_zero_vector) specify the type of tangent vector to use.
* $(_kw_vector_transport_method_default): $(_kw_vector_transport_method)
* `sub_problem=`[`convex_bundle_method_subsolver`](@ref)
* `sub_state=[`AllocatingEvaluation`](@ref)

"""
mutable struct ConvexBundleMethodState{
    P,
    T,
    Pr<:Union{F,AbstractManoptProblem} where {F},
    St<:AbstractManoptSolverState,
    R,
    A<:AbstractVector{<:R},
    B<:AbstractVector{Tuple{<:P,<:T}},
    C<:AbstractVector{T},
    D,
    I,
    IR<:AbstractInverseRetractionMethod,
    TR<:AbstractRetractionMethod,
    TS<:Stepsize,
    TSC<:StoppingCriterion,
    VT<:AbstractVectorTransportMethod,
} <: AbstractManoptSolverState where {R<:Real,P,T,I<:Int,Pr}
    atol_λ::R
    atol_errors::R
    bundle::B
    bundle_cap::I
    diameter::R
    domain::D
    g::T
    inverse_retraction_method::IR
    k_max::R
    last_stepsize::R
    linearization_errors::A
    m::R
    p::P
    p_last_serious::P
    retraction_method::TR
    stepsize::TS
    stop::TSC
    transported_subgradients::C
    vector_transport_method::VT
    X::T
    ε::R
    ξ::R
    λ::A
    sub_problem::Pr
    sub_state::St
    ϱ::Nothing# deprecated
    function ConvexBundleMethodState(
        M::TM,
        p::P=rand(M);
        atol_λ::R=eps(),
        atol_errors::R=eps(),
        bundle_cap::I=25,
        m::R=1e-2,
        diameter::R=50.0,
        domain::D=(M, p) -> isfinite(f(M, p)),
        k_max=0,
        stepsize::S=default_stepsize(M, ConvexBundleMethodState),
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenLagrangeMultiplierLess(1e-8) |
                               StopAfterIteration(5000),
        X::T=zero_vector(M, p),
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
        sub_problem::Pr=convex_bundle_method_subsolver,
        sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState}=AllocatingEvaluation(),
        k_size=nothing,# deprecated
        p_estimate=nothing,# deprecated
        ϱ=nothing,# deprecated
    ) where {
        D,
        IR<:AbstractInverseRetractionMethod,
        P,
        T,
        Pr,
        I,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        S<:Stepsize,
        VT<:AbstractVectorTransportMethod,
        R<:Real,
    }
        sub_state_storage = maybe_wrap_evaluation_type(sub_state)
        bundle = Vector{Tuple{P,T}}()
        g = zero_vector(M, p)
        last_stepsize = one(R)
        linearization_errors = Vector{R}()
        transported_subgradients = Vector{T}()
        ε = zero(R)
        λ = Vector{R}()
        ξ = zero(R)
        !all(isnothing.([k_size, p_estimate, ϱ])) &&
            @error "Keyword arguments `k_size`, `p_estimate`, and the field `ϱ` are not used anymore. Use the field `k_max` instead."
        return new{
            P,
            T,
            Pr,
            typeof(sub_state_storage),
            R,
            typeof(linearization_errors),
            typeof(bundle),
            typeof(transported_subgradients),
            D,
            I,
            IR,
            TR,
            S,
            SC,
            VT,
        }(
            atol_λ,
            atol_errors,
            bundle,
            bundle_cap,
            diameter,
            domain,
            g,
            inverse_retraction_method,
            k_max,
            last_stepsize,
            linearization_errors,
            m,
            p,
            copy(M, p),
            retraction_method,
            stepsize,
            stopping_criterion,
            transported_subgradients,
            vector_transport_method,
            X,
            ε,
            ξ,
            λ,
            sub_problem,
            sub_state_storage,
            ϱ,# deprecated
        )
    end
end
get_iterate(bms::ConvexBundleMethodState) = bms.p_last_serious
function set_iterate!(bms::ConvexBundleMethodState, M, p)
    copyto!(M, bms.p_last_serious, p)
    return bms
end
get_subgradient(bms::ConvexBundleMethodState) = bms.g
function default_stepsize(M::AbstractManifold, ::Type{ConvexBundleMethodState})
    return ConstantStepsize(M)
end
function show(io::IO, cbms::ConvexBundleMethodState)
    i = get_count(cbms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(cbms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Convex Bundle Method
    $Iter
    ## Parameters
    * tolerance parameter for the convex coefficients:  $(cbms.atol_λ)
    * tolerance parameter for the linearization errors: $(cbms.atol_errors)
    * bundle cap size:                                  $(cbms.bundle_cap)
    * current bundle size:                              $(length(cbms.bundle))
    * curvature upper bound:                            $(cbms.k_max)
    * descent test parameter:                           $(cbms.m)
    * diameter:                                         $(cbms.diameter)
    * inverse retraction:                               $(cbms.inverse_retraction_method)
    * retraction:                                       $(cbms.retraction_method)
    * Lagrange parameter value:                         $(cbms.ξ)
    * vector transport:                                 $(cbms.vector_transport_method)

    ## Stopping criterion
    $(status_summary(cbms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    DomainBackTrackingStepsize <: Stepsize

Implement a backtrack as long as we are ``q = \operatorname{retr}_p(X)``
yields a point closer to ``p`` than ``\lVert X \rVert_p`` or
``q`` is not on the domain.
For the domain this step size requires a `ConvexBundleMethodState`
"""
mutable struct DomainBackTrackingStepsize <: Manopt.Stepsize
    β::Float64
end
function (dbt::DomainBackTrackingStepsize)(
    amp::AbstractManoptProblem,
    cbms::ConvexBundleMethodState,
    ::Any,
    args...;
    tol=0.0,
    kwargs...,
)
    M = get_manifold(amp)
    t = 1.0
    q = retract(M, cbms.p_last_serious, -t * cbms.g, cbms.retraction_method)
    l = norm(M, cbms.p_last_serious, cbms.g)
    while !cbms.domain(M, q) ||
        (cbms.k_max > 0 && distance(M, cbms.p_last_serious, q) + tol < t * l)
        t *= dbt.β
        retract!(M, q, cbms.p_last_serious, -t * cbms.g, cbms.retraction_method)
    end
    return t
end

_doc_cbm_gk = raw"""
```math
g_k = \sum_{j\in J_k} λ_j^k \mathrm{P}_{p_k←q_j}X_{q_j},
```
"""
_doc_convex_bundle_method = """
    convex_bundle_method(M, f, ∂f, p)
    convex_bundle_method!(M, f, ∂f, p)

perform a convex bundle method ``p_{k+1} = $(_l_retr)_{p_k}(-g_k)`` where

$(_doc_cbm_gk)


and ``p_k`` is the last serious iterate, ``X_{q_j} ∈ ∂f(q_j)``, and the ``λ_j^k`` are solutions
to the quadratic subproblem provided by the [`convex_bundle_method_subsolver`](@ref).

Though the subdifferential might be set valued, the argument `∂f` should always
return one element from the subdifferential, but not necessarily deterministic.

For more details, see [BergmannHerzogJasa:2024](@cite).

# Input

* $(_arg_M)
* $(_arg_f)
* $(_arg_subgrad_f)
* $(_arg_p)

# Keyword arguments

* `atol_λ=eps()` : tolerance parameter for the convex coefficients in ``λ``.
* `atol_errors=eps()`: : tolerance parameter for the linearization errors.
* `bundle_cap=25``
* `m=1e-3`: : the parameter to test the decrease of the cost: ``f(q_{k+1}) ≤ f(p_k) + m ξ``.
* `diameter=50.0`: estimate for the diameter of the level set of the objective function at the starting point.
* `domain=(M, p) -> isfinite(f(M, p))`: a function to that evaluates to true when the current candidate is in the domain of the objective `f`, and false otherwise.
* $(_kw_evaluation_default): $(_kw_evaluation)
* `k_max=0`: upper bound on the sectional curvature of the manifold.
* `stepsize=default_stepsize(M, ConvexBundleMethodState)`, which defaults to [`ConstantStepsize`](@ref)`(M)`.
* $(_kw_inverse_retraction_method_default): $(_kw_inverse_retraction_method)
* $(_kw_retraction_method_default): $(_kw_retraction_method)
* `stopping_criterion=`[`StopWhenLagrangeMultiplierLess`](@ref)`(1e-8)`$(_sc_any)[`StopAfterIteration`](@ref)`(5000)`
* `X=`$(_link_zero_vector) specify the type of tangent vector to use.
* $(_kw_vector_transport_method_default): $(_kw_vector_transport_method)
* `sub_problem=`[`convex_bundle_method_subsolver`](@ref): a Manopt problem or a closed form solution as a function for the sub problem
* `sub_state=[`AllocatingEvaluation`](@ref): specify a solver for the sub problem or how the closed form solution function is evaluated.

$(_kw_others)

$(_doc_sec_output)
"""

@doc "$(_doc_convex_bundle_method)"
function convex_bundle_method(
    M::AbstractManifold, f::TF, ∂f::TdF, p=rand(M); kwargs...
) where {TF,TdF}
    p_star = copy(M, p)
    return convex_bundle_method!(M, f, ∂f, p_star; kwargs...)
end

@doc "$(_doc_convex_bundle_method)"
function convex_bundle_method!(
    M::AbstractManifold,
    f::TF,
    ∂f!!::TdF,
    p;
    atol_λ::R=eps(),
    atol_errors::R=eps(),
    bundle_cap::Int=25,
    diameter::R=π / 3,# was `k_max -> k_max === nothing ? π/2 : (k_max ≤ zero(R) ? typemax(R) : π/3)`,
    domain=(M, p) -> isfinite(f(M, p)),
    m::R=1e-3,
    k_max=0,
    stepsize::Stepsize=DomainBackTrackingStepsize(0.5),
    debug=[DebugWarnIfLagrangeMultiplierIncreases()],
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopWhenLagrangeMultiplierLess(1e-8; names=["-ξ"]), StopAfterIteration(5000)
    ),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    sub_problem=convex_bundle_method_subsolver,
    sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState}=evaluation,
    k_size=nothing,# deprecated
    p_estimate=nothing,# deprecated
    ϱ=nothing,# deprecated
    kwargs..., #especially may contain debug
) where {R<:Real,TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    sub_state_storage = maybe_wrap_evaluation_type(sub_state)
    bms = ConvexBundleMethodState(
        M,
        p;
        atol_λ=atol_λ,
        atol_errors=atol_errors,
        bundle_cap=bundle_cap,
        diameter=diameter,
        domain=domain,
        m=m,
        k_max=k_max,
        stepsize=stepsize,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        vector_transport_method=vector_transport_method,
        sub_problem=sub_problem,
        sub_state=sub_state_storage,
        k_size=k_size,# deprecated
        p_estimate=p_estimate,# deprecated
        ϱ=ϱ,# deprecated
    )
    bms = decorate_state!(bms; debug=debug, kwargs...)
    return get_solver_return(solve!(mp, bms))
end

function initialize_solver!(
    mp::AbstractManoptProblem, bms::ConvexBundleMethodState{P,T,Pr,St,R}
) where {P,T,Pr,St,R}
    M = get_manifold(mp)
    copyto!(M, bms.p_last_serious, bms.p)
    get_subgradient!(mp, bms.X, bms.p)
    copyto!(M, bms.g, bms.p_last_serious, bms.X)
    empty!(bms.bundle)
    push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
    empty!(bms.λ)
    push!(bms.λ, zero(R))
    empty!(bms.linearization_errors)
    push!(bms.linearization_errors, zero(R))
    empty!(bms.transported_subgradients)
    push!(bms.transported_subgradients, zero_vector(M, bms.p))
    return bms
end
function step_solver!(mp::AbstractManoptProblem, bms::ConvexBundleMethodState, i)
    M = get_manifold(mp)
    # Refactor to in-place
    for (j, (qj, Xj)) in enumerate(bms.bundle)
        vector_transport_to!(
            M,
            bms.transported_subgradients[j],
            qj,
            Xj,
            bms.p_last_serious,
            bms.vector_transport_method,
        )
    end
    _convex_bundle_subsolver!(M, bms)
    bms.g .= sum(bms.λ .* bms.transported_subgradients)
    bms.ε = sum(bms.λ .* bms.linearization_errors)
    bms.ξ = (-norm(M, bms.p_last_serious, bms.g)^2) - (bms.ε)
    step = get_stepsize(mp, bms, i)
    retract!(M, bms.p, bms.p_last_serious, -step * bms.g, bms.retraction_method)
    bms.last_stepsize = step
    get_subgradient!(mp, bms.X, bms.p)
    if get_cost(mp, bms.p) ≤ (get_cost(mp, bms.p_last_serious) + bms.m * bms.ξ)
        copyto!(M, bms.p_last_serious, bms.p)
    end
    v = findall(λj -> λj ≤ bms.atol_λ, bms.λ)
    if !isempty(v)
        deleteat!(bms.bundle, v)
        # Update sizes of subgradient and lambda linearization errors as well
        deleteat!(bms.λ, v)
        deleteat!(bms.linearization_errors, v)
        deleteat!(bms.transported_subgradients, v)
    end
    l = length(bms.bundle)
    if l == bms.bundle_cap
        #
        deleteat!(bms.bundle, 1)
        deleteat!(bms.λ, 1)
        deleteat!(bms.linearization_errors, 1)
        push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
        push!(bms.linearization_errors, 0.0)
        push!(bms.λ, 0.0)
    else
        # push to bundle and update subgradients, λ, and linearization_errors (+1 in length)
        push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
        push!(bms.linearization_errors, 0.0)
        push!(bms.λ, 0.0)
        push!(bms.transported_subgradients, zero_vector(M, bms.p))
    end
    for (j, (qj, Xj)) in enumerate(bms.bundle)
        v = if bms.k_max ≤ 0
            get_cost(mp, bms.p_last_serious) - get_cost(mp, qj) - (inner(
                M,
                qj,
                Xj,
                inverse_retract(M, qj, bms.p_last_serious, bms.inverse_retraction_method),
            ))
        else
            get_cost(mp, bms.p_last_serious) - get_cost(mp, qj) +
            norm(M, qj, Xj) * norm(
                M,
                qj,
                inverse_retract(M, qj, bms.p_last_serious, bms.inverse_retraction_method),
            )
        end
        bms.linearization_errors[j] = (0 ≥ v ≥ -bms.atol_errors) ? 0 : v
    end
    return bms
end
get_solver_result(bms::ConvexBundleMethodState) = bms.p_last_serious
function get_last_stepsize(::AbstractManoptProblem, bms::ConvexBundleMethodState, i)
    return bms.last_stepsize
end

#
#
# Dispatching on different types of subsolvers
# (a) closed form allocating
function _convex_bundle_subsolver!(
    M, bms::ConvexBundleMethodState{P,T,F,ClosedFormSubSolverState{AllocatingEvaluation}}
) where {P,T,F}
    bms.λ = bms.sub_problem(
        M, bms.p_last_serious, bms.linearization_errors, bms.transported_subgradients
    )
    return bms
end
# (b) closed form in-place
function _convex_bundle_subsolver!(
    M, bms::ConvexBundleMethodState{P,T,F,ClosedFormSubSolverState{InplaceEvaluation}}
) where {P,T,F}
    bms.sub_problem(
        M, bms.λ, bms.p_last_serious, bms.linearization_errors, bms.transported_subgradients
    )
    return bms
end
# (c) TODO: implement the case where problem and state are given and `solve!` is called

#
# Lagrange stopping criterion
function (sc::StopWhenLagrangeMultiplierLess)(
    mp::AbstractManoptProblem, bms::ConvexBundleMethodState, i::Int
)
    if i == 0 # reset on init
        sc.at_iteration = -1
    end
    M = get_manifold(mp)
    if (sc.mode == :estimate) && (-bms.ξ ≤ sc.tolerances[1]) && (i > 0)
        sc.values[1] = -bms.ξ
        sc.at_iteration = i
        return true
    end
    ng = norm(M, bms.p_last_serious, bms.g)
    if (sc.mode == :both) &&
        (bms.ε ≤ sc.tolerances[1]) &&
        (ng ≤ sc.tolerances[2]) &&
        (i > 0)
        sc.values[1] = bms.ε
        sc.values[2] = ng
        sc.at_iteration = i
        return true
    end
    return false
end
function (d::DebugWarnIfLagrangeMultiplierIncreases)(
    ::AbstractManoptProblem, st::ConvexBundleMethodState, i::Int
)
    (i < 1) && (return nothing)
    if d.status !== :No
        new_value = -st.ξ
        if new_value ≥ d.old_value * d.tol
            @warn """The Lagrange multiplier increased by at least $(d.tol).
            At iteration #$i the negative of the Lagrange multiplier, -ξ, increased from $(d.old_value) to $(new_value).\n
            Consider decreasing either the `diameter` keyword argument, or one
            of the parameters involved in the estimation of the sectional curvature, such as
            `k_max` in the `convex_bundle_method` call.
            of the parameters involved in the estimation of the sectional curvature, such as `k_max` in the `convex_bundle_method` call.
            """
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfLagrangeMultiplierIncreases(:Always) to get all warnings."
                d.status = :No
            end
        elseif new_value < zero(number_eltype(st.ξ))
            @warn """The Lagrange multiplier is positive.
            At iteration #$i the negative of the Lagrange multiplier, -ξ, became negative.\n
            Consider increasing either the `diameter` keyword argument, or changing
            one of the parameters involved in the estimation of the sectional curvature, such as
            `k_max` in the `convex_bundle_method` call.
            one of the parameters involved in the estimation of the sectional curvature, such as `k_max` in the `convex_bundle_method` call.
            """
        else
            d.old_value = min(d.old_value, new_value)
        end
    end
    return nothing
end

function (d::DebugStepsize)(
    dmp::P, bms::ConvexBundleMethodState, i::Int
) where {P<:AbstractManoptProblem}
    (i < 1) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_last_stepsize(dmp, bms, i))
    return nothing
end
