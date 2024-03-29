
@doc raw"""
    estimate_sectional_curvature(M::AbstractManifold, p)

Estimate the sectional curvature of a manifold ``\mathcal M`` at a point ``p \in \mathcal M``
on two random tangent vectors at ``p`` that are orthogonal to each other.

# See also

[`sectional_curvature`](@extref `ManifoldsBase.sectional_curvature-Tuple{AbstractManifold, Any, Any, Any}`)
"""
function estimate_sectional_curvature(M::AbstractManifold, p)
    X = rand(M; vector_at=p)
    Y = rand(M; vector_at=p)
    Y = Y - (inner(M, p, X, Y) / norm(M, p, X)^2 * X)
    return sectional_curvature(M, p, X, Y)
end

@doc raw"""
    ζ_1(ω, δ)

compute a curvature-dependent bound.
The formula reads

```math
\zeta_{1, ω}(δ)
\coloneqq
\begin{cases}
    1 & \text{if } ω ≥ 0, \\
    \sqrt{-ω} \, δ \cot(\sqrt{-ω} \, δ) & \text{if } ω < 0,
\end{cases}
```

where ``ω ≤ κ_p`` for all ``p ∈ \mathcal U`` is a lower bound to the sectional curvature in
a (strongly geodesically convex) bounded subset ``\mathcal U ⊆ \mathcal M`` with diameter ``δ``.
"""
function ζ_1(k_min, diameter)
    (k_min < zero(k_min)) && return sqrt(-k_min) * diameter * coth(sqrt(-k_min) * diameter)
    return one(k_min)
end

@doc raw"""
    ζ_2(Ω, δ)

compute a curvature-dependent bound.
The formula reads

```math
\zeta_{2, Ω}(δ) \coloneqq
\begin{cases}
    1 & \text{if } Ω ≤ 0,\\
    \sqrt{Ω} \, δ \cot(\sqrt{Ω} \, δ) & \text{if } Ω > 0,
\end{cases}
```

where ``Ω ≥ κ_p`` for all ``p ∈ \mathcal U`` is an upper bound to the sectional curvature in
a (strongly geodesically convex) bounded subset ``\mathcal U ⊆ \mathcal M`` with diameter ``δ``.
"""
function ζ_2(k_max, diameter)
    (k_max > zero(k_max)) && return sqrt(k_max) * diameter * cot(sqrt(k_max) * diameter)
    return one(k_max)
end

@doc raw"""
    close_point(M, p, tol; retraction_method=default_retraction_method(M, typeof(p)))

sample a random point close to ``p ∈ \mathcal M`` within a tolerance `tol`
and a [retraction](@extref ManifoldsBase :doc:`retractions`).
"""
function close_point(M, p, tol; retraction_method=default_retraction_method(M, typeof(p)))
    X = rand(M; vector_at=p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, retraction_method)
end

@doc raw"""
    ConvexBundleMethodState <: AbstractManoptSolverState

Stores option values for a [`convex_bundle_method`](@ref) solver.

# Fields

* `atol_λ`:                    (`eps()`) tolerance parameter for the convex coefficients in λ
* `atol_errors`:               (`eps()`) tolerance parameter for the linearization errors
* `bundle`:                    bundle that collects each iterate with the computed subgradient at the iterate
* `bundle_cap`:                (`25`) the maximal number of elements the bundle is allowed to remember
* `diameter`:                  (`50.0`) estimate for the diameter of the level set of the objective function at the starting point
* `domain`:                    (`(M, p) -> isfinite(f(M, p))`) a function to that evaluates to true when the current candidate is in the domain of the objective `f`, and false otherwise, e.g. : domain = (M, p) -> p ∈ dom f(M, p) ? true : false
* `g`:                         descent direction
* `inverse_retraction_method`: the inverse retraction to use within
* `linearization_errors`:      linearization errors at the last serious step
* `m`:                         (`1e-3`) the parameter to test the decrease of the cost: ``f(q_{k+1}) \le f(p_k) + m \xi``.
* `p`:                         current candidate point
* `p_last_serious`:            last serious iterate
* `retraction_method`:         the retraction to use within
* `stop`:                      a [`StoppingCriterion`](@ref)
* `transported_subgradients`:  subgradients of the bundle that are transported to p_last_serious
* `vector_transport_method`:   the vector transport method to use within
* `X`:                         (`zero_vector(M, p)`) the current element from the possible subgradients at `p` that was last evaluated.
* `stepsize`:                  ([`ConstantStepsize`](@ref)`(M)`) a [`Stepsize`](@ref)
* `ε`:                         convex combination of the linearization errors
* `λ`:                         convex coefficients that solve the subproblem
* `ξ`:                         the stopping parameter given by ``ξ = -\lvert g\rvert^2 – ε``
* `ϱ`:                         curvature-dependent bound
* `sub_problem`:               ([`convex_bundle_method_subsolver`]) a function that solves the sub problem on `M` given the last serious iterate `p_last_serious`, the linearization errors `linearization_errors`, and the transported subgradients `transported_subgradients`,
* `sub_state`:                 an [`AbstractEvaluationType`](@ref) indicating whether `sub_problem` works inplace of `λ` or allocates a solution

# Constructor

    ConvexBundleMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields with defaults besides `p_last_serious` which obtains the same type as `p`.
    You can use e.g. `X=` to specify the type of tangent vector to use

## Keyword arguments

* `k_max`:      upper bound on the sectional curvature of the manifold
* `k_size`:     (`100`) sample size for the estimation of the bounds on the sectional curvature of the manifold
* `p_estimate`: (`p`) the point around which to estimate the sectional curvature of the manifold
"""
mutable struct ConvexBundleMethodState{
    P,
    T,
    Pr,
    St,
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
} <: AbstractManoptSolverState where {R<:Real,P,T,I<:Int,Pr,St}
    atol_λ::R
    atol_errors::R
    bundle::B
    bundle_cap::I
    diameter::R
    domain::D
    g::T
    inverse_retraction_method::IR
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
    ϱ::R
    sub_problem::Pr
    sub_state::St
    function ConvexBundleMethodState(
        M::TM,
        p::P;
        atol_λ::R=eps(),
        atol_errors::R=eps(),
        bundle_cap::Integer=25,
        m::R=1e-2,
        diameter::R=50.0,
        domain::D,#(M, p) -> isfinite(f(M, p)),
        k_max=nothing,
        k_size::Int=100,
        p_estimate=p,
        stepsize::S=default_stepsize(M, SubGradientMethodState),
        ϱ=nothing,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenLagrangeMultiplierLess(1e-8) |
                               StopAfterIteration(5000),
        X::T=zero_vector(M, p),
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
        sub_problem::Pr=convex_bundle_method_subsolver,
        sub_state::St=AllocatingEvaluation(),
    ) where {
        D,
        IR<:AbstractInverseRetractionMethod,
        P,
        T,
        Pr,
        St,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        S<:Stepsize,
        VT<:AbstractVectorTransportMethod,
        R<:Real,
    }
        bundle = Vector{Tuple{P,T}}()
        g = zero_vector(M, p)
        last_stepsize = one(R)
        linearization_errors = Vector{R}()
        transported_subgradients = Vector{T}()
        ε = zero(R)
        λ = Vector{R}()
        ξ = zero(R)
        if ϱ === nothing
            if (k_max === nothing)
                s = [
                    estimate_sectional_curvature(
                        M,
                        close_point(
                            M, p_estimate, diameter / 2; retraction_method=retraction_method
                        ),
                    ) for _ in 1:k_size
                ]
                k_max = maximum(s)
            end
            ϱ = ζ_2(k_max, diameter)
        end
        return new{
            P,
            T,
            Pr,
            St,
            typeof(m),
            typeof(linearization_errors),
            typeof(bundle),
            typeof(transported_subgradients),
            typeof(domain),
            typeof(bundle_cap),
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
            ϱ,
            sub_problem,
            sub_state,
        )
    end
end
get_iterate(bms::ConvexBundleMethodState) = bms.p_last_serious
function set_iterate!(bms::ConvexBundleMethodState, M, p)
    copyto!(M, bms.p_last_serious, p)
    return bms
end
get_subgradient(bms::ConvexBundleMethodState) = bms.g

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
    * curvature-dependent bound:                        $(cbms.ϱ)
    * descent test parameter:                           $(cbms.m)
    * diameter:                                         $(cbms.diameter)
    * inverse retraction:                               $(cbms.inverse_retraction_method)
    * retraction:                                       $(cbms.retraction_method)
    * Lagrange parameter value:                         $(cbms.ξ)
    * vector transport:                                 $(cbms.vector_transport_method)

    ## Stopping Criterion
    $(status_summary(cbms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    convex_bundle_method(M, f, ∂f, p)

perform a convex bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)``, where ``\mathrm{retr}``
is a retraction and

```math
g_k = \sum_{j\in J_k} λ_j^k \mathrm{P}_{p_k←q_j}X_{q_j},
```

``p_k`` is the last serious iterate, ``X_{q_j} ∈ ∂f(q_j)``, and the ``λ_j^k`` are solutions
to the quadratic subproblem provided by the [`convex_bundle_method_subsolver`](@ref).

Though the subdifferential might be set valued, the argument `∂f` should always
return one element from the subdifferential, but not necessarily deterministic.

For more details, see [BergmannHerzogJasa:2024](@cite).

# Input

* `M`:  a manifold ``\mathcal M``
* `f`:   a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`: the subgradient ``∂f: \mathcal M → T\mathcal M`` of f
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p`:  (`rand(M)`) an initial value ``p_0 ∈ \mathcal M``

# Optional

* `atol_λ`:                    (`eps()`) tolerance parameter for the convex coefficients in λ.
* `atol_errors`:               (`eps()`) tolerance parameter for the linearization errors.
* `m`:                         (`1e-3`) the parameter to test the decrease of the cost: ``f(q_{k+1}) \le f(p_k) + m \xi``.
* `diameter`:                  (`50.0`) estimate for the diameter of the level set of the objective function at the starting point.
* `domain`:                    (`(M, p) -> isfinite(f(M, p))`) a function to that evaluates to true when the current candidate is in the domain of the objective `f`, and false otherwise, e.g. : domain = (M, p) -> p ∈ dom f(M, p) ? true : false.
* `k_max`:                     upper bound on the sectional curvature of the manifold.
* `k_size`:                    (`100``) sample size for the estimation of the bounds on the sectional curvature of the manifold if  `k_max` is not provided.
* `p_estimate`:                (`p`) the point around which to estimate the sectional curvature of the manifold.
* `α`:                         (`(i) -> one(number_eltype(X)) / i`) a function for evaluating suitable stepsizes when obtaining candidate points at iteration `i`.
* `ϱ`:                         curvature-dependent bound.
* `evaluation`:                ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by
   allocation (default) form `∂f(M, q)` or [`InplaceEvaluation`](@ref) in place, i.e. is
   of the form `∂f!(M, X, p)`.
* `inverse_retraction_method`: (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction method to use
* `retraction_method`:         (`default_retraction_method(M, typeof(p))`) a `retraction(M, p, X)` to use.
* `stopping_criterion`:        ([`StopWhenLagrangeMultiplierLess`](@ref)`(1e-8)`) a functor, see[`StoppingCriterion`](@ref), indicating when to stop
* `vector_transport_method`:   (`default_vector_transport_method(M, typeof(p))`) a vector transport method to use
* `sub_problem`:               a function evaluating with new allocations that solves the sub problem on `M` given the last serious iterate `p_last_serious`, the linearization errors `linearization_errors`, and the transported subgradients `transported_subgradients`

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
function convex_bundle_method(
    M::AbstractManifold, f::TF, ∂f::TdF, p=rand(M); kwargs...
) where {TF,TdF}
    p_star = copy(M, p)
    return convex_bundle_method!(M, f, ∂f, p_star; kwargs...)
end
@doc raw"""
    convex_bundle_method!(M, f, ∂f, p)

perform a bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)`` in place of `p`.

# Input

* `M`:  a manifold ``\mathcal M``
* `f`:  a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`: the (sub)gradient ``∂f:\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p`:  an initial value ``p_0=p ∈ \mathcal M``

for more details and all optional parameters, see [`convex_bundle_method`](@ref).
"""
function convex_bundle_method!(
    M::AbstractManifold,
    f::TF,
    ∂f!!::TdF,
    p;
    atol_λ::R=eps(),
    atol_errors::R=eps(),
    bundle_cap::Int=25,
    diameter::R=π / 3,# k_max -> k_max === nothing ? π/2 : (k_max ≤ zero(R) ? typemax(R) : π/3),
    domain=(M, p) -> isfinite(f(M, p)),
    m::R=1e-3,
    k_max=nothing,
    k_size::Int=100,
    p_estimate=p,
    stepsize::Stepsize=DecreasingStepsize(1, 1, 0, 1, 0, :relative),
    ϱ=nothing,
    debug=[DebugWarnIfLagrangeMultiplierIncreases()],
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopWhenLagrangeMultiplierLess(1e-8), StopAfterIteration(5000)
    ),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    sub_problem=convex_bundle_method_subsolver,
    sub_state=evaluation,
    kwargs..., #especially may contain debug
) where {R<:Real,TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
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
        k_size=k_size,
        p_estimate=p_estimate,
        stepsize=stepsize,
        ϱ=ϱ,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        vector_transport_method=vector_transport_method,
        sub_problem=sub_problem,
        sub_state=sub_state,
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
    # Refactor to inplace
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
    j = 1
    step = get_stepsize(mp, bms, j)
    retract!(M, bms.p, bms.p_last_serious, -step * bms.g, bms.retraction_method)
    while !bms.domain(M, bms.p) ||
        distance(M, bms.p, bms.p_last_serious) < step * norm(M, bms.p_last_serious, bms.g)
        j += 1
        step = get_stepsize(mp, bms, j)
        retract!(M, bms.p, bms.p_last_serious, -step * bms.g, bms.retraction_method)
    end
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
        # push to bundle and update subgradients, λ and linearization_errors (+1 in length)
        push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
        push!(bms.linearization_errors, 0.0)
        push!(bms.λ, 0.0)
        push!(bms.transported_subgradients, zero_vector(M, bms.p))
    end
    for (j, (qj, Xj)) in enumerate(bms.bundle)
        v =
            get_cost(mp, bms.p_last_serious) - get_cost(mp, qj) - (
                bms.ϱ * inner(
                    M,
                    qj,
                    Xj,
                    inverse_retract(
                        M, qj, bms.p_last_serious, bms.inverse_retraction_method
                    ),
                )
            )
        bms.linearization_errors[j] = (0 ≥ v ≥ -bms.atol_errors) ? 0 : v
    end
    return bms
end
get_solver_result(bms::ConvexBundleMethodState) = bms.p_last_serious
get_last_stepsize(bms::ConvexBundleMethodState) = bms.last_stepsize

#
#
# Dispatching on different types of subsolvers
# (a) closed form allocating
function _convex_bundle_subsolver!(
    M, bms::ConvexBundleMethodState{P,T,F,AllocatingEvaluation}
) where {P,T,F}
    bms.λ = bms.sub_problem(
        M, bms.p_last_serious, bms.linearization_errors, bms.transported_subgradients
    )
    return bms
end
# (b) closed form in-place
function _convex_bundle_subsolver!(
    M, bms::ConvexBundleMethodState{P,T,F,InplaceEvaluation}
) where {P,T,F}
    bms.sub_problem(
        M, bms.λ, bms.p_last_serious, bms.linearization_errors, bms.transported_subgradients
    )
    return bms
end
# (c) if necessary one could implement the case where we have problem and state and call solve!

#
# Lagrange stopping crtierion
function (sc::StopWhenLagrangeMultiplierLess)(
    mp::AbstractManoptProblem, bms::ConvexBundleMethodState, i::Int
)
    if i == 0 # reset on init
        sc.reason = ""
        sc.at_iteration = 0
    end
    M = get_manifold(mp)
    if (sc.mode == :estimate) && (-bms.ξ ≤ sc.tolerance[1]) && (i > 0)
        sc.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter -ξ = $(-bms.ξ) ≤ $(sc.tolerance[1]).\n"
        sc.at_iteration = i
        return true
    end
    ng = norm(M, bms.p_last_serious, bms.g)
    if (sc.mode == :both) && (bms.ε ≤ sc.tolerance[1]) && (ng ≤ sc.tolerance[2]) && (i > 0)
        sc.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter ε = $(bms.ε) ≤ $(sc.tolerance[1]) and |g| = $(ng) ≤ $(sc.tolerance[2]).\n"
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
            `k_max`, or `ϱ` in the `convex_bundle_method` call.
            """
            if d.status === :Once
                @warn "Further warnings will be supressed, use DebugWarnIfLagrangeMultiplierIncreases(:Always) to get all warnings."
                d.status = :No
            end
        elseif new_value < zero(number_eltype(st.ξ))
            @warn """The Lagrange multiplier is positive.
            At iteration #$i the negative of the Lagrange multiplier, -ξ, became negative.\n
            Consider increasing either the `diameter` keyword argument, or changing
            one of the parameters involved in the estimation of the sectional curvature, such as
            `k_max`, or `ϱ` in the `convex_bundle_method` call.
            """
        else
            d.old_value = min(d.old_value, new_value)
        end
    end
    return nothing
end

function (d::DebugStepsize)(
    ::P, bms::ConvexBundleMethodState, i::Int
) where {P<:AbstractManoptProblem}
    (i < 1) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_last_stepsize(bms))
    return nothing
end
