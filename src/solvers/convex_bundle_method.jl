function sectional_curvature(M, p)
    X = rand(M; vector_at=p)
    Y = rand(M; vector_at=p)
    Y = Y - inner(M, p, X, Y) / norm(M, p, X)^2 * X
    R = riemann_tensor(M, p, X, Y, Y)
    return inner(M, p, R, X) / (norm(M, p, X)^2 * norm(M, p, Y)^2 - inner(M, p, X, Y)^2)
end
function ζ_1(k_min, diam)
    (k_min < zero(k_min)) && return sqrt(-k_min) * diam * coth(sqrt(-k_min) * diam)
    (k_min ≥ zero(k_min)) && return one(k_min)
end
function ζ_2(k_max, diam)
    (k_max ≤ zero(k_max)) && return one(k_max)
    (k_max > zero(k_max)) && return sqrt(k_max) * diam * cot(sqrt(k_max) * diam)
end
function close_point(M, p, tol; retraction_method=default_retraction_method(M, typeof(p)))
    X = rand(M; vector_at=p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, retraction_method)
end

@doc raw"""
    ConvexBundleMethodState <: AbstractManoptSolverState
stores option values for a [`convex_bundle_method`](@ref) solver.

# Fields

* `atol_λ` - (eps()) tolerance parameter for the convex coefficients in λ
* `atol_errors` - (eps()) tolerance parameter for the linearization errors
* `bundle` - bundle that collects each iterate with the computed subgradient at the iterate
* `bundle_size` - (25) the size of the bundle
* `diam` - (50.0) estimate for the diameter of the level set of the objective function at the starting point
* `domain` - (@(M, p) -> isfinite(f(M, p))) a function to that evaluates to true when the current candidate is in the domain of the objective `f`, and false otherwise, e.g. : domain = (M, p) -> p ∈ dom f(M, p) ? true : false
* `g`- descent direction
* `inverse_retraction_method` - the inverse retraction to use within
* `lin_errors` - linearization errors at the last serious step
* `m` - (1e-3) the parameter to test the decrease of the cost: ``f(q_{k+1}) \le f(p_k) + m \xi``.
* `p` - current candidate point
* `p_last_serious` - last serious iterate
* `retraction_method` – the retraction to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `transported_subgradients` - subgradients of the bundle that are transported to p_last_serious
* `vector_transport_method` - the vector transport method to use within
* `X` - (`zero_vector(M, p)`) the current element from the possible subgradients at
`p` that was last evaluated.
* `stepsize` – ([`ConstantStepsize`](@ref)`(M)`) a [`Stepsize`](@ref)
* `ε` - convex combination of the linearization errors
* `λ` - convex coefficients that solve the subproblem
* `ξ` - the stopping parameter given by ξ = -|g|^2 - ε
* `ϱ` - curvature-dependent bound

# Constructor

ConvexBundleMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields above besides `p_last_serious` which obtains the same type as `p`.
    You can use e.g. `X=` to specify the type of tangent vector to use
    
## Keyword arguments

* `k_min` - lower bound on the sectional curvature of the manifold
* `k_max` - upper bound on the sectional curvature of the manifold
* `k_size` - (100) sample size for the estimation of the bounds on the sectional curvature of the manifold
* `p_estimate` - (p) the point around which to estimate the sectional curvature of the manifold
"""
mutable struct ConvexBundleMethodState{
    R,
    P,
    T,
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
} <: AbstractManoptSolverState where {R<:Real,P,T,I<:Int}
    atol_λ::R
    atol_errors::R
    bundle::B
    bundle_size::I
    diam::R
    domain::D
    g::T
    inverse_retraction_method::IR
    lin_errors::A
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
    function ConvexBundleMethodState(
        M::TM,
        p::P;
        atol_λ::R=eps(R),
        atol_errors::R=eps(R),
        bundle_size::Integer=25,
        m::R=1e-2,
        diam::R=50.0,
        domain::D=(M, p) -> isfinite(f(M, p)),
        k_min=nothing,
        k_max=nothing,
        k_size::Int=100,
        p_estimate=p,
        stepsize::S=default_stepsize(M, SubGradientMethodState),
        ϱ=nothing,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenBundleLess(1e-8) | StopAfterIteration(5000),
        X::T=zero_vector(M, p),
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
    ) where {
        D,
        IR<:AbstractInverseRetractionMethod,
        P,
        T,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        S<:Stepsize,
        VT<:AbstractVectorTransportMethod,
        R<:Real,
    }
        bundle = [(copy(M, p), zero_vector(M, p))]
        g = zero_vector(M, p)
        lin_errors = zeros(bundle_size)
        transported_subgradients = [zero_vector(M, p)]
        ε = zero(R)
        λ = [zero(R)]
        ξ = zero(R)
        if ϱ === nothing
            if (k_min === nothing) || (k_max === nothing)
                s = [
                    sectional_curvature(
                        M,
                        close_point(
                            M, p_estimate, diam / 2; retraction_method=retraction_method
                        ),
                    ) for _ in 1:k_size
                ]
            end
            (k_min === nothing) && (k_min = minimum(s))
            (k_max === nothing) && (k_max = maximum(s))
            ϱ = max(ζ_1(k_min, diam) - one(k_min), one(k_max) - ζ_2(k_max, diam))
        end
        return new{
            typeof(m),
            P,
            T,
            typeof(lin_errors),
            typeof(bundle),
            typeof(transported_subgradients),
            typeof(domain),
            typeof(bundle_size),
            IR,
            TR,
            S,
            SC,
            VT,
        }(
            atol_λ,
            atol_errors,
            bundle,
            bundle_size,
            diam,
            domain,
            g,
            inverse_retraction_method,
            lin_errors,
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
        )
    end
end
get_iterate(bms::ConvexBundleMethodState) = bms.p_last_serious
get_subgradient(bms::ConvexBundleMethodState) = bms.g

function show(io::IO, cbms::ConvexBundleMethodState)
    i = get_count(cbms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(cbms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Convex Bundle Method
    $Iter
    ## Parameters
    * tolerance parameter for the convex coefficients: $(cbms.atol_λ)
    * tolerance parameter for the linearization errors: $(cbms.atol_errors)
    * bundle size: $(cbms.bundle_size)
    * diameter: $(cbms.diam)
    * inverse retraction: $(cbms.inverse_retraction_method)
    * descent test parameter: $(cbms.m)
    * retraction: $(cbms.retraction_method)
    * vector transport: $(cbms.vector_transport_method)
    * stopping parameter value: $(cbms.ξ)
    * curvature-dependent bound: $(cbms.ϱ)

    ## Stopping Criterion
    $(status_summary(cbms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    convex_bundle_method(M, f, ∂f, p)

perform a convex bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)``, where ``\mathrm{retr}`` is a retraction and  
``g_k = \sum_{j\in J_k} λ_j^k \mathrm{P}_{p_k←q_j}X_{q_j},``
``p_k`` is the last serious iterate, ``X_{q_j}\in∂f(q_j)``, and the ``λ_j^k`` are solutions to the quadratic subproblem provided by the [`bundle_method_subsolver`](@ref).
Though the subdifferential might be set valued, the argument `∂f` should always
return _one_ element from the subdifferential, but not necessarily deterministic.

# Input
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`– the subgradient ``\partial f: \mathcal M→ T\mathcal M`` of f
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

# Optional
* `m` - (1e-3) the parameter to test the decrease of the cost: ``f(q_{k+1}) \le f(p_k) + m \xi``.
* `diam` - (50.0) estimate for the diameter of the level set of the objective function at the starting point.
* `domain` - (@(M, p) -> isfinite(f(M, p))) a function to that evaluates to true when the current candidate is in the domain of the objective `f`, and false otherwise, e.g. : domain = (M, p) -> p ∈ dom f(M, p) ? true : false.
* `k_min` - lower bound on the sectional curvature of the manifold.
* `k_max` - upper bound on the sectional curvature of the manifold.
* `k_size` - (100) sample size for the estimation of the bounds on the sectional curvature of the manifold if `k_min`
    and `k_max` are not provided.
* `p_estimate` - (p) the point around which to estimate the sectional curvature of the manifold.
* `α` - (@(j) -> one(number_eltype(X)) / j) (where j is an inner loop index) a function for evaluating suitable stepsizes when obtaining candidate points.
* `ϱ` - curvature-dependent bound.
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by
   allocation (default) form `∂f(M, q)` or [`InplaceEvaluation`](@ref) in place, i.e. is
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
function convex_bundle_method(
    M::AbstractManifold, f::TF, ∂f::TdF, p; kwargs...
) where {TF,TdF}
    p_star = copy(M, p)
    return convex_bundle_method!(M, f, ∂f, p_star; kwargs...)
end
@doc raw"""
    convex_bundle_method!(M, f, ∂f, p)

perform a bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)`` in place of `p`.

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`- the (sub)gradient ``\partial f:\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

for more details and all optional parameters, see [`convex_bundle_method`](@ref).
"""
function convex_bundle_method!(
    M::AbstractManifold,
    f::TF,
    ∂f!!::TdF,
    p;
    atol_λ::R=eps(),
    atol_errors::R=eps(),
    bundle_size::Int=25,
    diam::R=50.0,
    domain=(M, p) -> isfinite(f(M, p)),
    m::R=1e-3,
    k_min=nothing,
    k_max=nothing,
    k_size::Int=100,
    p_estimate=nothing,
    stepsize::Stepsize=DecreasingStepsize(1, 1, 0, 1, 0, :relative),
    ϱ=nothing,
    debug=[DebugWarnIfStoppingParameterIncreases()],
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopWhenBundleLess(1e-8), StopAfterIteration(5000)
    ),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
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
        bundle_size=bundle_size,
        diam=diam,
        domain=domain,
        m=m,
        k_min=k_min,
        k_max=k_max,
        k_size=k_size,
        p_estimate=p,
        stepsize=stepsize,
        ϱ=ϱ,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        vector_transport_method=vector_transport_method,
    )
    bms = decorate_state!(bms; debug=debug, kwargs...)
    return get_solver_return(solve!(mp, bms))
end
@doc raw"""
    bundle_method_subsolver(M, bms<:Union{ConvexBundleMethodState, ProxBundleMethodState})

solver for the subproblem of both the convex and proximal bundle methods.

The subproblem for the convex bundle method is
```math
    \arg\min_{\lambda \in \mathbb R^{|J_k|}}
    \frac{1}{2} ||\sum_{j \in J_k} \lambda_j \mathrm{P}_{p_k←q_j} X_{q_j}||^2 + \sum_{j \in J_k} \lambda_j \, c_j^k
```

```math
    \text{s. t.}
    \quad
    \sum_{j \in J_k} \lambda_j = 1
    ,
    \quad
    \lambda_j
    \ge
    0
    \quad \text{for all }
    j \in J_k
    ,
```
where ``J_k = \{j \in J_{k-1} \ | \ \lambda_j > 0\} \cup \{k\}``.

The subproblem for the proximal bundle method is
```math
    \arg\min_{\lambda \in \mathbb R^{|L_l|}}
    \quad
    \frac{1}{2 \mu_l} ||\sum_{j \in L_l} \lambda_j \mathrm{P}_{p_k←q_j} X_{q_j}||^2 + \sum_{j \in L_l} \lambda_j \, c_j^k
```

```math
    \text{s. t.}
    \quad
    \sum_{j \in L_l} \lambda_j = 1
    ,
    \quad
    \lambda_j
    \ge
    0
    \quad \text{for all }
    j \in L_l
    ,
```
where ``L_l = \{k\}`` if ``q_k`` is a serious iterate, and ``L_l = L_{l-1} \cup \{k\}`` otherwise.
"""
function bundle_method_subsolver(::Any, ::Any)
    throw(
        ErrorException("""Both packages "QuadraticModels" and "RipQP" need to be loaded.""")
    )
end
function initialize_solver!(mp::AbstractManoptProblem, bms::ConvexBundleMethodState)
    M = get_manifold(mp)
    copyto!(M, bms.p_last_serious, bms.p)
    get_subgradient!(mp, bms.X, bms.p)
    copyto!(M, bms.g, bms.p_last_serious, bms.X)
    bms.bundle = [(copy(M, bms.p), copy(M, bms.p, bms.X))]
    return bms
end
function step_solver!(mp::AbstractManoptProblem, bms::ConvexBundleMethodState, i)
    M = get_manifold(mp)
    bms.transported_subgradients = [
        vector_transport_to(M, qj, Xj, bms.p_last_serious, bms.vector_transport_method) for
        (qj, Xj) in bms.bundle
    ]
    bms.λ = bundle_method_subsolver(M, bms)
    bms.g .= sum(bms.λ .* bms.transported_subgradients)
    bms.ε = sum(bms.λ .* bms.lin_errors)
    bms.ξ = -norm(M, bms.p_last_serious, bms.g)^2 - bms.ε
    j = 1
    step = get_stepsize(mp, bms, j)
    retract!(M, bms.p, bms.p_last_serious, -step * bms.g, bms.retraction_method)
    while !bms.domain(M, bms.p)
        j += 1
        step = get_stepsize(mp, bms, j)
        println("   STEP = $step")
        retract!(M, bms.p, bms.p_last_serious, -step * bms.g, bms.retraction_method)
    end
    get_subgradient!(mp, bms.X, bms.p)
    if get_cost(mp, bms.p) ≤ (get_cost(mp, bms.p_last_serious) + bms.m * bms.ξ)
        copyto!(M, bms.p_last_serious, bms.p)
    end
    v = findall(λj -> λj ≤ bms.atol_λ, bms.λ)
    if !isempty(v)
        y = copy(M, bms.bundle[1][1])
        deleteat!(bms.bundle, v)
    end
    l = length(bms.bundle)
    push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
    if l == bms.bundle_size
        y = copy(M, bms.bundle[1][1])
        deleteat!(bms.bundle, 1)
    end
    bms.lin_errors = [
        get_cost(mp, bms.p_last_serious) - get_cost(mp, qj) - inner(
            M,
            qj,
            Xj,
            inverse_retract(M, qj, bms.p_last_serious, bms.inverse_retraction_method),
        ) +
        bms.ϱ *
        norm(
            M, qj, inverse_retract(M, qj, bms.p_last_serious, bms.inverse_retraction_method)
        ) *
        norm(M, qj, Xj) for (qj, Xj) in bms.bundle
    ]
    bms.lin_errors = [
        zero(bms.atol_errors) ≥ x ≥ -bms.atol_errors ? zero(bms.atol_errors) : x for
        x in bms.lin_errors
    ]
    return bms
end
get_solver_result(bms::ConvexBundleMethodState) = bms.p_last_serious

"""
    StopWhenBundleLess <: StoppingCriterion

Two stopping criteria for [`convex_bundle_method`](@ref) to indicate to stop when either

* the parameters ε and |g|

are less than given tolerances tole and tolg respectively, or

* the parameter -ξ = |g|^2 + ε

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
    mp::AbstractManoptProblem, bms::ConvexBundleMethodState, i::Int
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
    mp::AbstractManoptProblem, bms::ConvexBundleMethodState, i::Int
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

@doc raw"""
    DebugWarnIfStoppingParameterIncreases <: DebugAction

print a warning if the stopping parameter of the bundle method increases.

# Constructor
    DebugWarnIfStoppingParameterIncreases(warn=:Once; tol=1e2)

Initialize the warning to warning level (`:Once`) and introduce a tolerance for the test of `1e2`.

The `warn` level can be set to `:Once` to only warn the first time the cost increases,
to `:Always` to report an increase every time it happens, and it can be set to `:No`
to deactivate the warning, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always:`
"""
mutable struct DebugWarnIfStoppingParameterIncreases <: DebugAction
    # store if we need to warn – :Once, :Always, :No, where all others are handled
    # the same as :Always
    status::Symbol
    old_value::Float64
    tol::Float64
    function DebugWarnIfStoppingParameterIncreases(warn::Symbol=:Once; tol=1e2)
        return new(warn, Float64(Inf), tol)
    end
end
function (d::DebugWarnIfStoppingParameterIncreases)(
    p::AbstractManoptProblem, st::ConvexBundleMethodState, i::Int
)
    (i < 1) && (return nothing)
    if d.status !== :No
        new_value = -st.ξ
        if new_value ≥ d.old_value * d.tol
            @warn """The stopping parameter increased by at least $(d.tol).
            At iteration #$i the stopping parameter -ξ increased from $(d.old_value) to $(new_value).\n
            Consider decreasing either the diameter by changing the `diam` keyword argument, or one 
            of the parameters involved in the estimation of the sectional curvature, such as `k_min`,
            `k_max`, or `ϱ` in the `convex_bundle_method` call.
            """
            if d.status === :Once
                @warn "Further warnings will be supressed, use DebugWarnIfStoppingParameterIncreases(:Always) to get all warnings."
                d.status = :No
            end
        elseif new_value < zero(number_eltype(st.ξ))
            @warn """The stopping parameter is negative.
            At iteration #$i the stopping parameter -ξ became negative.\n
            Consider increasing either the diameter by changing the `diam` keyword argument, or changing 
            one of the parameters involved in the estimation of the sectional curvature, such as `k_min`,
            `k_max`, or `ϱ` in the `convex_bundle_method` call.
            """
        else
            d.old_value = min(d.old_value, new_value)
        end
    end
    return nothing
end
function show(io::IO, di::DebugWarnIfStoppingParameterIncreases)
    return print(io, "DebugWarnIfStoppingParameterIncreases(; tol=\"$(di.tol)\")")
end
function (d::DebugStepsize)(
    p::P, s::O, i::Int
) where {P<:AbstractManoptProblem,O<:ConvexBundleMethodState}
    (i < 1) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_last_stepsize(p, s, i))
    return nothing
end
