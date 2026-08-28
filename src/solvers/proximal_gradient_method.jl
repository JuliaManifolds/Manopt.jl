#
#
# Subproblem default cost and gradient
@doc """
    ProximalGradientNonsmoothCost{F, R, P} <: AbstractManifoldFunction

Stores the nonsmooth part ``h`` of the proximal gradient objective ``f = g + h``, as well as the stepsize parameter ``λ ∈ ℝ``.

This struct is also a functor `(M, q) -> v` that can be used as a cost function within a solver, primarily for solving the proximal map subproblem formulation in the proximal gradient method, which reads

```math
    $(_tex(:prox))_{λ h}(p) = $(_tex(:argmin))_{q ∈ $(_math(:Manifold))} h(q) + $(_tex(:frac, "1", "2λ"))$(_math(:distance))^2(q, p)
```

Hence, the functor reads

```math
    (M, q) ↦ h(q) + \frac{1}{2λ} $(_math(:distance))^2(q, p)
```

and `p` is the proximity point where the proximal map is evaluated, i.e. the argument `p` of the proximal map ``$(_tex(:prox))_{λ h}``.

## Fields

* `cost::F` - the nonsmooth part ``h`` of the proximal gradient objective, i.e. the part of the objective whose proximal map is sought
* `λ::R` - the stepsize parameter for the proximal map
* `proximity_point::P` - point where the proximal map is evaluated, i.e. the argument ``p`` of the proximal map ``$(_tex(:prox))_{λ h} (p)`` that we want to solve for

# Constructor
    ProximalGradientNonsmoothCost(cost, λ, proximity_point)
"""
mutable struct ProximalGradientNonsmoothCost{F, R, P} <: AbstractManifoldFunction
    cost::F
    λ::R
    proximity_point::P
end
function set_parameter!(pgnc::ProximalGradientNonsmoothCost, ::Val{:λ}, λ)
    pgnc.λ = λ
    return pgnc
end
get_parameter(pgnc::ProximalGradientNonsmoothCost, ::Val{:λ}) = pgnc.λ
function set_parameter!(pgnc::ProximalGradientNonsmoothCost, ::Val{:proximity_point}, p)
    pgnc.proximity_point = p
    return pgnc
end
function get_parameter(pgnc::ProximalGradientNonsmoothCost, ::Val{:proximity_point})
    return pgnc.proximity_point
end

function (pgnc::ProximalGradientNonsmoothCost)(M::AbstractManifold, p)
    return pgnc.cost(M, p) + (1 / 2 * pgnc.λ) * distance(M, p, pgnc.proximity_point)^2
end

@doc """
    ProximalGradientNonsmoothSubgradient{F, R, P} <: AbstractManifoldFunction

Stores a subgradient of the nonsmooth part ``h`` of the proximal gradient objective ``f = g + h``, as well as the stepsize parameter ``λ ∈ ℝ``.

This struct is also a functor in both formats
    * `(M, p) -> X` to compute the gradient in allocating fashion.
This is primarily used for computing a subgradient of the cost function ``h(q) + $(_tex(:frac, "1", "2λ"))$(_math(:distance))^2(q, p)`` that defines proximal map in the proximal gradient method. This reads
```math
    ∂h(q) - $(_tex(:frac, "1", "λ"))$(_tex(:log))_q p
```
is the proximity point where the proximal map is evaluated, i.e. the argument ``p`` of the proximal map ``$(_tex(:prox))_{λ h} (p)``.

## Fields

* `X::F` - the subgradient of the nonsmooth part of the total objective, i.e. the part of the objective whose proximal map is sought
* `λ::R` - the stepsize parameter for the proximal map
* `proximity_point::P` - point where the proximal map is evaluated, i.e. the argument of the proximal map that we want to solve for

# Constructor


    ProximalGradientNonsmoothSubgradient(cost, λ, proximity_point)
"""
mutable struct ProximalGradientNonsmoothSubgradient{F, R, P} <: AbstractManifoldFunction
    X::F
    λ::R
    proximity_point::P
end
function set_parameter!(pgns::ProximalGradientNonsmoothSubgradient, ::Val{:λ}, λ)
    pgns.λ = λ
    return pgns
end
get_parameter(pgns::ProximalGradientNonsmoothSubgradient, ::Val{:λ}) = pgns.λ
function set_parameter!(pgns::ProximalGradientNonsmoothSubgradient, ::Val{:proximity_point}, p)
    pgns.proximity_point = p
    return pgns
end
function get_parameter(pgns::ProximalGradientNonsmoothSubgradient, ::Val{:proximity_point})
    return pgns.proximity_point
end
# Default, compute the subgradient of the proximal map given the subgradient of the nonsmooth part X
function (pgng::ProximalGradientNonsmoothSubgradient)(M::AbstractManifold, p)
    return pgng.X(M, p) - 1 / pgng.λ * log(M, p, pgng.proximity_point)
end

#
#
# State
@doc """
    ProximalGradientMethodState <: AbstractManoptSolverState

State for the [`proximal_gradient_method`](@ref) solver.

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:inverse_retraction_method))
* `a` - point after acceleration step
$(_fields(:p; add_properties = [:as_Iterate]))
* `q` - point for storing gradient step
$(_fields(:retraction_method))
* `X` - tangent vector for storing gradient
$(_fields(:stopping_criterion; name = "stop"))
* `acceleration` - a function `(problem, state, k) -> state` to compute an acceleration before the gradient step
* `stepsize` - a function or [`Stepsize`](@ref) object to compute the stepsize
* `last_stepsize` - stores the last computed stepsize
$(_fields(:sub_problem; name = "sub_problem", type = "Union{`[`AbstractManoptProblem`](@ref)`, F}"))
    or `missing` to take the proximal map from the [`ManifoldProximalGradientObjective`](@ref)
$(_fields(:sub_state)). This field is ignored, if the `sub_problem` is `missing`.

# Constructor

    ProximalGradientMethodState(M::AbstractManifold; kwargs...)

Generate the state for a given manifold `M` with initial iterate `p`.

## Input

$(_args(:M))

# Keyword arguments

* `stepsize=default_stepsize(M, ProximalGradientMethodState)`
$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
$(_kwargs(:inverse_retraction_method))
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:retraction_method))
* `acceleration=(p, s, k) -> (copyto!(get_manifold(M), s.a, s.p); s)` by default no acceleration is performed
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(100)"))
$(_kwargs(:sub_problem; default = "missing"))
$(_kwargs(:sub_state; default = _glossary[:Variable][:evaluation][:default]))
$(_kwargs(:X; add_properties = [:as_Memory]))
"""
mutable struct ProximalGradientMethodState{
        P, T, Pr <: Union{<:AbstractManoptProblem, F, Missing} where {F}, St <: Union{<:AbstractManoptSolverState, Missing},
        C <: AbstractDict{Symbol},
        A, SC <: StoppingCriterion, S <: Stepsize, RM <: AbstractRetractionMethod, IRM <: AbstractInverseRetractionMethod, R,
    } <: AbstractManoptSolverState
    a::P
    acceleration::A
    callbacks::C
    inverse_retraction_method::IRM
    last_stepsize::R
    p::P
    q::P
    retraction_method::RM
    stepsize::S
    stop::SC
    sub_problem::Pr
    sub_state::St
    X::T
    function ProximalGradientMethodState(
            sub_problem::Pr, sub_state::St;
            a::P, acceleration::A,
            callbacks::C = Dict{Symbol, Function}(),
            inverse_retraction_method::IRM, last_stepsize::R,
            p::P, q::P,
            retraction_method::RM, stepsize::S, stopping_criterion::SC,
            X::T,
        ) where {
            P, T, Pr <: Union{<:AbstractManoptProblem, F, Missing} where {F}, St <: Union{<:AbstractManoptSolverState, Missing},
            C <: AbstractDict{Symbol},
            A, SC <: StoppingCriterion, S <: Stepsize, RM <: AbstractRetractionMethod, IRM <: AbstractInverseRetractionMethod, R,
        }
        return new{P, T, Pr, St, C, A, SC, S, RM, IRM, R}(
            a, acceleration, callbacks,
            inverse_retraction_method, last_stepsize, p, q,
            retraction_method, stepsize, stopping_criterion,
            sub_problem, sub_state, X,
        )
    end
end
ProximalGradientMethodState(M::AbstractManifold, st::AbstractManoptSolverState; kwargs...) = error("Proximal Gradient Method state can not be constructed based on $M and the sub state $st, a sub_problem is missing")
function ProximalGradientMethodState(
        M::AbstractManifold;
        callbacks::C = Dict{Symbol, Function}(),
        p::P = rand(M),
        acceleration::A = function (pr, st, k)
            copyto!(get_manifold(pr), st.a, st.p)
            return st
        end,
        stepsize::S = default_stepsize(M, ProximalGradientMethodState),
        stopping_criterion::SC = StopWhenGradientMappingNormLess(1.0e-2) | StopAfterIteration(5000) | StopWhenChangeLess(M, 1.0e-9),
        X::T = zero_vector(M, p),
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
        sub_problem::Pr = missing,
        sub_state::St = missing,
    ) where {
        P, T, SC <: StoppingCriterion, A,
        Pr <: Union{<:AbstractManoptProblem, F, Missing} where {F}, St <: Union{<:AbstractManoptSolverState, <:AbstractEvaluationType, Missing},
        RM <: AbstractRetractionMethod, IRM <: AbstractInverseRetractionMethod, S <: Stepsize,
        C <: AbstractDict{Symbol},
    }
    sub_state_ = (sub_state isa AbstractEvaluationType) ? ClosedFormSubSolverState() : sub_state
    sub_problem_ = (ismissing(sub_problem) || Pr <: AbstractManoptProblem) ? sub_problem : maybe_wrap_function(sub_problem, p, sub_state)
    return ProximalGradientMethodState(
        sub_problem_, sub_state_;
        callbacks = callbacks,
        a = copy(M, p), acceleration = acceleration,
        stepsize = stepsize, last_stepsize = zero(number_eltype(p)),
        p = p, q = copy(M, p),
        stopping_criterion = stopping_criterion, X = X,
        retraction_method = retraction_method, inverse_retraction_method = inverse_retraction_method,
    )
end

get_iterate(pgms::ProximalGradientMethodState) = pgms.p

function set_iterate!(pgms::ProximalGradientMethodState, M, p)
    pgms.p = p
    copyto!(M, pgms.p, p)
    return pgms
end
provided_callbacks(::Type{ProximalGradientMethodState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:BeforeSubsolver, :Stepsize, :Subsolver])
get_callbacks(pgms::ProximalGradientMethodState) = pgms.callbacks

function Base.show(io::IO, pgms::ProximalGradientMethodState)
    print(io, "ProximalGradientMethodState(", pgms.sub_problem, ", ", pgms.sub_problem, ";")
    print(io, " callbacks = ", pgms.callbacks, ",")
    print(io, " a = ", pgms.a, ", acceleration = ", pgms.acceleration, ", stepsize = ", pgms.stepsize)
    print(io, ", last_stepsize = ", pgms.last_stepsize, ", p = ", pgms.p, ", q = ", pgms.q)
    print(io, ", stopping_criterion = ", pgms.stop, ", X = ", pgms.X)
    print(io, ", retraction_method = ", pgms.retraction_method, ", inverse_retraction_method = ", pgms.inverse_retraction_method)
    return print(io, ")")
end
function status_summary(pgms::ProximalGradientMethodState; context::Symbol = :default)
    i = get_count(pgms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pgms.stop) ? "Yes" : "No"
    _is_inline(context) && (return "$(repr(pgms)) – $(Iter) $(has_converged(pgms) ? "(converged)" : "")")
    as = _callbacks_summary(pgms)
    s = """
    # Solver state for `Manopt.jl`s Proximal Gradient Method
    $Iter
    ## Parameters$(as)
    * retraction_method:              $(pgms.retraction_method)
    * stepsize:                       $(typeof(pgms.stepsize))
    * acceleration:                   $(typeof(pgms.acceleration))

    ## Stopping criterion
    $(_in_str(status_summary(pgms.stop; context = context); indent = 0, headers = 1))
    This indicates convergence: $Conv"""
    return s
end

#
#
# Stepsize
@doc """
    ProximalGradientMethodBacktrackingStepsize <: Stepsize

A functor for backtracking line search in proximal gradient methods.

# Fields

* `initial_stepsize::T` - initial step size guess
* `sufficient_decrease::T` - sufficient decrease parameter (default: 0.5)
* `contraction_factor::T` - step size reduction factor (default: 0.5)
* `strategy::Symbol` - `:nonconvex` or `:convex` (default: `:nonconvex`)
* `candidate_point::P` - a working point used during backtracking
* `last_stepsize::T` - the last computed stepsize

# Constructor
    ProximalGradientMethodBacktrackingStepsize(M::AbstractManifold; kwargs...)

## Keyword arguments

* `initial_stepsize=1.0`: initial stepsize to try
* `stop_when_stepsize_less=1e-8`: smallest stepsize when to stop (the last one before is taken)
* `sufficient_decrease=0.5`: sufficient decrease parameter
* `contraction_factor=0.5`: step size reduction factor
* `strategy=:nonconvex`: backtracking strategy, either `:convex` or `:nonconvex`
* `k_max=0.0`: an upper bound to the sectional curvatures of the manifold, only for the `:convex` strategy
* `δ=1e-2`: parameter for backtracking in case `k_max > 0`, only for the `:convex` strategy
"""
mutable struct ProximalGradientMethodBacktrackingStepsize{P, T} <: Stepsize
    initial_stepsize::T
    sufficient_decrease::T
    contraction_factor::T
    strategy::Symbol
    candidate_point::P
    last_stepsize::T
    stop_when_stepsize_less::T
    warm_start_factor::T
    k_max::T
    δ::T

    function ProximalGradientMethodBacktrackingStepsize(
            M::AbstractManifold;
            initial_stepsize::T = 1.0, sufficient_decrease::T = 0.5, contraction_factor::T = 0.5,
            strategy::Symbol = :nonconvex, stop_when_stepsize_less::T = 1.0e-8, warm_start_factor::T = 1.0,
            k_max::T = 0.0,
            δ::T = 1.0e-2,
        ) where {T}
        0 < sufficient_decrease < 1 ||
            throw(DomainError(sufficient_decrease, "sufficient_decrease must be in (0, 1)"))
        0 < contraction_factor < 1 ||
            throw(DomainError(contraction_factor, "contraction_factor must be in (0, 1)"))
        initial_stepsize > 0 ||
            throw(DomainError(initial_stepsize, "initial_stepsize must be positive"))
        strategy in [:convex, :nonconvex] ||
            throw(DomainError(strategy, "strategy must be either :convex or :nonconvex"))
        stop_when_stepsize_less > 0 || throw(
            DomainError(
                stop_when_stepsize_less, "stop_when_stepsize_less must be positive"
            ),
        )
        warm_start_factor > 0 ||
            throw(DomainError(warm_start_factor, "warm_start_factor must be positive"))

        (k_max > 0 && δ ≤ 0) &&
            throw(DomainError(δ, "the tolerance parameter δ must be positive if k_max > 0"))

        p = rand(M)
        return new{typeof(p), T}(
            initial_stepsize, sufficient_decrease, contraction_factor, strategy, p,
            initial_stepsize, stop_when_stepsize_less, warm_start_factor, k_max, δ
        )
    end
end

get_initial_stepsize(s::ProximalGradientMethodBacktrackingStepsize) = s.initial_stepsize
get_last_stepsize(s::ProximalGradientMethodBacktrackingStepsize) = s.last_stepsize

function Base.show(io::IO, pgb::ProximalGradientMethodBacktrackingStepsize)
    print(io, "ProximalGradientMethodBacktrackingStepsize(; initial_stepsize = ", pgb.initial_stepsize)
    print(io, ", sufficient_decrease = ", pgb.sufficient_decrease, ", contraction_factor = ", pgb.contraction_factor)
    print(io, ", strategy = :$(pgb.strategy), candidate_point = ", pgb.candidate_point)
    print(io, ", last_stepsize = ", pgb.last_stepsize, ", stop_when_stepsize_less = ", pgb.stop_when_stepsize_less)
    print(io, ", warm_start_factor = ", pgb.warm_start_factor)
    print(io, ", k_max = ", pgb.k_max, ", δ = ", pgb.δ)
    return print(io, ")")
end
function status_summary(pgb::ProximalGradientMethodBacktrackingStepsize; context::Symbol = :default)
    (context === :short) && return (repr(pgb))
    (context === :inline) && return "A proximal gradient backtracking step size (last step size: $(pgb.last_stepsize))"
    return """
    A backtracking method tailored for the proximal gradient method
    (last step size: $(pgb.last_stepsize))

    ## Parameters
    * contraction factor:       $(_MANOPT_INDENT)$(pgb.contraction_factor)
    * sufficient decrease:      $(_MANOPT_INDENT)$(pgb.sufficient_decrease)
    * strategy:                 $(_MANOPT_INDENT):$(pgb.strategy)
    * stop when step size less: $(_MANOPT_INDENT)$(pgb.stop_when_stepsize_less)
    * warm start factor:        $(_MANOPT_INDENT)$(pgb.warm_start_factor)
    """
end
function (s::ProximalGradientMethodBacktrackingStepsize)(
        mp::AbstractManoptProblem, st::ProximalGradientMethodState, k::Int, args...; kwargs...
    )
    # Initialization
    M = get_manifold(mp)
    p = st.a  # Current point (post-acceleration)
    X = st.X  # Current gradient

    # For the convex case, start with the last stepsize (warm start)
    # For the nonconvex case, reset to initial stepsize
    λ = if s.strategy === :convex && k > 1
        min(s.initial_stepsize, s.warm_start_factor * s.last_stepsize)
    else
        s.initial_stepsize
    end

    # Get the objective and temporary state
    objective = get_objective(mp)

    # Temporary state for backtracking that doesn't affect the main state
    pgm_temp = ProximalGradientMethodState(
        M;
        p = copy(M, p), X = zero_vector(M, p),
        sub_problem = st.sub_problem, sub_state = st.sub_state,
        retraction_method = st.retraction_method, inverse_retraction_method = st.inverse_retraction_method,
    )

    while λ > s.stop_when_stepsize_less
        # Perform gradient step with current λ
        direction = -λ * X
        retract!(M, pgm_temp.a, p, direction, st.retraction_method)
        distance_gradient = norm(M, p, direction)


        # Perform proximal step with current λ
        _pgm_proximal_step(mp, pgm_temp, λ)
        candidate_point = copy(M, pgm_temp.p)

        # Compute log_p(candidate_point) and its squared norm for the conditions
        log_p_q = inverse_retract(M, p, candidate_point, st.inverse_retraction_method)
        distance_candidate = norm(M, p, log_p_q)
        squared_distance = distance_candidate^2
        π_k = s.k_max ≤ eps(eltype(s.k_max)) ? Inf : π / √s.k_max
        r_δ = π_k / (2 + s.δ)

        if max(distance_gradient, distance_candidate) ≤ r_δ
            if s.strategy === :nonconvex
                # Nonconvex descent condition
                if get_cost(mp, p) - get_cost(mp, candidate_point) >=
                        (s.sufficient_decrease / λ) * squared_distance
                    s.last_stepsize = λ
                    return λ
                end
            elseif s.strategy === :convex
                g_p = get_cost_smooth(M, objective, p)
                g_q = get_cost_smooth(M, objective, candidate_point)


                ζ_δ = s.k_max ≤ zero(eltype(s.k_max)) ? one(eltype(s.k_max)) : π / (2 + s.δ) * cot(π / (2 + s.δ))

                # Convex descent condition
                if g_q <= g_p + inner(M, p, X, log_p_q) + (ζ_δ / 2λ) * squared_distance
                    s.last_stepsize = λ
                    return λ
                end
            end
        end

        # Reduce step size
        λ *= s.contraction_factor
    end
    return λ
end

@doc """
    ProximalGradientMethodBacktracking(; kwargs...)
    ProximalGradientMethodBacktracking(M::AbstractManifold; kwargs...)

Compute a stepsize for the proximal gradient method using a backtracking line search.

For the nonconvex case, the condition is:

```math
f(p) - f(T_{λ}(p)) ≥ γλ$(_tex(:norm, "G_{λ}(p)"))^2
```

where ``G_{λ}(p) = (1/λ) * $(_tex(:log))_p(T_{λ}(p))`` is the gradient mapping.

For the convex case, the condition is:

```math
g(T_{λ}(p)) ≤ g(p) + ⟨$(_tex(:grad)) g(p), $(_tex(:log))_p T_{λ}(p)⟩ + $(_tex(:frac, "ζ_δ", "2λ")) $(_math(:distance))^2(p, T_{λ}(p))
```

Returns a stepsize `λ` that satisfies the specified condition.

$(_note(:ManifoldDefaultsFactory, "ProximalGradientMethodBacktrackingStepsize"))
"""
function ProximalGradientMethodBacktracking(args...; kwargs...)
    return ManifoldDefaultsFactory(
        Manopt.ProximalGradientMethodBacktrackingStepsize, args...; kwargs...
    )
end

"""
    default_stepsize(M::AbstractManifold, ::Type{<:ProximalGradientMethodState})

Returns the default proximal stepsize, which is a nonconvex backtracking strategy.
"""
function default_stepsize(M::AbstractManifold, ::Type{<:ProximalGradientMethodState})
    return ProximalGradientMethodBacktrackingStepsize(
        M; initial_stepsize = 1.5, strategy = :nonconvex
    )
end

#
#
# Acceleration
@doc """
    ProximalGradientMethodAcceleration{P, T, F}

Compute an acceleration step

```math
a^{(k)} = $(_tex(:retr))_{p^{(k)}}$(_tex(:bigl))(
  -β_k$(_tex(:invretr))_{p^{(k)}}(p)
$(_tex(:bigr)))
```

where ``p^{(k)}`` is the current iterate from the [`ProximalGradientMethodState`](@ref)s
field `p` and the result is stored in `state.a`. The field `p` in this struct stores the last iterate.

The retraction and its inverse are taken from the state.

# Fields

* `p` - the last iterate
* `β` - acceleration parameter function or value
* `inverse_retraction_method` - method for inverse retraction
* `X` - tangent vector for computations

# Constructor

    ProximalGradientMethodAcceleration(M::AbstractManifold; kwargs...)

Generate the state for a given manifold `M` with initial iterate `p`.

## Input

$(_args(:M))

# Keyword arguments

* `β = k -> (k-1)/(k+2)` - acceleration parameter function or value
* `inverse_retraction_method` - method for inverse retraction
* `p` - initial point
* `X` - initial tangent vector
"""
mutable struct ProximalGradientMethodAcceleration{P, T, F, ITR <: AbstractInverseRetractionMethod}
    β::F
    inverse_retraction_method::ITR
    p::P
    X::T
    function ProximalGradientMethodAcceleration(;
            β::F, inverse_retraction_method::ITR, p::P, X::T
        ) where {P, T, F, ITR <: AbstractInverseRetractionMethod}
        return new{P, T, F, ITR}(β, inverse_retraction_method, p, X)
    end
end

function ProximalGradientMethodAcceleration(
        M::AbstractManifold;
        p::P = rand(M),
        X::T = zero_vector(M, p),
        β::F = (k) -> (k - 1) / (k + 2),
        inverse_retraction_method::I = default_inverse_retraction_method(M, typeof(p)),
    ) where {P, T, F, I <: AbstractInverseRetractionMethod}
    return ProximalGradientMethodAcceleration(
        β = β, inverse_retraction_method = inverse_retraction_method, p = p, X = X
    )
end

function (pga::ProximalGradientMethodAcceleration)(
        amp::AbstractManoptProblem, pgms::ProximalGradientMethodState, k
    )
    # compute the step
    M = get_manifold(amp)
    # inverse retract and store in X
    inverse_retract!(M, pga.X, pgms.p, pga.p)
    # retract with step and store in a
    retract!(M, pgms.a, pgms.p, -pga.β(k) * pga.X)
    # save current p for next time as last iterate
    copyto!(M, pga.p, pgms.p)
    return pgms
end

function Base.show(io::IO, pga::ProximalGradientMethodAcceleration)
    print(io, "ProximalGradientMethodAcceleration(; p = ", pga.p, ", X = ", pga.X)
    print(io, ", β = ", pga.β, ", inverse_retraction_method = ", pga.inverse_retraction_method)
    return print(io, ")")
end

#
#
# Stopping Criterion
function (sc::StopWhenGradientMappingNormLess)(
        mp::AbstractManoptProblem, s::ProximalGradientMethodState, k::Int
    )
    M = get_manifold(mp)
    if k == 0 # reset on init
        sc.at_iteration = -1
    end
    if (k > 0)
        sc.last_change =
            1 / s.last_stepsize * norm(
            M, s.q, inverse_retract(M, s.q, get_iterate(s), s.inverse_retraction_method)
        )
        if sc.last_change < sc.threshold
            sc.at_iteration = k
            return true
        end
    end
    return false
end

#
#
# Debug
function (d::DebugWarnIfStepsizeCollapsed)(
        ::AbstractManoptProblem,
        st::ProximalGradientMethodState{P, T, Pr, St, C, A, SC, TStS},
        k::Int,
    ) where {P, T, Pr, St, C, A, SC, TStS <: ProximalGradientMethodBacktrackingStepsize}
    (k < 1) && (return nothing)
    s = st.stepsize
    if d.status !== :No
        if s.last_stepsize ≤ s.stop_when_stepsize_less
            @warn "Backtracking stopped because the stepsize fell below the threshold $(s.stop_when_stepsize_less)."
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfStepsizeCollapsed(:Always) to get all warnings."
                d.status = :No
            end
        end
    end
    return nothing
end

#
#
# Solver
_doc_prox_grad_method = """
    proximal_gradient_method(M, f, g, grad_g, p=rand(M); prox_nonsmooth=missing, kwargs...)
    proximal_gradient_method(M, mpgo::ManifoldProximalGradientObjective, p=rand(M); kwargs...)
    proximal_gradient_method!(M, f, g, grad_g, p; prox_nonsmooth=missing, kwargs...)
    proximal_gradient_method!(M, mpgo::ManifoldProximalGradientObjective, p; kwargs...)

Perform the proximal gradient method as introduced in [BergmannJasaJohnPfeffer:2025:1](@cite) and [BergmannJasaJohnPfeffer:2025:2](@cite).
See also [FengHuangSongYingZeng:2021](@cite) for a similar approach.

Given the minimization problem

```math
$(_tex(:argmin))_{p∈$(_math(:Manifold))} f(p),
$(_tex(:quad)) $(_tex(:text, " where ")) $(_tex(:quad)) f(p) = g(p) + h(p).
```

This method performs the (intrinsic) proximal gradient method algorithm.

Let ``λ_k ≥ 0`` be a sequence of (proximal) parameters, initialize
``p^{(0)} = p``, and ``k=0``.

Then perform as long as the stopping criterion is not fulfilled
```math
p^{(k+1)} = prox_{λ_kh}$(_tex(:Bigl))(
$(_tex(:retr))_{a^{(k)}}$(_tex(:bigl))(-λ_k $(_tex(:grad)) g(a^{(k)}$(_tex(:bigr)))
$(_tex(:Bigr))),
```
where ``a^{(k)}=p^{(k)}`` by default, but it allows to introduce some acceleration before
computing the gradient step.

# Input

$(_args([:M, :f]))
  total cost function ``f = g + h``
* `g`:              the smooth part of the cost function
* `grad_g`:           a gradient `(M,p) -> X` or `(M, X, p) -> X` of the smooth part ``g`` of the problem
$(_args(:p))

# Keyword arguments

* `acceleration=(p, s, k) -> (copyto!(get_manifold(M), s.a, s.p); s)`: a function `(problem, state, k) -> state` to compute an acceleration, that is performed before the gradient step - the default is to copy the current point to the acceleration point, i.e. no acceleration is performed
$(_kwargs(:callbacks; add_properties = [:process_note]))
$(_kwargs(:evaluation))
* `prox_nonsmooth = missing`:          a proximal map `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the (possibly) nonsmoooth part ``h`` of ``f``
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`ProximalGradientMethodState`](@ref)`)"))
  that by default uses a [`ProximalGradientMethodBacktracking`](@ref).
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(100)"))
$(_kwargs(:sub_problem; type = "Union{`[`AbstractManoptProblem`](@ref)`, F, Missing}", default = "missing"))
  or `missing` to take the proximal map from the [`ManifoldProximalGradientObjective`](@ref)
$(_kwargs(:sub_state; default = "evaluation")). This field is ignored, if the `sub_problem` is `missing`.

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_prox_grad_method)"
function proximal_gradient_method(
        M::AbstractManifold, f, g, grad_g, p = rand(M);
        prox_nonsmooth = missing, evaluation = AllocatingEvaluation(), kwargs...,
    )
    mpgo = ManifoldProximalGradientObjective(
        f, g, grad_g, prox_nonsmooth; evaluation = evaluation
    )
    return proximal_gradient_method(M, mpgo, p; evaluation = evaluation, kwargs...)
end

function proximal_gradient_method(
        M::AbstractManifold, mpgo::O, p = rand(M); kwargs...
    ) where {O <: Union{ManifoldProximalGradientObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(proximal_gradient_method; kwargs...)
    q = copy(M, p)
    return proximal_gradient_method!(M, mpgo, q; kwargs...)
end
calls_with_kwargs(::typeof(proximal_gradient_method)) = (proximal_gradient_method!,)

@doc "$(_doc_prox_grad_method)"
function proximal_gradient_method!(
        M::AbstractManifold, f, g, grad_g, p;
        prox_nonsmooth = missing, evaluation = AllocatingEvaluation(), kwargs...,
    )
    mpgo = ManifoldProximalGradientObjective(
        f, g, grad_g, prox_nonsmooth; evaluation = evaluation
    )
    return proximal_gradient_method!(M, mpgo, p; evaluation = evaluation, kwargs...)
end
function proximal_gradient_method!(
        M::AbstractManifold, mpgo::O, p;
        acceleration = function (pr, st, k)
            copyto!(get_manifold(pr), st.a, st.p)
            return st
        end,
        callbacks = Dict{Symbol, Function}(),
        debug = [DebugWarnIfStepsizeCollapsed()],
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, ProximalGradientMethodState
        ),
        cost_nonsmooth::Union{Missing, Function} = missing,
        subgradient_nonsmooth::Union{Missing, Function} = missing,
        stopping_criterion::S = StopWhenGradientMappingNormLess(1.0e-7) |
            StopAfterIteration(5000) |
            StopWhenChangeLess(M, 1.0e-9),
        X = zero_vector(M, p),
        retraction_method = default_retraction_method(M, typeof(p)),
        inverse_retraction_method = default_inverse_retraction_method(M, typeof(p)),
        sub_problem = if ismissing(mpgo.proximal_map_h!)
            DefaultManoptProblem(
                M,
                ManifoldSubgradientObjective(
                    ProximalGradientNonsmoothCost(cost_nonsmooth, 0.1, p),
                    ProximalGradientNonsmoothSubgradient(subgradient_nonsmooth, 0.1, p),
                ),
            )
        else
            missing
        end,
        sub_state = if !ismissing(mpgo.proximal_map_h!)
            AllocatingEvaluation()
        else
            SubGradientMethodState(
                M;
                p = p,
                stepsize = Manopt.DecreasingStepsize(
                    M; exponent = 1, factor = 1, subtrahend = 0, length = 1, shift = 0, type = :absolute
                ),
                stopping_criterion = StopAfterIteration(2500) | StopWhenSubgradientNormLess(1.0e-8),
            )
        end,
        kwargs...,
    ) where {
        O <: Union{ManifoldProximalGradientObjective, AbstractDecoratedManifoldObjective},
        S <: StoppingCriterion,
    }
    keywords_accepted(proximal_gradient_method!; kwargs...)
    # Check whether either the right defaults were provided or a `sub_problem`.
    if ismissing(mpgo.proximal_map_h!) && ismissing(cost_nonsmooth)
        error(
            """
            The `sub_problem` is not correctly initialized. Provide _one of_ the following setups
            * `prox_nonsmooth` keyword argument as a closed form solution,
            * `cost_nonsmooth` keyword argument for the (possibly nonsmooth) part of the cost function whose proximal map is to be computed,
            """,
        )
    end
    dmpgo = decorate_objective!(M, mpgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpgo)
    pgms = ProximalGradientMethodState(
        M;
        callbacks = process_callbacks_arg(callbacks, ProximalGradientMethodState),
        p = p,
        acceleration = acceleration,
        stepsize = _produce_type(stepsize, M, p),
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
        stopping_criterion = stopping_criterion,
        sub_problem = sub_problem,
        sub_state = sub_state,
        X = X,
    )
    dpgms = decorate_state!(pgms; debug = debug, kwargs...)
    solve!(dmp, dpgms)
    return get_solver_return(get_objective(dmp), dpgms)
end
calls_with_kwargs(::typeof(proximal_gradient_method!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(amp::AbstractManoptProblem, pgms::ProximalGradientMethodState)
    M = get_manifold(amp)
    zero_vector!(M, pgms.X, pgms.p)
    copyto!(M, pgms.a, pgms.p)
    initialize_stepsize!(pgms.stepsize)
    return pgms
end

function step_solver!(amp::AbstractManoptProblem, pgms::ProximalGradientMethodState, k)
    M = get_manifold(amp)
    # Store previous iterate
    copyto!(M, pgms.q, pgms.p)

    # (Possible) Acceleration
    pgms.acceleration(amp, pgms, k)

    # Evaluate the gradient at (possibly) accelerated point
    get_gradient!(amp, pgms.X, pgms.a)

    # Compute stepsize using the provided stepsize object
    pgms.last_stepsize = get_stepsize(amp, pgms, k)
    callback(:Stepsize, amp, pgms, k)

    # Gradient step with chosen stepsize
    retract!(M, pgms.a, pgms.a, -pgms.last_stepsize * pgms.X, pgms.retraction_method)

    # Proximal step with chosen stepsize
    callback(:BeforeSubsolver, amp, pgms, k)
    _pgm_proximal_step(amp, pgms, pgms.last_stepsize)
    callback(:Subsolver, amp, pgms, k)

    return pgms
end

# (I) Problem is missing -> use prox from objective
function _pgm_proximal_step(
        amp::AbstractManoptProblem, pgms::ProximalGradientMethodState{P, T, Missing}, λ::Real
    ) where {P, T}
    get_proximal_map!(amp, pgms.p, λ, pgms.a)
    return pgms
end

# (II) Problem is a subsolver -> solve
function _pgm_proximal_step(
        amp::AbstractManoptProblem,
        pgms::ProximalGradientMethodState{P, T, <:AbstractManoptProblem, <:AbstractManoptSolverState},
        λ::Real,
    ) where {P, T}
    M = get_manifold(amp)
    # set lambda
    set_parameter!(pgms.sub_problem, Val(:Objective), Val(:Cost), Val(:λ), λ)
    set_parameter!(pgms.sub_problem, Val(:Objective), Val(:SubGradient), Val(:λ), λ)
    # set the proximity point of the subproblem
    set_parameter!(pgms.sub_problem, Val(:Objective), Val(:Cost), Val(:proximity_point), pgms.a)
    set_parameter!(pgms.sub_problem, Val(:Objective), Val(:SubGradient), Val(:proximity_point), pgms.a)
    # set start value to a
    set_iterate!(pgms.sub_state, M, copy(M, pgms.a))
    solve!(pgms.sub_problem, pgms.sub_state)
    copyto!(M, pgms.p, get_solver_result(pgms.sub_state))
    return pgms
end
