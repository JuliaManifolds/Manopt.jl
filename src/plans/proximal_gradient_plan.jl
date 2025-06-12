@doc raw"""
    ManifoldProximalGradientObjective{E,<:AbstractEvaluationType, TC, TG, TGG, TP} <: AbstractManifoldObjective{E,TC,TGG}

Model an objective of the form
```math
    f(p) = g(p) + h(p),\qquad p \in \mathcal M,
```
where ``g: \mathcal M → \bar{ℝ}`` is a differentiable function
and ``h: → \bar{ℝ}`` is a (possibly) lower semicontinous, and proper function.

This objective provides the total cost ``f``, its smooth component ``g``,
as well as ``\operatorname{grad} g`` and ``\operatorname{prox}_{λ} h``.

# Fields

* `cost`: the overall cost ``f = g + h``
* `cost_g`: the smooth cost component ``g``
* `gradient_g!!`: the gradient ``\operatorname{grad} g``
* `proximal_map_h!!`: the proximal map ``\operatorname{prox}_{λ} h``

# Constructor
    ManifoldProximalGradientObjective(f, g, grad_g, prox_h;
        evalauation=[`AllocatingEvaluation`](@ref)
    )

Generate the proximal gradient objective given the total cost `f = g + h`, smooth cost `g`, the gradient of the smooth component `grad_g`, and the proximal map of the nonsmooth component `prox_h`.

## Keyword arguments

* `evaluation=`[`AllocatingEvaluation`](@ref): whether the gradient and proximal map
  is given as an allocation function or an in-place ([`InplaceEvaluation`](@ref)).
"""
struct ManifoldProximalGradientObjective{E<:AbstractEvaluationType,TC,TG,TGG,TP} <:
       AbstractManifoldCostObjective{E,TC}
    cost::TC # f = g + h
    cost_g::TG # smooth part
    gradient_g!!::TGG
    proximal_map_h!!::TP
    function ManifoldProximalGradientObjective(
        f::TC, g::TG, grad_g::TGG, prox_h::TP; evaluation::E=AllocatingEvaluation()
    ) where {TC,TG,TGG,TP,E<:AbstractEvaluationType}
        return new{E,TC,TG,TGG,TP}(f, g, grad_g, prox_h)
    end
end

"""
    get_gradient(M::AbstractManifold, mgo::ManifoldProximalGradientObjective, p)
    get_gradient!(M::AbstractManifold, X, mgo::ManifoldProximalGradientObjective, p)

Evaluate the gradient of the smooth part of a [`ManifoldProximalGradientObjective`](@ref) `mgo` at `p`.
"""
function get_gradient(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, p
)
    return mpgo.gradient_g!!(M, p)
end

function get_gradient(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    mpgo.gradient_g!!(M, X, p)
    return X
end

function get_gradient!(
    M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, p
)
    copyto!(M, X, p, mpgo.gradient_g!!(M, p))
    return X
end

function get_gradient!(
    M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, p
)
    mpgo.gradient_g!!(M, X, p)
    return X
end

"""
    get_cost_g(M::AbstractManifold, mpgo::ManifoldProximalGradientObjective, p)

Evaluate the smooth part `g` of the cost function at point `p`.
"""
function get_cost_g(M::AbstractManifold, mpgo::ManifoldProximalGradientObjective, p)
    return mpgo.cost_g(M, p)
end

"""
    get_cost_h(M::AbstractManifold, mpgo::ManifoldProximalGradientObjective, p)

Evaluate the nonsmooth part `h` of the cost function at point `p`.
"""
function get_cost_h(M::AbstractManifold, mpgo::ManifoldProximalGradientObjective, p)
    return mpgo.cost_h(M, p)
end

"""
    get_cost_g(M::AbstractManifold, objective, p)

Helper function to extract the smooth part g of a proximal gradient objective.
Falls back to normal cost function if the objective doesn't have a separate g component.
"""
function get_cost_g(M::AbstractManifold, objective, p)
    if hasfield(typeof(objective), :cost_g)
        return objective.cost_g(M, p)
    else
        # Fallback: use the regular cost function
        return get_cost(M, objective, p)
    end
end

@doc raw"""
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalGradientObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalGradientObjective, λ, p)

Evaluate proximal map of the nonsmooth component ``h`` of the [`ManifoldProximalGradientObjective`](@ref)` mpo`
at the point `p` on `M` with parameter ``λ>0``.
"""
function get_proximal_map(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, λ, p
)
    return mpgo.proximal_map_h!!(M, λ, p)
end

function get_proximal_map!(
    M::AbstractManifold,
    q,
    mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation},
    λ,
    p,
)
    copyto!(M, q, mpgo.proximal_map_h!!(M, λ, p))
    return q
end

function get_proximal_map(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, λ, p
)
    q = allocate_result(M, get_proximal_map, p)
    mpgo.proximal_map_h!!(M, q, λ, p)
    return q
end

function get_proximal_map!(
    M::AbstractManifold, q, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, λ, p
)
    mpgo.proximal_map_h!!(M, q, λ, p)
    return q
end
#
# Method State
@doc """
    ProximalGradientMethodState <: AbstractManoptSolverState

State for the [`proximal_gradient_method`](@ref) solver.

# Fields

$(_var(:Field, :inverse_retraction_method))
* `a` - point after acceleration step
$(_var(:Field, :p; add=[:as_Iterate]))
* `q` - point for storing gradient step
$(_var(:Field, :retraction_method))
* `X` - tangent vector for storing gradient
$(_var(:Field, :stopping_criterion, "stop"))
* `acceleration` - a function `(problem, state, k) -> state` to compute an acceleration before the gradient step
* `stepsize` - a function or [`Stepsize`](@ref) object to compute the stepsize
* `last_stepsize` - stores the last computed stepsize
$(_var(:Field, :sub_problem, "sub_problem", "Union{AbstractManoptProblem, F}"; add="or nothing to take the proximal map from the [`ManifoldProximalGradientObjective`](@ref)"))
$(_var(:Field, :sub_state; add="This field is ignored, if the `sub_problem` is `Nothing`"))

# Constructor

    ProximalGradientMethodState(M::AbstractManifold; kwargs...)

Generate the state for a given manifold `M` with initial iterate `p`.

## Input

$(_var(:Argument, :M; type=true))

# Keyword arguments

* `stepsize=default_stepsize(M, ProximalGradientMethodState)`
$(_var(:Field, :inverse_retraction_method))
$(_var(:Keyword, :p; add=:as_Initial))
$(_var(:Keyword, :retraction_method))
* `acceleration=(p, s, k) -> (copyto!(get_manifold(M), s.a, s.p); s)` by default no acceleration is performed
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(100)`"))
$(_var(:Keyword, :sub_problem; default="nothing"))
$(_var(:Keyword, :sub_state; default=_var(:evaluation, :default)))
$(_var(:Keyword, :X; add=:as_Memory))
"""
mutable struct ProximalGradientMethodState{
    P,
    T,
    Pr<:Union{<:AbstractManoptProblem,F,Nothing} where {F},
    St<:AbstractManoptSolverState,
    A,
    S<:StoppingCriterion,
    TStepsize<:Stepsize,
    RM<:AbstractRetractionMethod,
    IRM<:AbstractInverseRetractionMethod,
    R,
} <: AbstractManoptSolverState
    a::P
    acceleration::A
    stepsize::TStepsize
    last_stepsize::R
    p::P
    q::P
    stop::S
    X::T
    retraction_method::RM
    inverse_retraction_method::IRM
    sub_problem::Pr
    sub_state::St
end

function ProximalGradientMethodState(
    M::AbstractManifold;
    p::P=rand(M),
    acceleration::A=function (pr, st, k)
        copyto!(get_manifold(pr), st.a, st.p)
        return st
    end,
    stepsize::TS=default_stepsize(M, ProximalGradientMethodState),
    stopping_criterion::S=StopWhenGradientMappingNormLess(1e-2) |
                          StopAfterIteration(5000) |
                          StopWhenChangeLess(M, 1e-9),
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    inverse_retraction_method::IRM=default_inverse_retraction_method(M, typeof(p)),
    sub_problem::Pr=nothing,
    sub_state::St=AllocatingEvaluation(),
) where {
    P,
    T,
    S<:StoppingCriterion,
    A,
    Pr<:Union{<:AbstractManoptProblem,F,Nothing} where {F},
    St<:Union{<:AbstractManoptSolverState,<:AbstractEvaluationType},
    RM<:AbstractRetractionMethod,
    IRM<:AbstractInverseRetractionMethod,
    TS<:Stepsize,
}
    _sub_state = if sub_state isa AbstractEvaluationType
        ClosedFormSubSolverState(; evaluation=sub_state)
    else
        sub_state
    end

    last_stepsize = zero(number_eltype(p))
    return ProximalGradientMethodState{
        P,T,Pr,typeof(_sub_state),A,S,TS,RM,IRM,typeof(last_stepsize)
    }(
        copy(M, p),
        acceleration,
        stepsize,
        last_stepsize,
        p,
        copy(M, p),
        stopping_criterion,
        X,
        retraction_method,
        inverse_retraction_method,
        sub_problem,
        _sub_state,
    )
end

get_iterate(pgms::ProximalGradientMethodState) = pgms.p

function set_iterate!(pgms::ProximalGradientMethodState, p)
    pgms.p = p
    return p
end

function show(io::IO, pgms::ProximalGradientMethodState)
    i = get_count(pgms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pgms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Proximal Gradient Method
    $Iter

    ## Parameters

    * retraction_method:              $(pgms.retraction_method)
    * stepsize:                       $(typeof(pgms.stepsize))
    * acceleration:                   $(typeof(pgms.acceleration))

    ## Stopping criterion

    $(status_summary(pgms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
#
# Stepsize
@doc raw"""
    ProximalGradientMethodBacktracking <: Stepsize

A functor for backtracking line search in proximal gradient methods.

# Fields

* `initial_stepsize::T` - initial step size guess
* `sufficient_decrease::T` - sufficient decrease parameter (default: 0.5)
* `contraction_factor::T` - step size reduction factor (default: 0.5)
* `strategy::Symbol` - `:nonconvex` or `:convex` (default: `:convex`)
* `candidate_point::P` - a working point used during backtracking
* `last_stepsize::T` - the last computed stepsize

# Constructor
    ProximalGradientMethodBacktracking(M::AbstractManifold; kwargs...)

## Keyword arguments

* `initial_stepsize=1.0`: initial stepsize to try
* `stop_when_stepsize_less=1e-4`: smallest stepsize when to stop (the last one before is taken)
* `sufficient_decrease=0.5`: sufficient decrease parameter
* `contraction_factor=0.5`: step size reduction factor
* `strategy=:nonconvex`: backtracking strategy, either `:convex` or `:nonconvex`
"""
mutable struct ProximalGradientMethodBacktracking{P,T} <: Stepsize
    initial_stepsize::T
    sufficient_decrease::T
    contraction_factor::T
    strategy::Symbol
    candidate_point::P
    last_stepsize::T
    stop_when_stepsize_less::T

    function ProximalGradientMethodBacktracking(
        M::AbstractManifold;
        initial_stepsize::T=1.0,
        sufficient_decrease::T=0.5,
        contraction_factor::T=0.5,
        strategy::Symbol=:nonconvex,
        stop_when_stepsize_less::T=1e-4,
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

        p = rand(M)
        return new{typeof(p),T}(
            initial_stepsize,
            sufficient_decrease,
            contraction_factor,
            strategy,
            p,
            initial_stepsize,
            stop_when_stepsize_less,
        )
    end
end

get_initial_stepsize(s::ProximalGradientMethodBacktracking) = s.initial_stepsize

@doc """
    (s::ProximalGradientMethodBacktracking)(mp, st, i)

Compute a stepsize for the proximal gradient method using a backtracking line search.

For the nonconvex case, the condition is:

```math
f(p) - f(T_{λ}(p)) ≥ γλ$(_tex(:norm, "G_{λ}(p)"))^2
```

where `G_{λ}(p) = (1/λ) * $(_tex(:log))_p(T_{λ}(p))` is the gradient mapping.

For the convex case, the condition is:
```math
g(T_{λ}(p)) ≤ g(p) + ⟨$(_tex(:grad)) g(p), $(_tex(:log))_p T_{λ}(p)⟩ + $(_tex(:frac, "1", "2λ")) $(_math(:distance))^2(p, T_{λ}(p))
```

Returns a stepsize `λ` that satisfies the specified condition.
"""
function (s::ProximalGradientMethodBacktracking)(
    mp::AbstractManoptProblem, st::ProximalGradientMethodState, i::Int, args...; kwargs...
)
    # Initialization
    M = get_manifold(mp)
    p = st.a  # Current point (post-acceleration)
    X = st.X  # Current gradient

    # For convex case, start with the last stepsize (warm start)
    # For nonconvex case, reset to initial stepsize
    λ = if s.strategy === :convex && i > 1
        s.last_stepsize
    else
        s.initial_stepsize
    end

    # Get the objective and temporary state
    objective = get_objective(mp)

    # Temporary state for backtracking that doesn't affect the main state
    pgm_temp = ProximalGradientMethodState(
        M;
        p=copy(M, p),  # Start from current accelerated point
        X=zero_vector(M, p),
        sub_problem=st.sub_problem,
        sub_state=st.sub_state,
        retraction_method=st.retraction_method,
        inverse_retraction_method=st.inverse_retraction_method,
    )

    while λ > s.stop_when_stepsize_less
        # Perform gradient step with current λ
        retract!(M, pgm_temp.a, p, -λ * X, st.retraction_method)

        # Perform proximal step with current λ
        _pgm_proximal_step(mp, pgm_temp, λ)
        candidate_point = pgm_temp.p

        # Compute log_p(candidate_point) and its squared norm for the conditions
        log_p_q = inverse_retract(M, p, candidate_point, st.inverse_retraction_method)
        log_p_q_norm_squared = norm(M, p, log_p_q)^2

        if s.strategy === :nonconvex
            # Nonconvex descent condition
            if get_cost(mp, p) - get_cost(mp, candidate_point) >=
                (s.sufficient_decrease / λ) * log_p_q_norm_squared
                s.last_stepsize = λ
                return λ
            end
        else
            g_p = get_cost_g(M, objective, p)
            g_q = get_cost_g(M, objective, candidate_point)

            # Convex descent condition
            if g_q <= g_p + inner(M, p, X, log_p_q) + (1 / 2λ) * log_p_q_norm_squared
                s.last_stepsize = λ
                return λ
            end
        end

        # Reduce step size
        λ *= s.contraction_factor
    end
    return λ
end

function ProxGradBacktracking(args...; kwargs...)
    return ManifoldDefaultsFactory(
        Manopt.ProximalGradientMethodBacktracking, args...; kwargs...
    )
end

"""
    default_stepsize(M::AbstractManifold, ::Type{<:ProximalGradientMethodState})

Returns the default proximal stepsize, which is a nonconvex backtracking strategy.
"""
function default_stepsize(M::AbstractManifold, ::Type{<:ProximalGradientMethodState})
    return ProximalGradientMethodBacktracking(M; initial_stepsize=1.5, strategy=:nonconvex)
end

@doc """
    ProxGradAcceleration{P, T, F}

Compute an acceleration step

```math
a^{(k)} = $(_tex(:retr))_{p^{(k)}}$(_tex(:bigl))(
  -β_k$(_tex(:invretr))_{p^{(k)}}(p)
$(_tex(:bigr)))
```

where `p^{(k)}` is the current iterate from the [`ProximalGradientMethodState`](@ref)s
field `p` and the result is stored in `state.a`. The field `p` in this struct stores the last iterate.

The retraction and its inverse are taken from the state.

# Fields

* `p` - the last iterate
* `β` - acceleration parameter function or value
* `inverse_retraction_method` - method for inverse retraction
* `X` - tangent vector for computations

# Constructor

    ProxGradAcceleration(M::AbstractManifold; kwargs...)

Generate the state for a given manifold `M` with initial iterate `p`.

## Input

$(_var(:Argument, :M; type=true))

# Keyword arguments

* `β = k -> (k-1)/(k+2)` - acceleration parameter function or value
* `inverse_retraction_method` - method for inverse retraction
* `p` - initial point
* `X` - initial tangent vector
"""
mutable struct ProxGradAcceleration{P,T,F,ITR}
    β::F
    inverse_retraction_method::ITR
    p::P
    X::T
end

function ProxGradAcceleration(
    M::AbstractManifold;
    p::P=rand(M),
    X::T=zero_vector(M, p),
    β::F=(k) -> (k - 1) / (k + 2),
    inverse_retraction_method::I=default_inverse_retraction_method(M, typeof(p)),
) where {P,T,F,I<:AbstractInverseRetractionMethod}
    return ProxGradAcceleration{P,T,F,I}(β, inverse_retraction_method, p, X)
end

function (pga::ProxGradAcceleration)(
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

"""
    StopWhenGradientMappingNormLess <: StoppingCriterion

A stopping criterion based on the gradient mapping norm for proximal gradient methods.

# Fields

$(_var(:Field, :at_iteration))
$(_var(:Field, :last_change))
* `threshold`: the threshold for the change to check (run under to stop)

# Constructor

    StopWhenGradientMappingNormLess(ε)

Create a stopping criterion with threshold `ε` for the gradient mapping for the [`proximal_gradient_method`](@ref).
That is, this criterion indicates to stop when the gradient mapping has a norm less than `ε`.
The gradient mapping G_λ(p) is defined as -(1/λ) * log_p(T_λ(p)), where T_λ(p) is the proximal mapping prox_λ f(exp_p(-λ * grad f(p))).
"""
mutable struct StopWhenGradientMappingNormLess{TF} <: StoppingCriterion
    threshold::TF
    last_change::TF
    at_iteration::Int
    function StopWhenGradientMappingNormLess(ε::TF) where {TF}
        return new{TF}(ε, zero(ε), -1)
    end
end

function (sc::StopWhenGradientMappingNormLess)(
    mp::AbstractManoptProblem, s::ProximalGradientMethodState, i::Int
)
    M = get_manifold(mp)
    if i == 0 # reset on init
        sc.at_iteration = -1
    end
    if (i > 0)
        sc.last_change =
            1 / s.last_stepsize * norm(
                M, s.q, inverse_retract(M, s.q, get_iterate(s), s.inverse_retraction_method)
            )
        if sc.last_change < sc.threshold
            sc.at_iteration = i
            return true
        end
    end
    return false
end

function get_reason(c::StopWhenGradientMappingNormLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the gradient mapping norm ($(c.last_change)) is less than $(c.threshold).\n"
    end
    return ""
end

function status_summary(c::StopWhenGradientMappingNormLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|G| < $(c.threshold): $s"
end

indicates_convergence(c::StopWhenGradientMappingNormLess) = true

function show(io::IO, c::StopWhenGradientMappingNormLess)
    return print(
        io, "StopWhenGradientMappingNormLess($(c.threshold))\n    $(status_summary(c))"
    )
end
