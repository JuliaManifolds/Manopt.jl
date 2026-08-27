#
#
#
"""
    StepsizeMessage{TBound, TS}

A message struct to hold stepsize information, when e.g.
a step size underflow happens at a certain iteration.

# Fields
- `at_iteration::Int`: The iteration at which the message was set
- `bound::TBound`: The bound that was hit
- `value::TS`: The corresponding value that either caused the message or provides additional information

# Constructor

    StepsizeMessage(; bound::TBound = 0.0, value::TS = 0.0)

"""
mutable struct StepsizeMessage{TBound <: Real, TS <: Real}
    at_iteration::Int
    bound::TBound
    value::TS
end

function StepsizeMessage{TBound, TS}() where {TBound <: Real, TS <: Real}
    return StepsizeMessage{TBound, TS}(-1, zero(TS), zero(TS))
end

function StepsizeMessage(
        ; bound::TBound = 0.0, value::TS = 0.0
    ) where {TBound <: Real, TS <: Real}
    return StepsizeMessage{TBound, TS}(-1, bound, value)
end

"""
    get_message(a)

Given a certain structure `a` from within `Manopt.jl`, retrieve its last message of
information, e.g. warnings from a step size.
If no message is available, an empty string is returned.
"""
function get_message end

"""
    reset_messages!(messages::NamedTuple)

Given a named tuple of [`StepsizeMessage`](@ref)s, reset all messages to default values,
i.e. `at_iteration = -1`, `bound = 0`, `value = 0`.
"""
function reset_messages!(messages::NamedTuple)
    for m in messages
        m.at_iteration = -1
        m.bound = 0
        m.value = 0
    end
    return messages
end

"""
    set_message!(messages::NamedTuple, key::Symbol; at=nothing, bound=nothing, value=nothing)

Given a named tuple of [`StepsizeMessage`](@ref)s, set the message identified by `key` to the provided values,
i.e. if they are not `nothing`.
"""
function set_message!(
        messages::NamedTuple, key::Symbol;
        at::Union{Nothing, Int} = nothing,
        bound = nothing,
        value = nothing,
    )
    haskey(messages, key) && set_message!(messages[key], at, bound, value)
    return messages
end
"""
    set_message!(message::StepsizeMessage, at=nothing, bound=nothing, value=nothing)

Set the fields of a single [`StepsizeMessage`](@ref) to the provided values,
i.e. to those that are not `nothing`.
"""
function set_message!(
        msg::StepsizeMessage{TBound, TS},
        at::Union{Nothing, Int} = nothing,
        bound::Union{TBound, Nothing} = nothing,
        value::Union{TS, Nothing} = nothing
    ) where {TBound <: Real, TS <: Real}
    isnothing(at) || (msg.at_iteration = at)
    isnothing(bound) || (msg.bound = bound)
    return isnothing(value) || (msg.value = value)
end

#
#
# --- Initial Guess functions
"""
    ConstantInitialGuess{TF} <: AbstractInitialLinesearchGuess

Implement a constant initial guess for line searches.

# Constructor

    ConstantInitialGuess(α::TF)

where `α` is the constant initial step size.
"""
struct ConstantInitialGuess{TF} <: AbstractInitialLinesearchGuess
    α::TF
end
ConstantInitialGuess() = ConstantInitialGuess(1.0)

function (cig::ConstantInitialGuess)(
        ::AbstractManoptProblem, ::AbstractManoptSolverState, ::Int, ::Real, η; kwargs...
    )
    return cig.α
end

"""
    ArmijoInitialGuess <: AbstractInitialLinesearchGuess

Implement the initial guess for an Armijo line search.

The initial step size is chosen as `min(l, max_stepsize(M, p) / norm(M, p, η))`,
where `l` is the last step size used, `p` the current point and `η` the search direction.

The default provided is based on the [`max_stepsize`](@ref)`(M, p)`.

# Constructor

    ArmijoInitialGuess()
"""
struct ArmijoInitialGuess <: AbstractInitialLinesearchGuess end

function (::ArmijoInitialGuess)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, ::Int, l::Real, η; kwargs...
    )
    M = get_manifold(mp)
    X = get_gradient(s)
    p = get_iterate(s)
    grad_norm = norm(M, p, X)
    max_step = max_stepsize(M, p)
    return ifelse(isfinite(max_step), min(l, max_step / grad_norm), l)
end


#
#
# --- Displaying concrete messages
"""
    get_message(s::Symbol, args...)

For a certain set of symbols `s`, this message function turns
them into human readable strings. The arguments usually contain
an iteration number `k` or bounds to communicate to the user.
"""
get_message(s::Symbol, args...) = get_message(Val(s), args...)
get_message(s::Symbol, msg::StepsizeMessage) = get_message(Val(s), msg.at_iteration, msg.value, msg.bound)

"""
    get_message(:non_descent_direction, k::Int)

Display a message string for a non-descent direction encountered at iteration `k`.
"""
function get_message(::Val{:non_descent_direction}, k::Int = -1, value::Real = NaN, bound::Real = 0)
    (k < 0) && (return "")
    s = (k == 0) ? "the beginning" : "iteration #$k"
    v_str = isnan(value) ? "" : "(⟨η, grad_f(p)⟩ = $value ≥ $bound)"
    return (k >= 0) ? "At $s: Non-descent direction η encountered $v_str." : ""
end

"""
    get_message(:stepsize_exceeds, k::Int, step::Real = NaN, bound::Real = NaN)

Display a message string for a stepsize exceeding a certain bound at iteration `k`
and the step size `step` chosen instead.
"""
function get_message(::Val{:stepsize_exceeds}, k::Int = -1, value::Real = NaN, bound::Real = NaN)
    (k < 0) && (return "")
    s = (k == 0) ? "the beginning" : "iteration #$k"
    s_str = isnan(value) ? "" : "Reducing to $value"
    b_str = isnan(bound) ? "" : "($bound)"
    return (k > 0) ? "At $s: Maximal step size bound $b_str exceeded. $s_str." : ""
end
"""
    get_message(:stop_decreasing, k::Int=-1, step::Real = NaN)

Display a message string for stopping the decrease of the step size at iteration `k`
and the step size `step` chosen instead.
"""
function get_message(::Val{:stop_decreasing}, k::Int = -1, value::Real = NaN, bound::Int = -1)
    (k < 0) && (return "")
    s = (k == 0) ? "the beginning" : "iteration #$k"
    s_str = isnan(bound) ? "" : "($bound)"
    v_str = isnan(value) ? "" : "Continuing with a stepsize of $value."
    return (k > 0) ? "At $s: Maximal number of decrease steps $s_str reached. Aborting decrease. $v_str" : ""
end
"""
    get_message(:stop_increasing, k::Int=-1, step::Real = NaN)

Display a message string for stopping the increase of the step size at iteration `k`
and the step size `step` chosen instead.
"""
function get_message(::Val{:stop_increasing}, k::Int = -1, value::Real = NaN, bound::Int = -1)
    (k < 0) && (return "")
    s = (k == 0) ? "the beginning" : "iteration #$k"
    s_str = isnan(bound) ? "" : "($bound)"
    v_str = isnan(value) ? "" : "Continuing with a stepsize of $value."
    return (k > 0) ? "At $s: Maximal number of increase steps $s_str reached. Aborting increase. $v_str" : ""
end
"""
    get_message(:stepsize_less, k::Int=-1, step::Real = NaN, bound::Real = NaN)

Display a message string for the step size falling below its minimal bound at iteration `k`
and the step size `step` used instead.
"""
function get_message(::Val{:stepsize_less}, k::Int = -1, value::Real = NaN, bound::Real = NaN)
    (k < 0) && (return "")
    s = (k == 0) ? "the beginning" : "iteration #$k"
    s_str = isnan(value) ? "" : " Falling back to a stepsize of $value."
    b_str = isnan(bound) ? "" : "($bound)"
    return (k > 0) ? "At $s: Minimal stepsize less than bound $b_str reached.$s_str" : ""
end

#
#
# ---
@doc """
    ArmijoLinesearchStepsize <: Linesearch

A functor `(problem, state, k, X; kwargs...) -> s` to provide an Armijo line search to compute a step size,
based on the search direction `X`.

# Fields

* `additional_decrease_condition`: specify a condition a new point has to additionally
  fulfill. The default accepts all points.
* `additional_increase_condition`: specify a condition that additionally to
  checking a valid increase has to be fulfilled. The default accepts all points.
* `candidate_point`:               to store an interim result
* `initial_stepsize`:              an initial step size
$(_kwargs(:retraction_method))
* `contraction_factor`:            factor the step size is multiplied with in the backtracking loop
* `sufficient_decrease`:           gain within Armijo's rule
* `last_stepsize`:                 the last step size to start the search with
$(_kwargs(:initial_guess))
* `messages::NamedTuple`:          a named tuple to store possible [`StepsizeMessage`](@ref) about the stepsize search.
* `stop_when_stepsize_less`:       smallest stepsize when to stop (the last one before is taken)
* `stop_when_stepsize_exceeds`:    largest stepsize when to stop.
* `stop_increasing_at_step`:       last step to increase the stepsize (phase 1),
* `stop_decreasing_at_step`:       last step size to decrease the stepsize (phase 2),

Pass `:Messages` to a `debug=` to see `@info`s when these happen.

# Constructor

    ArmijoLinesearchStepsize(M::AbstractManifold; kwargs...)

where the fields are set from the keyword arguments below and the retraction defaults to the
default retraction on `M`.

## Keyword arguments

* `candidate_point=allocate_result(M, rand)`
* `initial_stepsize=1.0`
$(_kwargs(:retraction_method))
* `contraction_factor=0.95`
* `sufficient_decrease=0.1`
* `last_stepsize=initial_stepsize`
* `initial_guess=`[`ArmijoInitialGuess`](@ref)`()`
* `stop_when_stepsize_less=0.0`: stop when the stepsize decreased below this value.
* `stop_when_stepsize_exceeds=`[`max_stepsize`](@ref)`(M)`: provide an absolute maximal step size.
* `stop_increasing_at_step=100`: for the initial increase test, stop after these many steps
* `stop_decreasing_at_step=1000`: in the backtrack, stop after these many steps
"""
mutable struct ArmijoLinesearchStepsize{TRM <: AbstractRetractionMethod, P, I, F <: Real, IGF, DF, IF, MSGS} <:
    Linesearch
    candidate_point::P
    contraction_factor::F
    initial_guess::IGF
    initial_stepsize::F
    last_stepsize::F
    retraction_method::TRM
    sufficient_decrease::F
    stop_when_stepsize_less::F
    stop_when_stepsize_exceeds::F
    stop_increasing_at_step::I
    stop_decreasing_at_step::I
    additional_decrease_condition::DF
    additional_increase_condition::IF
    messages::MSGS
    function ArmijoLinesearchStepsize(;
            additional_decrease_condition::DF, additional_increase_condition::IF,
            candidate_point::P, contraction_factor::F, initial_stepsize::F, last_stepsize::F,
            initial_guess::IGF, retraction_method::TRM,
            stop_when_stepsize_less::F, stop_when_stepsize_exceeds::F, sufficient_decrease::F,
            stop_increasing_at_step::I, stop_decreasing_at_step::I, messages::MSGS
        ) where {TRM <: AbstractRetractionMethod, P, I <: Integer, F <: Real, IGF, DF, IF, MSGS}
        return new{TRM, P, I, F, IGF, DF, IF, MSGS}(
            candidate_point, contraction_factor, initial_guess, initial_stepsize,
            last_stepsize, retraction_method, sufficient_decrease,
            stop_when_stepsize_less, stop_when_stepsize_exceeds, stop_increasing_at_step, stop_decreasing_at_step,
            additional_decrease_condition, additional_increase_condition, messages,
        )
    end
    function ArmijoLinesearchStepsize(
            M::AbstractManifold;
            additional_decrease_condition::DF = (M, p) -> true, additional_increase_condition::IF = (M, p) -> true,
            candidate_point::P = allocate_result(M, rand),
            contraction_factor::Real = 0.95, initial_stepsize::Real = 1.0, last_stepsize::Real = initial_stepsize,
            initial_guess::IGF = ArmijoInitialGuess(), retraction_method::TRM = default_retraction_method(M),
            stop_when_stepsize_less::Real = 0.0, stop_when_stepsize_exceeds::Real = max_stepsize(M),
            stop_increasing_at_step::Integer = 100, stop_decreasing_at_step::Integer = 1000,
            sufficient_decrease::Real = 0.1,
        ) where {TRM <: AbstractRetractionMethod, P, IGF, DF, IF}
        R = promote_type(
            typeof(contraction_factor), typeof(initial_stepsize), typeof(last_stepsize),
            typeof(stop_when_stepsize_exceeds), typeof(stop_when_stepsize_less), typeof(sufficient_decrease),
        )
        cf = convert(R, contraction_factor); is = convert(R, initial_stepsize); ls = convert(R, last_stepsize)
        swse = convert(R, stop_when_stepsize_exceeds); swsl = convert(R, stop_when_stepsize_less)
        sd = convert(R, sufficient_decrease)
        I = promote_type(typeof(stop_increasing_at_step), typeof(stop_decreasing_at_step))
        sias = convert(I, stop_increasing_at_step); sdas = convert(I, stop_decreasing_at_step)
        msgs = (;
            non_descent_direction = StepsizeMessage{R, R}(),
            stop_decreasing = StepsizeMessage{I, R}(), stop_increasing = StepsizeMessage{I, R}(),
            stepsize_less = StepsizeMessage{R, R}(), stepsize_exceeds = StepsizeMessage{R, R}(),
        )
        return ArmijoLinesearchStepsize(;
            additional_decrease_condition = additional_decrease_condition,
            additional_increase_condition = additional_increase_condition,
            candidate_point = maybe_wrap_variable(candidate_point), contraction_factor = cf, initial_stepsize = is, last_stepsize = ls,
            initial_guess = initial_guess, retraction_method = retraction_method,
            stop_when_stepsize_less = swsl, stop_when_stepsize_exceeds = swse, sufficient_decrease = sd,
            stop_increasing_at_step = sias, stop_decreasing_at_step = sdas, messages = msgs
        )
    end
end
function ArmijoLinesearchStepsize(M::AbstractManifold, p; kwargs...)
    return ArmijoLinesearchStepsize(M; candidate_point = copy(M, p), kwargs...)
end
function (a::ArmijoLinesearchStepsize)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int, η = (-get_gradient(mp, get_iterate(s)));
        gradient = nothing, kwargs...,
    )
    p = get_iterate(s)
    grad = isnothing(gradient) ? get_gradient(mp, get_iterate(s)) : gradient
    return a(mp, p, grad, η; initial_guess = a.initial_guess(mp, s, k, a.last_stepsize, η), kwargs...)
end
function (a::ArmijoLinesearchStepsize)(
        mp::AbstractManoptProblem, p, X, η; initial_guess::Real = 1.0,
        stop_when_stepsize_exceeds = nothing, kwargs...
    )
    reset_messages!(a.messages)
    l = norm(get_manifold(mp), p, η)
    swse = if isnothing(stop_when_stepsize_exceeds)
        (a.stop_when_stepsize_exceeds / l)
    else
        stop_when_stepsize_exceeds
    end
    a.last_stepsize = linesearch_backtrack!(
        get_manifold(mp), a.candidate_point,
        (M, p) -> get_cost_function(get_objective(mp))(M, p), p,
        initial_guess, a.sufficient_decrease, a.contraction_factor, η;
        gradient = X, retraction_method = a.retraction_method,
        stop_when_stepsize_less = (a.stop_when_stepsize_less / l),
        stop_when_stepsize_exceeds = swse,
        stop_increasing_at_step = a.stop_increasing_at_step,
        stop_decreasing_at_step = a.stop_decreasing_at_step,
        additional_decrease_condition = a.additional_decrease_condition,
        additional_increase_condition = a.additional_increase_condition,
        report_messages_in = a.messages,
    )
    return a.last_stepsize
end
get_initial_stepsize(a::ArmijoLinesearchStepsize) = a.initial_stepsize
function get_last_stepsize(step::ArmijoLinesearchStepsize, ::Any...)
    return step.last_stepsize
end
function Base.show(io::IO, a_ls::ArmijoLinesearchStepsize)
    print(io, "ArmijoLinesearch(; additional_decrease_condition = ", a_ls.additional_decrease_condition)
    print(io, ", additional_increase_condition = ", a_ls.additional_increase_condition)
    print(io, ", candidate_point = ", a_ls.candidate_point, ", contraction_factor = ", a_ls.contraction_factor)
    print(io, ", initial_stepsize = ", a_ls.initial_stepsize, ", initial_guess = ", a_ls.initial_guess)
    print(io, ", last_stepsize = ", a_ls.last_stepsize)
    print(io, ", retraction_method = ", a_ls.retraction_method, ", stop_when_stepsize_less = ", a_ls.stop_when_stepsize_less)
    print(io, ", stop_when_stepsize_exceeds = ", a_ls.stop_when_stepsize_exceeds, ", sufficient_decrease = ", a_ls.sufficient_decrease)
    print(io, ", stop_increasing_at_step = ", a_ls.stop_increasing_at_step, ", stop_decreasing_at_step = ", a_ls.stop_decreasing_at_step)
    return print(io, ", messages = ", a_ls.messages, ")")
end
function status_summary(a_ls::ArmijoLinesearchStepsize; context::Symbol = :default)
    (context === :short) && return repr(a_ls)
    (context === :inline) && return "An Armijo backtracking line search (last stepsize: $(a_ls.last_stepsize))"
    return """
    Armijo backtracking line search
    A line search based on sufficient decrease backtracking (last stepsize: $(a_ls.last_stepsize))

    ## Parameters
    * contraction_factor:  $(_MANOPT_INDENT)$(a_ls.contraction_factor)
    * initial guess:       $(_MANOPT_INDENT)$(a_ls.initial_guess)
    * initial stepsize:    $(_MANOPT_INDENT)$(a_ls.initial_stepsize)
    * retraction method:   $(_MANOPT_INDENT)$(a_ls.retraction_method)
    * sufficient decrease: $(_MANOPT_INDENT)$(a_ls.sufficient_decrease)
    """
end
function get_message(a::ArmijoLinesearchStepsize)
    s = [get_message(kv[1], kv[2]) for kv in pairs(a.messages)]
    return join([m for m in s if length(m) > 0], "\n")
end
function get_parameter(a::ArmijoLinesearchStepsize, ::Val{:DecreaseCondition}, args...)
    return get_parameter(a.additional_decrease_condition, args...)
end
function get_parameter(a::ArmijoLinesearchStepsize, ::Val{:IncreaseCondition}, args...)
    return get_parameter(a.additional_increase_condition, args...)
end
function set_parameter!(a::ArmijoLinesearchStepsize, ::Val{:DecreaseCondition}, args...)
    set_parameter!(a.additional_decrease_condition, args...)
    return a
end
function set_parameter!(a::ArmijoLinesearchStepsize, ::Val{:IncreaseCondition}, args...)
    set_parameter!(a.additional_increase_condition, args...)
    return a
end
"""
    ArmijoLinesearch(; kwargs...)
    ArmijoLinesearch(M::AbstractManifold; kwargs...)

Specify a step size that performs an Armijo line search. It is given a function ``f:$(_math(:Manifold))→ℝ``
and its Riemannian gradient ``$(_tex(:grad))f: $(_math(:Manifold))→$(_math(:TangentBundle))``,
the current point ``p∈$(_math(:Manifold))`` and a search direction ``X∈$(_math(:TangentSpace))``.

Then the step size ``s`` is found by reducing the initial step size ``s`` until

```math
f($(_tex(:retr))_p(sX)) ≤ f(p) - τs ⟨ X, $(_tex(:grad))f(p) ⟩_p
```

is fulfilled, for a sufficient decrease value ``τ ∈ (0,1)``.

To be a bit more optimistic, if ``s`` already fulfils this, a first search is done,
__increasing__ the given ``s`` until for a first time this step does not hold.

Overall, a step size is sought that provides _enough decrease_, see
[Boumal:2023; p. 58](@cite) for more information.

# Keyword arguments

* `additional_decrease_condition=(M, p) -> true`:
  specify an additional criterion that has to be met to accept a step size in the decreasing loop
* `additional_increase_condition::IF=(M, p) -> true`:
  specify an additional criterion that has to be met to accept a step size in the (initial) increase loop
* `candidate_point=allocate_result(M, rand)`:
  specify a point to be used as memory for the candidate points.
* `contraction_factor=0.95`: how to update ``s`` in the decrease step
* `initial_stepsize=1.0`: specify an initial step size
* `initial_guess=`[`ArmijoInitialGuess`](@ref)`()`: Compute the initial step size of
  a line search based on this function. See [`AbstractInitialLinesearchGuess`](@ref) for details.
$(_kwargs(:retraction_method))
* `stop_when_stepsize_less=0.0`: a safeguard, stop when the decreasing step is below this (nonnegative) bound.
* `stop_when_stepsize_exceeds=max_stepsize(M)`: a safeguard to not choose a too long step size when initially increasing
* `stop_increasing_at_step=100`: stop the initial increasing loop after this amount of steps. Set to `0` to never increase in the beginning
* `stop_decreasing_at_step=1000`: maximal number of Armijo decreases / tests to perform
* `sufficient_decrease=0.1`: the sufficient decrease parameter ``τ``

For the stop safe guards you can pass `:Messages` to a `debug=` to see `@info` messages when these happen.

$(_note(:ManifoldDefaultsFactory, "ArmijoLinesearchStepsize"))
"""
function ArmijoLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.ArmijoLinesearchStepsize, args...; requires_point = true, kwargs...)
end

@doc """
    AdaptiveWNGradientStepsize{I<:Integer,R<:Real,F<:Function} <: Stepsize

A functor `(problem, state, k, X) -> s` implementing the adaptive gradient method introduced by [GrapigliaStella:2023](@cite).
See [`AdaptiveWNGradient`](@ref) for the mathematical details.

# Fields

* `count_threshold::I`: an `Integer` for ``$(_tex(:hat, "c"))``
* `minimal_bound::R`: the value for ``b_{$(_tex(:text, "min"))}``
* `alternate_bound::F`: how to determine ``$(_tex(:hat, "b"))_k`` as a function of `(bmin, bk, hat_c) -> hat_bk`
* `gradient_reduction::R`: the gradient reduction factor threshold ``α ∈ [0,1)``
* `gradient_bound::R`: the bound ``b_k``.
* `weight::R`: ``ω_k``, initialized to ``ω_0 =`` `norm(M, p, X)` if this is not zero, `1.0` otherwise.
* `count::I`: ``c_k``, initialized to ``c_0 = 0``.

# Constructor

    AdaptiveWNGradientStepsize(M::AbstractManifold; kwargs...)

## Keyword arguments

* `adaptive=true`: switches the `gradient_reduction` ``α`` (if `true`) to `0`.
* `alternate_bound = (bk, hat_c) ->  min(gradient_bound == 0 ? 1.0 : gradient_bound, max(minimal_bound, bk / (3 * hat_c)))`
* `count_threshold=4`
* `gradient_reduction::R=adaptive ? 0.9 : 0.0`
* `gradient_bound=norm(M, p, X)`
* `minimal_bound=1e-4`
$(_kwargs(:p)) only used to define the `gradient_bound`
$(_kwargs(:X)) only used to define the `gradient_bound`
"""
mutable struct AdaptiveWNGradientStepsize{I <: Integer, R <: Real, F} <: Stepsize
    count_threshold::I
    minimal_bound::R
    alternate_bound::F
    gradient_reduction::R
    gradient_bound::R
    weight::R
    count::I
    function AdaptiveWNGradientStepsize(;
            count_threshold::I, minimal_bound::R, alternate_bound::F, gradient_reduction::R,
            gradient_bound::R, weight::R, count::I
        ) where {I <: Integer, R <: Real, F}
        return new{I, R, F}(
            count_threshold, minimal_bound, alternate_bound, gradient_reduction, gradient_bound, weight, count
        )
    end
end

function AdaptiveWNGradientStepsize(
        M::AbstractManifold;
        p = rand(M), X = zero_vector(M, p), adaptive::Bool = true,
        count_threshold::I = 4,
        minimal_bound::Real = 1.0e-4,
        gradient_reduction::Real = adaptive ? 0.9 : 0.0,
        gradient_bound::Real = norm(M, p, X),
        alternate_bound = (bk, hat_c) -> min(
            gradient_bound == 0 ? 1.0 : gradient_bound, max(minimal_bound, bk / (3 * hat_c))
        ), kwargs...,
    ) where {I <: Integer}
    R = promote_type(typeof(minimal_bound), typeof(gradient_reduction), typeof(gradient_bound))
    g = gradient_bound == 0 ? one(R) : convert(R, gradient_bound)
    return AdaptiveWNGradientStepsize(;
        count_threshold = count_threshold, count = zero(I),
        minimal_bound = convert(R, minimal_bound), alternate_bound = alternate_bound,
        gradient_reduction = convert(R, gradient_reduction), gradient_bound = g, weight = g,
    )
end
function AdaptiveWNGradientStepsize(M::AbstractManifold, p; kwargs...)
    return AdaptiveWNGradientStepsize(M; p = p, kwargs...)
end
function (awng::AdaptiveWNGradientStepsize)(
        mp::AbstractManoptProblem, s::AbstractGradientSolverState, i, args...;
        gradient = nothing, kwargs...,
    )
    grad = isnothing(gradient) ? get_gradient(mp, get_iterate(s)) : gradient
    M = get_manifold(mp)
    p = get_iterate(s)
    isnan(awng.weight) || (awng.weight = norm(M, p, grad)) # init ω_0
    if i == 0 # init fields
        awng.weight = norm(M, p, grad) # init ω_0
        (awng.weight == 0) && (awng.weight = 1.0)
        awng.count = 0
        return 1 / awng.gradient_bound
    end
    grad_norm = norm(M, p, grad)
    if grad_norm < awng.gradient_reduction * awng.weight # grad norm < αω_{k-1}
        if awng.count + 1 == awng.count_threshold
            awng.gradient_bound = awng.alternate_bound(
                awng.gradient_bound, awng.count_threshold
            )
            awng.weight = grad_norm
            awng.count = 0
        else
            awng.gradient_bound = awng.gradient_bound + grad_norm^2 / awng.gradient_bound
            #weight stays unchanged
            awng.count += 1
        end
    else
        awng.gradient_bound = awng.gradient_bound + grad_norm^2 / awng.gradient_bound
        #weight stays unchanged
        awng.count = 0
    end
    return 1 / awng.gradient_bound
end
get_initial_stepsize(awng::AdaptiveWNGradientStepsize) = 1 / awng.gradient_bound
get_last_stepsize(awng::AdaptiveWNGradientStepsize) = 1 / awng.gradient_bound
function Base.show(io::IO, awng::AdaptiveWNGradientStepsize)
    print(io, "AdaptiveWNGradientStepsize(; count_threshold = ", awng.count_threshold, ", count = ", awng.count)
    print(io, ", minimal_bound = ", awng.minimal_bound, ", alternate_bound = ", awng.alternate_bound)
    print(io, ", gradient_reduction = ", awng.gradient_reduction, ", gradient_bound = ", awng.gradient_bound)
    print(io, ", weight = ", awng.weight)
    return print(io, ")")
end
function status_summary(awng::AdaptiveWNGradientStepsize; context::Symbol = :default)
    (context === :short) && return repr(awng)
    (context === :inline) && return "An adaptive WN gradient step size"
    return """
    An adaptive WN gradient step size
    (last step size: $(1 / awng.gradient_bound))

    ## Parameters
    * count threshold:   $(_MANOPT_INDENT)$(awng.count_threshold)
    * minimal_bound:     $(_MANOPT_INDENT)$(awng.minimal_bound)
    * gradient reduction:$(_MANOPT_INDENT)$(awng.gradient_reduction)
    """
end
"""
    AdaptiveWNGradient(; kwargs...)
    AdaptiveWNGradient(M::AbstractManifold; kwargs...)

A stepsize based on the adaptive gradient method introduced by [GrapigliaStella:2023](@cite).

Given a positive threshold ``$(_tex(:hat, "c")) ∈ ℕ``,
an minimal bound ``b_{$(_tex(:text, "min"))} > 0``,
an initial ``b_0 ≥ b_{$(_tex(:text, "min"))}``, and a
gradient reduction factor threshold ``α ∈ [0,1)``.

Set ``c_0=0`` and use ``ω_0 = $(_tex(:norm, "$(_tex(:grad)) f(p_0)"; index = "p_0"))``.

For the first iterate use the initial step size ``s_0 = $(_tex(:frac, "1", "b_0"))``.

Then, given the last gradient ``X_{k-1} = $(_tex(:grad)) f(x_{k-1})``,
and a previous ``ω_{k-1}``, the values ``(b_k, ω_k, c_k)`` are computed
using ``X_k = $(_tex(:grad)) f(p_k)`` and the following cases

If ``$(_tex(:norm, "X_k"; index = "p_k")) ≤ αω_{k-1}``, then let
``$(_tex(:hat, "b"))_{k-1} ∈ [b_{$(_tex(:text, "min"))},b_{k-1}]`` and set

```math
(b_k, ω_k, c_k) = $(
    _tex(
        :cases,
        "$(_tex(:bigl))($(_tex(:hat, "b"))_{k-1}, $(_tex(:norm, "X_k"; index = "p_k")), 0 $(_tex(:bigr))) & $(_tex(:text, " if ")) c_{k-1}+1 = $(_tex(:hat, "c"))",
        "$(_tex(:bigl))( b_{k-1} + $(_tex(:frac, _tex(:norm, "X_k"; index = "p_k") * "^2", "b_{k-1}")), ω_{k-1}, c_{k-1}+1 $(_tex(:Bigr))) & $(_tex(:text, " if ")) c_{k-1}+1<$(_tex(:hat, "c"))",
    )
)
```

If ``$(_tex(:norm, "X_k"; index = "p_k")) > αω_{k-1}``, the set

```math
(b_k, ω_k, c_k) = $(_tex(:Bigl))( b_{k-1} + $(_tex(:frac, _tex(:norm, "X_k"; index = "p_k") * "^2", "b_{k-1}")), ω_{k-1}, 0 $(_tex(:Bigr)))
```

and return the step size ``s_k = $(_tex(:frac, "1", "b_k"))``.

Note that for ``α=0`` this is the Riemannian variant of `WNGRad`.

## Keyword arguments

* `adaptive=true`: switches the `gradient_reduction ``α`` (if `true`) to `0`.
* `alternate_bound = (bk, hat_c) ->  min(gradient_bound == 0 ? 1.0 : gradient_bound, max(minimal_bound, bk / (3 * hat_c))`:
  how to determine ``$(_tex(:hat, "k"))_k`` as a function of `(bmin, bk, hat_c) -> hat_bk`
* `count_threshold=4`:  an `Integer` for ``$(_tex(:hat, "c"))``
* `gradient_reduction::R=adaptive ? 0.9 : 0.0`: the gradient reduction factor threshold ``α ∈ [0,1)``
* `gradient_bound=norm(M, p, X)`: the bound ``b_k``.
* `minimal_bound=1e-4`: the value ``b_{$(_tex(:text, "min"))}``
$(_kwargs(:p)) only used to define the `gradient_bound`
$(_kwargs(:X)) only used to define the `gradient_bound`
"""
function AdaptiveWNGradient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.AdaptiveWNGradientStepsize, args...; requires_point = true, kwargs...)
end

## TODO Introduce the factory as well: BarzileiBorwein and document the formula there.

@doc """
    BarzileiBorweinStepsize{T, R<:Real, IRM, RM, VTM, TSSA} <: Stepsize

Compute a stepsize based on the Barzilei-Borwein rule.

Consider the current iterate and gradient ``p_k`` and ``X_k`` as well as the last
iterate and gradient ``p_{k-1}`` and ``X_{k-1}``. Note that in a gradient scheme this also
yields the relation that ``p_k = $(_tex(:exp))_{p_{k-1}}(αX_{k-1}))`` when ``α`` is the stepsize
from the last iteration.

We compute the changes of the iterates and the vectors, respectively

```math
s = $(_tex(:invretr))_{p_{k}}(p_{k-1})
$(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad))
y_{k} = X_k - $(_math(:VectorTransport, "p_{k-1}", "p_k"))(X_{k-1}),
```
where alternatively ``s = α$(_math(:VectorTransport, "p_{k-1}", "p_k"))(X_{k-1})`` if the `last_stepsize =`
was passed to the function.

Then the Barzilai—Borwein step size is

```math
α^{$(_tex(:text, "BB"))} = $(
    _tex(
        :cases,
        "$(_tex(:min))(α_{$(_tex(:text, "max"))}, $(_tex(:max))(α_{$(_tex(:text, "min"))}, τ_{k})), & $(_tex(:text, "if")) ⟨s_{k}, y_{k}⟩_{p_k} > 0,",
        "α_{$(_tex(:text, "max"))}, & $(_tex(:text, "else,"))"
    )
)
```

where

```math
τ_{k} = $(_tex(:frac, "⟨s, s⟩_{p_k}", "⟨s, y⟩_{p_k}")),
```

for the `:direct` strategy, or

```math
τ_{k} =  $(_tex(:frac, "⟨s, y⟩_{p_k}", "⟨y, y⟩_{p_k}")),
```

for the `:inverse` strategy. The `:alternating` strategy uses the direct for odd, the inverse for even iterations `k`.

# Fields

$(_fields(:inverse_retraction_method))
* `min_stepsize`:          lower bound ``α_{$(_tex(:text, "min"))}`` for the Barzilai-Borwein step size, greater than zero
* `max_stepsize`:          upper bound ``α_{$(_tex(:text, "max"))}`` for the Barzilai-Borwein step size, greater than ``α_{$(_tex(:text, "min"))}``.
$(_fields(:retraction_method))
* `strategy`:                 defines if the new step size is computed using the `:direct`, `:inverse` or `:alternating` strategy
* `storage`:                  (for `:Iterate` and `:Gradient`) a [`StoreStateAction`](@ref)
$(_fields(:vector_transport_method))

# Constructor

    BarzileiBorweinStepsize(M::AbstractManifold; kwargs...)
    BarzileiBorweinStepsize(M::AbstractManifold, p; kwargs...)

## Keyword arguments

$(_kwargs(:inverse_retraction_method))
* `min_stepsize=1e-3`
* `max_stepsize=1e3`
$(_kwargs(:retraction_method))
* `strategy=:direct`
* `storage=`[`StoreStateAction`](@ref)`(M; store_fields=[:Iterate, :Gradient])`
$(_kwargs(:vector_transport_method))
"""
mutable struct BarzileiBorweinStepsize{
        T, R <: Real,
        IRM <: AbstractInverseRetractionMethod, RM <: AbstractRetractionMethod,
        VTM <: AbstractVectorTransportMethod, TSSA <: StoreStateAction,
    } <: Stepsize
    inverse_retraction_method::IRM
    min_stepsize::R
    max_stepsize::R
    retraction_method::RM
    s::T
    storage::TSSA
    strategy::Symbol
    vector_transport_method::VTM
    y::T
    function BarzileiBorweinStepsize(
            M::AbstractManifold;
            p::P = rand(M), X::T = zero_vector(M, p),
            min_stepsize::R = 1.0e-3,
            max_stepsize::R = 1.0e3,
            retraction_method::RM = default_retraction_method(M, typeof(p)),
            inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
            storage::Union{Nothing, StoreStateAction} = StoreStateAction(
                M; store_fields = [:Iterate, :Gradient]
            ),
            strategy::Symbol = :direct,
            vector_transport_method::VTM = default_vector_transport_method(M),
        ) where {
            IRM <: AbstractInverseRetractionMethod, RM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod, P, R <: Real, T,
        }
        if strategy ∉ [:direct, :inverse, :alternating]
            @warn string(
                "The strategy '", strategy, "' is not defined. The 'direct' strategy is used instead.",
            )
            strategy = :direct
        end
        if min_stepsize <= 0.0
            throw(
                DomainError(min_stepsize, "The lower bound for the Barzilei–Borwein step size has to be positive."),
            )
        end
        if max_stepsize <= min_stepsize
            throw(
                DomainError(
                    max_stepsize, "The upper bound for the step size lower bound.",
                ),
            )
        end
        X_ = maybe_wrap_variable(X)
        p_ = maybe_wrap_variable(p)
        return new{typeof(X_), R, IRM, RM, VTM, typeof(storage)}(
            inverse_retraction_method, min_stepsize, max_stepsize,
            retraction_method, ManifoldsBase.copy(M, p_, X_), storage, strategy, vector_transport_method, X_
        )
    end
end
function (bb::BarzileiBorweinStepsize)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int, η = (-get_gradient(mp, get_iterate(s)));
        gradient = nothing, last_stepsize = nothing, kwargs...
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    X = isnothing(gradient) ? get_gradient(mp, p) : gradient
    if !has_storage(bb.storage, PointStorageKey(:Iterate)) || !has_storage(bb.storage, VectorStorageKey(:Gradient))
        # first time call: get old grad/iterate and store.
        p_old = get_iterate(s)
        X_old = X
    else
        #fetch
        p_old = get_storage(bb.storage, PointStorageKey(:Iterate))
        X_old = get_storage(bb.storage, VectorStorageKey(:Gradient))
    end
    update_storage!(bb.storage, mp, s)

    # compute the y_k – difference of gradients, but remember to transport
    vector_transport_to!(M, bb.y, p_old, X_old, p, bb.vector_transport_method)
    copyto!(M, bb.y, p, X .- bb.y)
    # for s_k there are two possibilities
    if !isnothing(last_stepsize) # Variant 1: Someone gave us the last stepsize, so we can just use -last_stepsize * X_old and transport it to the current iterate
        vector_transport_to!(M, bb.s, p_old, -last_stepsize * X_old, p, bb.vector_transport_method)
    else # Variant 2: otherwise we use the inverse retraction method
        inverse_retract!(M, bb.s, p, p_old, bb.inverse_retraction_method)
    end
    #compute the new Barzilai-Borwein step size
    s1 = real(inner(M, p, bb.s, bb.y))
    s2 = real(inner(M, p, bb.y, bb.y))
    s2 = s2 == 0 ? 1.0 : s2
    s3 = real(inner(M, p, bb.s, bb.s))
    #indirect strategy
    return if bb.strategy == :inverse
        if s1 > 0
            stepsize = min(bb.max_stepsize, max(bb.min_stepsize, s1 / s2))
        else
            stepsize = bb.max_stepsize
        end
        #alternating strategy
    elseif bb.strategy == :alternating
        if s1 > 0
            if k % 2 == 0
                stepsize = min(bb.max_stepsize, max(bb.min_stepsize, s1 / s2))
            else
                stepsize = min(bb.max_stepsize, max(bb.min_stepsize, s3 / s1))
            end
        else
            stepsize = bb.max_stepsize
        end
    else # default: direct strategy
        if s1 > 0
            stepsize = min(bb.max_stepsize, max(bb.min_stepsize, s2 / s1))
        else
            stepsize = bb.max_stepsize
        end
    end
end
function Base.show(io::IO, a::BarzileiBorweinStepsize)
    return print(io, "BarzileiBorweinStepsize TODO")
end

"""
    ConstantStepsize <: Stepsize

A functor `(problem, state, ...) -> s` to provide a constant step size `s`.

# Fields

* `length`: constant value for the step size
* `type`:   a symbol that indicates whether the stepsize is relatively (:relative),
    with respect to the gradient norm, or absolutely (:absolute) constant.

# Constructors

    ConstantStepsize(s::Real, t::Symbol=:relative)

initialize the stepsize to a constant `s` of type `t`.

    ConstantStepsize(
        M::AbstractManifold=DefaultManifold(),
        s=min(injectivity_radius(M)/2, 1.0);
        type::Symbol=:relative
    )
"""
mutable struct ConstantStepsize{R <: Real} <: Stepsize
    length::R
    type::Symbol
end
function ConstantStepsize(
        M::AbstractManifold, length::R = min(injectivity_radius(M) / 2, 1.0); type = :relative
    ) where {R <: Real}
    return ConstantStepsize{R}(length, type)
end
function (cs::ConstantStepsize)(
        amp::AbstractManoptProblem,
        ams::AbstractManoptSolverState,
        ::Any,
        args...;
        gradient = nothing,
        kwargs...,
    )
    s = cs.length
    if cs.type == :absolute
        grad = isnothing(gradient) ? get_gradient(amp, get_iterate(ams)) : gradient
        ns = norm(get_manifold(amp), get_iterate(ams), grad)
        if ns > eps(eltype(s))
            s /= ns
        end
    end
    return s
end
get_initial_stepsize(s::ConstantStepsize) = s.length
function Base.show(io::IO, cs::ConstantStepsize)
    return print(io, "ConstantLength($(cs.length); type=:$(cs.type))")
end
function status_summary(s::ConstantStepsize; context::Symbol = :default)
    (context === :short) && return repr(s)
    r = (s.type === :absolute ? "absolute" : "relative")
    return "A $r constant step size of length $(s.length)"
end

"""
    ConstantLength(s; kwargs...)
    ConstantLength(M::AbstractManifold, s; kwargs...)

Specify a [`Stepsize`](@ref) that is constant.

# Input

* `M` (optional)
* `s=min(injectivity_radius(M)/2, 1.0)`: the length to use.

# Keyword argument

* `type::Symbol=:relative` specify the type of constant step size. Possible values are
  * `:relative` – scale the gradient tangent vector ``X`` to ``s*X``
  * `:absolute` – scale the gradient to an absolute step length ``s``, that is ``$(_tex(:frac, "s", _tex(:norm, "X")))X``

$(_note(:ManifoldDefaultsFactory, "ConstantStepsize"))
"""
function ConstantLength(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.ConstantStepsize, args...; kwargs...)
end

@doc """
    CubicBracketingLinesearchStepsize{R<:Real,I<:Integer,TRM,VTM,P,T} <: Linesearch

Do a bracketing line search to find a step size ``α`` that finds a
local minimum along the search direction ``X`` starting from ``p``,
utilizing cubic polynomial interpolation.
See [`CubicBracketingLinesearch`](@ref) for the mathematical details.

# Fields
$(_fields(:p; name = "candidate_point"))
  as temporary storage for candidates
* `candidate_direction::T`: temporary storage for the transported search direction
* `initial_stepsize::R`: the step size to start the search with
* `last_stepsize::R`
$(_fields(:retraction_method))
* `stepsize_increase::R`:  step size increase factor ``>1``
* `max_iterations::I`: maximum number of iterations
* `sufficient_curvature::R`: target reduction of the curvature ``(0,1)``
* `min_bracket_width::R`: minimal size of the bracket ``[a,b]``
* `hybrid::Bool`: use the hybrid strategy
* `max_stepsize::R`: maximal stepsize
$(_fields(:vector_transport_method))

# Constructor

    CubicBracketingLinesearchStepsize(M::AbstractManifold; kwargs...)
    CubicBracketingLinesearchStepsize(M::AbstractManifold, p; kwargs...)

## Keyword arguments

$(_kwargs(:p; name = "candidate_point")) as temporary storage for candidates
* `initial_stepsize=1.0`: the step size to start the search with
$(_kwargs(:retraction_method))
* `stepsize_increase=1.5`:  step size increase factor ``>1``
* `max_iterations=100`: maximum number of iterations
* `sufficient_curvature=0.2`: target reduction of the curvature ``(0,1)``
* `min_bracket_width=1e-4`: minimal size of the bracket ``[a,b]``
* `hybrid=true`: use the hybrid strategy
* `max_stepsize= max_stepsize(M)`: maximal stepsize
$(_kwargs(:vector_transport_method))
"""
mutable struct CubicBracketingLinesearchStepsize{
        R <: Real,
        I <: Integer,
        TRM <: AbstractRetractionMethod,
        VTM <: AbstractVectorTransportMethod,
        P,
        T,
    } <: Linesearch
    candidate_direction::T
    candidate_point::P
    initial_stepsize::R
    last_stepsize::R
    retraction_method::TRM
    stepsize_increase::R
    max_iterations::I
    sufficient_curvature::R
    min_bracket_width::R
    hybrid::Bool
    vector_transport_method::VTM
    max_stepsize::R
    function CubicBracketingLinesearchStepsize(
            M::AbstractManifold;
            candidate_point::P = allocate_result(M, rand),
            candidate_direction::T = zero_vector(M, candidate_point),
            initial_stepsize::R = 1.0,
            retraction_method::TRM = default_retraction_method(M),
            stepsize_increase::R = 1.5,
            max_iterations::I = 100,
            sufficient_curvature::R = 0.2,
            min_bracket_width::R = 1.0e-4,
            hybrid::Bool = true,
            vector_transport_method::VTM = default_vector_transport_method(M),
            max_stepsize::Real = max_stepsize(M),
        ) where {R <: Real, I <: Integer, TRM, VTM, P, T}
        p = maybe_wrap_variable(candidate_point)
        X = maybe_wrap_variable(candidate_direction)
        return new{R, I, TRM, VTM, typeof(p), typeof(X)}(X, p, initial_stepsize, initial_stepsize, retraction_method, stepsize_increase, max_iterations, sufficient_curvature, min_bracket_width, hybrid, vector_transport_method, max_stepsize)
    end
end
function CubicBracketingLinesearchStepsize(M::AbstractManifold, p; kwargs...)
    candidate_point = allocate(p)
    candidate_direction = zero_vector(M, candidate_point)
    return CubicBracketingLinesearchStepsize(
        M; candidate_point = candidate_point, candidate_direction = candidate_direction, kwargs...
    )
end

"""
    UnivariateTriple{R <: Real}

Triple of stepsize, function value and derivative value.

# Fields
* `t::R`: stepsize
* `f::R`: cost at stepsize `t`
* `df::R`: derivative of the cost at stepsize `t`
"""
struct UnivariateTriple{R <: Real}
    t::R
    f::R
    df::R
end

"""
    update_bracket(a::UnivariateTriple, b::UnivariateTriple, c::UnivariateTriple)

Updates bracket w.r.t. the bracketing strategy in [Hager:1989](@cite) (R3) - (R5).

# Input
* `a::UnivariateTriple{R}`: triple of bracket value `a`
* `b::UnivariateTriple{R}`: triple of bracket value `b`
* `c::UnivariateTriple{R}`: triple of update value
"""
function update_bracket(a::UnivariateTriple{R}, b::UnivariateTriple{R}, c::UnivariateTriple{R}) where {R}
    if (c.t > max(a.t, b.t) || c.t < min(a.t, b.t))
        throw(
            DomainError(
                c.t,
                "Bracket interval does not contain update value"
            ),
        )
    end
    if (c.f > a.f)
        #(R3)
        a, b = a, c
    elseif (c.f < a.f)
        #(R4)
        if (c.df * (a.t - c.t) ≤ 0)
            a, b = c, a
        else
            a, b = c, b
        end
    else
        #(R5)
        if (c.df * (a.t - c.t) < 0)
            a, b = c, a
        elseif (a.df * (b.t - a.t) < 0)
            a, b = a, c
        else
            a, b = c, b
        end
    end
    return a, b
end

"""
    cubic_polynomial_argmin(a::UnivariateTriple, b::UnivariateTriple; warn::Bool = true)

Returns the local minimizer of the cubic polynomial ``p`` with ``p(a.t)=a.f``, ``p(b.t)=b.f``,
``p'(a.t)=a.df``, ``p'(b.t)=b.df``.

# Input
* `a::UnivariateTriple{R}`: triple of bracket value `a`
* `b::UnivariateTriple{R}`: triple of bracket value `b`

# Keyword arguments
* `warn::Bool`: Boolean value if warnings should be displayed
"""
function cubic_polynomial_argmin(a::UnivariateTriple{R}, b::UnivariateTriple{R}; warn::Bool = true) where {R}
    (a.f > b.f && warn) && @warn "value bracket condition not met."
    (a.df * (b.t - a.t) > 0 && warn) && @warn "derivative bracket condition not met."

    Δ = b.t - a.t
    v = a.df + b.df - 3 * (b.f - a.f) / Δ
    discriminant = v^2 - a.df * b.df
    #negative discriminants only occur with roundoff errors at 0
    discriminant = max(discriminant, 0.0)
    w = sign(Δ) * sqrt(discriminant)
    denom_a = a.df + v - w
    denom_b = b.df + v + w
    if (denom_a > denom_b)
        return a.t + Δ * a.df / denom_a
    else
        return b.t - Δ * b.df / denom_b
    end
end

"""
    secant(a::UnivariateTriple, b::UnivariateTriple)

Returns the extremum of the quadratic polynomial ``p`` with
``p'(a.t)=a.df``, ``p'(b.t)=b.df``.

The result is algebraically equivalent to `(a.t * b.df - b.t * a.df) / (b.df - a.df)`
but the used formula is more numerically stable.

# Input
* `a::UnivariateTriple{R}`: triple of bracket value `a`
* `b::UnivariateTriple{R}`: triple of bracket value `b`
"""
function secant(a::UnivariateTriple{R}, b::UnivariateTriple{R}) where {R}
    return (a.t + b.t) / 2 + (b.t - a.t) * (a.df + b.df) / (2 * (a.df - b.df))
end

"""
    cubic_stepsize_update_step(a::Real, b::Real, c::Real, τ::Real)

Step function to determine the stepsize update `c` described in
[Hager:1989](@cite).

# Input
* `a::Real`: first value of the bracket
* `b::Real`: second value of the bracket
* `c::Real`: update value
* `τ::Real`: minimal step tolerance
"""
function cubic_stepsize_update_step(a::Real, b::Real, c::Real, τ::Real)
    y = min(a, b)
    z = max(a, b)
    if (y + τ ≤ c && c ≤ z - τ)
        return c
    end
    if (c > (a + b) / 2)
        return max(z - τ, (a + b) / 2)
    else
        return min(y + τ, (a + b) / 2)
    end
end

"""
    get_univariate_triple!(mp::AbstractManoptProblem, cbls::CubicBracketingLinesearchStepsize, p, η, t::Real)

Get the `UnivariateTriple` of the problem `mp` related to the step with
stepsize ``t`` from ``p`` in direction ``η``.

# Input
* `mp::AbstractManoptProblem`
* `cbls::CubicBracketingLinesearchStepsize`: containing `retraction_method`, `vector_transport` and the temporary `candidate_point` and `candidate_direction`
* `p`: point in the manifold of `mp`
* `η`: search direction at `p`
* `t::Real`: step size
"""
function get_univariate_triple!(mp::AbstractManoptProblem, cbls::CubicBracketingLinesearchStepsize, p, η, t::Real)
    M = get_manifold(mp)
    cbls.last_stepsize = t
    ManifoldsBase.retract_fused!(M, cbls.candidate_point, p, η, t, cbls.retraction_method)
    vector_transport_to!(M, cbls.candidate_direction, p, η, cbls.candidate_point, cbls.vector_transport_method)
    f, df = get_cost_and_differential(mp, cbls.candidate_point, cbls.candidate_direction)
    return UnivariateTriple(t, f, df)
end

function (cbls::CubicBracketingLinesearchStepsize)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int, η = (-get_gradient(mp, get_iterate(s))); kwargs...,
    )
    M = get_manifold(mp)
    p = get_iterate(s)

    init = UnivariateTriple(0.0, get_cost(M, get_objective(mp), p), get_differential(mp, p, η; gradient = s.X, evaluated = true))

    check_curvature(c::UnivariateTriple) = abs(c.df) < cbls.sufficient_curvature * abs(init.df)

    n_iter = 0
    max_step = cbls.max_stepsize
    if :stop_when_stepsize_exceeds in keys(kwargs)
        max_step = min(max_step, kwargs[:stop_when_stepsize_exceeds])
    end
    t = min(cbls.last_stepsize, max_step)
    c_old = init
    c = get_univariate_triple!(mp, cbls, p, η, t)
    a, b = nothing, nothing
    # Construct initial bracket
    while ((n_iter += 1) <= cbls.max_iterations)
        (c.f < init.f && check_curvature(c)) && return t
        if (c.f ≥ c_old.f && c_old.df * (c.t - c_old.t) < 0)
            (a, b) = c_old, c
            break
        end
        if (c.f ≤ c_old.f && c.df * (c_old.t - c.t) < 0)
            (a, b) = c, c_old
            break
        end
        (t == max_step) && return t
        t *= cbls.stepsize_increase
        t = min(t, max_step)
        c_old = c
        c = get_univariate_triple!(mp, cbls, p, η, t)
    end

    while ((n_iter += 1) <= cbls.max_iterations)
        # Step 1
        abs(a.t - b.t) < cbls.min_bracket_width && break
        l = 2 * abs(a.t - b.t)
        γ = cubic_polynomial_argmin(a, b)
        t = cubic_stepsize_update_step(a.t, b.t, γ, cbls.min_bracket_width)
        c = get_univariate_triple!(mp, cbls, p, η, t)
        check_curvature(c) && break
        a_old = a
        a, b = update_bracket(a, b, c)
        if (cbls.hybrid)
            while ((n_iter += 1) <= cbls.max_iterations)
                # Step 2
                abs(a.t - b.t) < cbls.min_bracket_width && return t
                l = l / 2
                abs(a_old.t - t) > l && break
                # Step 3
                (c.df - a_old.df) / (c.t - a_old.t) ≤ 0 && break
                # Step 4
                γ = cubic_polynomial_argmin(a_old, c; warn = false)
                (γ < min(a.t, b.t) || γ > max(a.t, b.t)) && break
                t = cubic_stepsize_update_step(a.t, b.t, γ, cbls.min_bracket_width)
                c = get_univariate_triple!(mp, cbls, p, η, t)
                check_curvature(c) && return t
                a_old = a
                a, b = update_bracket(a, b, c)
            end
            # Step 5
            t = (a.t + b.t) / 2
            c = get_univariate_triple!(mp, cbls, p, η, t)
            check_curvature(c) && break
            a, b = update_bracket(a, b, c)
        end
    end
    return t
end
function Base.show(io::IO, cbls::CubicBracketingLinesearchStepsize)
    return print(
        io,
        "CubicBracketingLinesearch(; initial_stepsize = $(cbls.initial_stepsize),  stepsize_increase = $(cbls.stepsize_increase),  sufficient_curvature = $(cbls.sufficient_curvature),  min_bracket_width = $(cbls.min_bracket_width),  hybrid = $(cbls.hybrid),  retraction_method = $(cbls.retraction_method),  vector_transport_method = $(cbls.vector_transport_method),  max_stepsize = $(cbls.max_stepsize))",
    )
end
function status_summary(cbls::CubicBracketingLinesearchStepsize)
    return "$(cbls)\nand a computed last stepsize of $(cbls.last_stepsize)"
end

@doc """
    CubicBracketingLinesearch(; kwargs...)
    CubicBracketingLinesearch(M::AbstractManifold; kwargs...)

A functor representing the curvature minimizing cubic bracketing scheme introduced
in [Hager:1989](@cite). Firstly, a bracket ``[a,b]`` is generated by multiplying
``t_0`` chosen as `last_stepsize` (or in case of the first iteration `initial_stepsize`) repeatedly with
the `stepsize_increase > 1` until the bracket conditions

```math
    ϕ'(a)(b-a) < 0  \\quad \\text{and} \\quad ϕ(a) ≤ ϕ(b).
```

are satisfied by either ``[a,b] = [t_{k-1},t_k]``, ``[a,b] = [t_k,t_{k-1}]``, ``[a,b] = [0,t_k]``, or ``[a,b] = [t_k,0]``.
Here, ``ϕ(t)`` denotes the cost function when performing
a step with size ``t`` into direction ``η``.
Over the iteration, the bracket ``[a,b]`` is repeatedly
updated using a cubic polynomial using values of ``ϕ, ϕ'`` at ``a,b``.
The update value ``c`` is the local minimum of the polynomial, and the bracket condition
ensures that it lies in between ``a`` and ``b``. We note that the update strategy taken from
[Hager:1989](@cite) ensures that the updated bracket satisfies the bracket condition.

If the parameter `hybrid` is set to `true`, the hybrid approach from [Hager:1989](@cite)
is activated, which prevents slow convergence in edge cases.

The algorithm terminates if at any point the found candidate stepsize suffices the curvature condition
induced by `sufficient_curvature`, or the bracket ``[a,b]`` is smaller than `min_bracket_width`.

# Keyword arguments

$(_kwargs(:p)) to store an interim result
* `initial_stepsize=1.0`: the step size to start the search with
$(_kwargs(:retraction_method))
* `stepsize_increase=1.5`:  step size increase factor ``>1``
* `max_iterations=100`: maximum number of iterations
* `sufficient_curvature=0.2`: target reduction of the curvature ``(0,1)``
* `min_bracket_width=1e-4`: minimal size of the bracket ``[a,b]``
* `hybrid=true`: use the hybrid strategy
$(_kwargs(:vector_transport_method))

$(_note(:ManifoldDefaultsFactory, "CubicBracketingLinesearch"))
"""
function CubicBracketingLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(CubicBracketingLinesearchStepsize, args...; requires_point = true, kwargs...)
end


@doc """
    DecreasingStepsize(M::AbstractManifold; kwargs...)

A functor `(problem, state, ...) -> s` to provide a decreasing step size `s`.

# Fields

* `exponent`:   a value ``e``, the exponent the shifted iteration number is raised to
  in the denominator
* `factor`:     a value ``f`` to multiply the initial step size with every iteration
* `length`:     the initial step size ``l``.
* `subtrahend`: a value ``a`` that is subtracted every iteration
* `shift`:      shift the denominator iterator ``k`` by ``s``.
* `type`:       a symbol that indicates whether the stepsize is relatively (:relative),
    with respect to the gradient norm, or absolutely (:absolute) constant.

In total the complete formulae reads for the ``k``th iterate as

```math
s_k = $(_tex(:frac, "(l -  k a)f^k", "(k + s)^e"))
```

and hence the default simplifies to just ``s_k = $(_tex(:frac, "l", "k"))``

# Constructor

    DecreasingStepsize(M::AbstractManifold;
        length=isinf(manifold_dimension(M)) ? 1.0 : manifold_dimension(M)/2,
        factor=1.0,
        subtrahend=0.0,
        exponent=1.0,
        shift=0.0,
        type=:relative,
    )

initializes all fields, where none of them is mandatory. The `length` defaults to half the
manifold dimension, or to ``1`` if that dimension is infinite.
"""
mutable struct DecreasingStepsize{R <: Real} <: Stepsize
    length::R
    factor::R
    subtrahend::R
    exponent::R
    shift::R
    type::Symbol
    function DecreasingStepsize(;
            length::R, factor::R, subtrahend::R, exponent::R, shift::R, type::Symbol
        ) where {R}
        return new{R}(length, factor, subtrahend, exponent, shift, type)
    end
end
function DecreasingStepsize(
        M::AbstractManifold;
        length::Real = isinf(manifold_dimension(M)) ? 1.0 : manifold_dimension(M) / 2,
        factor::Real = 1.0, subtrahend::Real = 0.0, exponent::Real = 1.0, shift::Real = 0.0,
        type::Symbol = :relative,
    )
    R = promote_type(typeof(length), typeof(factor), typeof(subtrahend), typeof(exponent), typeof(shift))
    l = convert(R, length); f = convert(R, factor); s = convert(R, subtrahend); e = convert(R, exponent); t = convert(R, shift)
    return DecreasingStepsize(;
        length = l, factor = f, subtrahend = s, exponent = e, shift = t, type = type
    )
end
function (s::DecreasingStepsize)(
        amp::P, ams::O, k::Int, args...; kwargs...
    ) where {P <: AbstractManoptProblem, O <: AbstractManoptSolverState}
    ds = (s.length - k * s.subtrahend) * (s.factor^k) / ((k + s.shift)^(s.exponent))
    if s.type == :absolute
        ns = norm(get_manifold(amp), get_iterate(ams), get_gradient(ams))
        if ns > eps(eltype(ds))
            ds /= ns
        end
    end
    return ds
end
get_initial_stepsize(s::DecreasingStepsize) = s.length
function Base.show(io::IO, s::DecreasingStepsize)
    print(io, "DecreasingStepsize(; length = ", s.length, ", exponent = ", s.exponent, ", factor = ", s.factor)
    return print(io, ", subtrahend = ", s.subtrahend, ", shift = ", s.shift, ", type = :$(s.type))")
end
function status_summary(s::DecreasingStepsize; context::Symbol = :default)
    (context === :short) && return repr(s)
    (context === :inline) && return "A decreasing stepsize ($(s.length) - k*$(s.subtrahend)) * $(s.factor)^k) / (k + $(s.shift))^$(s.exponent)"
    return """
    A decreasing step size
    For the `k`th iterate compute

    ((l -  k*a)f^k) / (k + s)^e

    ## Parameters
    * length l: $(_MANOPT_INDENT)$(s.length)
    * subtrahend a: $(_MANOPT_INDENT)$(s.subtrahend)
    * factor f: $(_MANOPT_INDENT)$(s.factor)
    * shift s: $(_MANOPT_INDENT)$(s.shift)
    * exponent e: $(_MANOPT_INDENT)$(s.exponent)
    * type : $(_MANOPT_INDENT):$(s.type)
    """
end
"""
    DecreasingLength(; kwargs...)
    DecreasingLength(M::AbstractManifold; kwargs...)

Specify a [`Stepsize`](@ref) that is decreasing as ``s_k = $(_tex(:frac, "(l - ak)f^k", "(k+s)^e"))``
with the following

# Keyword arguments

* `exponent=1.0`:   the exponent ``e`` in the denominator
* `factor=1.0`:     the factor ``f`` in the nominator
* `length=isinf(manifold_dimension(M)) ? 1.0 : manifold_dimension(M)/2`: the initial step size ``l``.
* `subtrahend=0.0`: a value ``a`` that is subtracted every iteration
* `shift=0.0`:      shift the denominator iterator ``k`` by ``s``.
* `type::Symbol=:relative` specify the type of step size. Possible values are
  * `:relative` – scale the gradient tangent vector ``X`` to ``s_k*X``
  * `:absolute` – scale the gradient to an absolute step length ``s_k``, that is ``$(_tex(:frac, "s_k", _tex(:norm, "X")))X``

$(_note(:ManifoldDefaultsFactory, "DecreasingStepsize"))
"""
function DecreasingLength(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.DecreasingStepsize, args...; kwargs...)
end

@doc raw"""
    DistanceOverGradientsStepsize{R<:Real,P} <: Stepsize

A functor `(problem, state, k, ...) -> s` providing the Riemannian Distance over Gradients (RDoG) step size.

This step size is learning-rate-free: it adapts using the maximum distance travelled from the
start point together with the accumulated squared gradient norms.
See [`DistanceOverGradients`](@ref) for the mathematical details.

# Fields

* `initial_distance::R`: initial distance estimate ``ϵ>0``
* `max_distance::R`: tracked maximum distance ``\bar r_t``
* `gradient_sum::R`: accumulated sum ``G_t``
* `initial_point`: stored start point ``p_0``
* `use_curvature::Bool`: toggle curvature correction ``ζ_κ``
* `sectional_curvature_bound::R`: lower bound ``κ`` used in ``ζ_κ`` when `use_curvature=true`
* `last_stepsize::R`: last computed stepsize

# Constructor

    DistanceOverGradientsStepsize(M::AbstractManifold, p; kwargs...)

where `p` is the initial point, from which the distance is tracked.

## Keyword arguments

* `initial_distance=1e-3`: initial estimate ``ϵ``
* `use_curvature=false`: whether to use ``ζ_κ``
* `sectional_curvature_bound=0.0`: lower curvature bound ``κ`` (if known)

# References

[DoddSharrockNemeth:2024](@cite): Learning-Rate-Free Stochastic Optimization over
Riemannian Manifolds (RDoG).
"""
mutable struct DistanceOverGradientsStepsize{R <: Real, P} <: Stepsize
    initial_distance::R
    max_distance::R
    gradient_sum::R
    initial_point::P
    use_curvature::Bool
    sectional_curvature_bound::R
    last_stepsize::R
    function DistanceOverGradientsStepsize(;
            initial_distance::R, max_distance::R, gradient_sum::R, initial_point::P,
            use_curvature::Bool, sectional_curvature_bound::R, last_stepsize::R
        ) where {R <: Real, P}
        return new{R, P}(
            initial_distance, max_distance, gradient_sum, initial_point, use_curvature,
            sectional_curvature_bound, last_stepsize,
        )
    end
end
function DistanceOverGradientsStepsize(
        M::AbstractManifold, p;
        initial_distance::R1 = 1.0e-3, use_curvature::Bool = false, sectional_curvature_bound::R2 = 0.0,
    ) where {R1 <: Real, R2 <: Real}
    R = promote_type(R1, R2)
    id = convert(R, initial_distance)
    κ = convert(R, sectional_curvature_bound)
    p_ = maybe_wrap_variable(p)
    return DistanceOverGradientsStepsize(;
        initial_distance = id, max_distance = id, gradient_sum = zero(R), initial_point = copy(M, p_),
        use_curvature = use_curvature, sectional_curvature_bound = κ, last_stepsize = zero(R)
    )
end

@doc raw"""
    geometric_curvature_function(κ::Real, d::Real)

Compute the geometric curvature function ``ζ_κ(d)`` used by the RDoG stepsize:

```math
ζ_κ(d) =
\begin{cases}
1, & \text{if } κ \ge 0,\\[4pt]
\dfrac{\sqrt{|κ|}\,d}{\tanh(\sqrt{|κ|}\,d)}, & \text{if } κ < 0.
\end{cases}
```

For small arguments, a Taylor approximation is used for numerical stability.
"""
function geometric_curvature_function(κ::Real, d::Real)
    if κ < 0 && d > 0
        sqrt_abs_κ = sqrt(abs(κ))
        arg = sqrt_abs_κ * d
        return arg / tanh(arg)
    else
        return 1.0
    end
end

function (rdog::DistanceOverGradientsStepsize{R, P})(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, i, args...;
        gradient = nothing, kwargs...,
    ) where {R, P}
    M = get_manifold(mp)
    p = get_iterate(s)
    grad = isnothing(gradient) ? get_gradient(mp, p) : gradient
    # Compute gradient norm
    grad_norm_sq = clamp(norm(M, p, grad)^2, eps(R), typemax(R))
    if i == 0
        # Initialize on first call
        rdog.gradient_sum = grad_norm_sq
        rdog.initial_point = copy(M, p)
        rdog.max_distance = rdog.initial_distance

        # Initial stepsize
        if rdog.use_curvature
            ζ = geometric_curvature_function(
                rdog.sectional_curvature_bound, rdog.max_distance
            )
            stepsize = rdog.initial_distance / (sqrt(ζ) * sqrt(max(grad_norm_sq, eps(R))))
        else
            stepsize = rdog.initial_distance / sqrt(max(grad_norm_sq, eps(R)))
        end
    else
        # Update gradient sum
        rdog.gradient_sum += grad_norm_sq

        # Update max distance
        current_distance = distance(M, rdog.initial_point, p)
        rdog.max_distance = max(rdog.max_distance, current_distance)

        # Compute stepsize
        if rdog.use_curvature
            ζ = geometric_curvature_function(
                rdog.sectional_curvature_bound, rdog.max_distance
            )
            stepsize = rdog.max_distance / (sqrt(ζ) * sqrt(rdog.gradient_sum))
        else
            stepsize = rdog.max_distance / sqrt(rdog.gradient_sum)
        end
    end
    rdog.last_stepsize = stepsize
    return stepsize
end

get_initial_stepsize(rdog::DistanceOverGradientsStepsize) = rdog.last_stepsize
get_last_stepsize(rdog::DistanceOverGradientsStepsize) = rdog.last_stepsize

function Base.show(io::IO, rdog::DistanceOverGradientsStepsize)
    print(io, "DistanceOverGradientsStepsize(; initial_distance = ", rdog.initial_distance)
    print(io, ", use_curvature = ", rdog.use_curvature, ", sectional_curvature_bound = ", rdog.sectional_curvature_bound)
    print(io, ", max_distance = ", rdog.max_distance, ", gradient_sum = ", rdog.gradient_sum)
    print(io, ", initial_point = ", rdog.initial_point, ", last_stepsize = ", rdog.last_stepsize)
    return print(io, ")")
end
function status_summary(rdog::DistanceOverGradientsStepsize; context::Symbol = :default)
    (context === :short) && return repr(rdog)
    s = rdog.use_curvature ? "including a curvature correction" : ""
    (context === :inline) && return "A distance over gradients step size $s (last stepsize: $(rdog.last_stepsize))"
    s2 = !rdog.use_curvature ? "" : "* sectional curvature bound:$(_MANOPT_INDENT)$(rdog.sectional_curvature_bound)"
    return """
    A distance over gradients step size
    (last stepsize: $(rdog.last_stepsize))

    ## Parameters
    * use curvature correction: $(_MANOPT_INDENT)$(rdog.use_curvature)$(s2)
    * sum of gradients:         $(_MANOPT_INDENT)$(rdog.gradient_sum)
    * maximal distance r_t:     $(_MANOPT_INDENT)$(rdog.max_distance)
    """
end
doc_DoG_main = raw"""
    DistanceOverGradients(; kwargs...)
    DistanceOverGradients(M::AbstractManifold; kwargs...)

Create a factory for the [`DistanceOverGradientsStepsize`](@ref), the
Riemannian Distance over Gradients (RDoG) learning-rate-free stepsize from
[DoddSharrockNemeth:2024](@cite). It adapts without manual tuning, by combining the maximum
distance from the start point with the accumulated gradient norms, optionally corrected by
the geometric curvature term ``ζ_κ``.

Definitions used by the implementation:

* ``\bar r_t := \max(\,ϵ,\, \max_{0\le s\le t} d(p_0, p_s)\,)`` tracks the maximum geodesic
  distance from the initial point ``p_0`` using the current iterate ``p_t``.
* ``G_t := \displaystyle\sum_{s=0}^t \lVert g_s \rVert^2``, where ``g_s = \operatorname{grad} f(p_s)``.

At iteration ``t`` the stepsize used here is

```math
η_t =
\begin{cases}
\frac{\bar r_t}{\sqrt{G_t}}, & \text{if we do not use curvature,}\\
\frac{\bar r_t}{\sqrt{\,ζ_κ(\bar r_t)\,}\,\sqrt{G_t}}, & \text{if we use curvature.}
\end{cases}
```

with the geometric curvature function ``ζ_κ(d)`` defined in
[`geometric_curvature_function`](@ref). The initialization in this
implementation follows the paper: on the first call (``t=0``), we set
``G_0=\lVert g_0\rVert^2``, ``\bar r_0 = ϵ`` and take

```math
η_0 =
\begin{cases}
\frac{ϵ}{\lVert g_0\rVert}, & \text{if we do not use curvature,}\\
\frac{ϵ}{\sqrt{\,ζ_κ(ϵ)\,}\,\lVert g_0\rVert}, & \text{if we use curvature.}
\end{cases}
```

On subsequent calls, the state is updated as implemented:
``G_t ← G_{t-1} + \lVert g_t\rVert^2`` and ``\bar r_t ← \max(\bar r_{t-1}, d(p_0,p_t))``.

## Keyword arguments

* `initial_distance=1e-3`: initial distance estimate ``ϵ``
* `use_curvature=false`: whether to include ``ζ_κ``
* `sectional_curvature_bound=0.0`: curvature lower bound ``κ`` (if known)
"""
@doc """
$(doc_DoG_main)

$(_note(:ManifoldDefaultsFactory, "DistanceOverGradientsStepsize"))
"""
function DistanceOverGradients(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.DistanceOverGradientsStepsize, args...; requires_point = true, kwargs...)
end

@doc """
    NonmonotoneLinesearchStepsize{P,T,R<:Real,I<:Integer,TRM,VTM,TSSA,MSGS,IG} <: Linesearch

A functor representing a nonmonotone line search using the Barzilai-Borwein step size [IannazzoPorcelli:2017](@cite).

# Fields

$(_fields(:initial_guess))
* `memory_size`:           number of iterations after which the cost value needs to be lower than the current one
* `bb_min_stepsize`:          lower bound for the Barzilai-Borwein step size, greater than zero
* `bb_max_stepsize`:          upper bound for the Barzilai-Borwein step size, greater than `bb_min_stepsize`
* `last_stepsize`:     the last computed stepsize
$(_fields(:retraction_method))
* `strategy`:                 defines if the new step size is computed using the `:direct`, `:inverse` or `:alternating` strategy
* `storage`:                  (for `:Iterate` and `:Gradient`) a [`StoreStateAction`](@ref)
* `stepsize_reduction`:       step size reduction factor contained in the interval (0,1)
* `sufficient_decrease`:     sufficient decrease parameter contained in the interval (0,1)
$(_fields(:vector_transport_method))
* `candidate_point`:          to store an interim result
* `stop_when_stepsize_less`:    smallest stepsize when to stop (the last one before is taken)
* `stop_when_stepsize_exceeds`: largest stepsize when to stop.
* `stop_increasing_at_step`:    last step to increase the stepsize (phase 1),
* `stop_decreasing_at_step`:    last step size to decrease the stepsize (phase 2),

# Constructor

    NonmonotoneLinesearchStepsize(M::AbstractManifold; kwargs...)
    NonmonotoneLinesearchStepsize(M::AbstractManifold, p; kwargs...)

## Keyword arguments

* `p=allocate_result(M, rand)`: to store an interim result
* `initial_guess = (problem, state, k, last_stepsize, η) -> k == 0 ? 1.0 : last_stepsize`
   function to provide an initial guess for the stepsize
* `memory_size=10`
* `bb_min_stepsize=1e-3`
* `bb_max_stepsize=1e3`
$(_kwargs(:retraction_method))
* `strategy=:direct`
* `storage=`[`StoreStateAction`](@ref)`(M; store_fields=[:Iterate, :Gradient])`
* `stepsize_reduction=0.5`
* `sufficient_decrease=1e-4`
* `stop_when_stepsize_less=0.0`
* `stop_when_stepsize_exceeds=`[`max_stepsize`](@ref)`(M)`
* `stop_increasing_at_step=100`
* `stop_decreasing_at_step=1000`
$(_kwargs(:vector_transport_method))
"""
mutable struct NonmonotoneLinesearchStepsize{
        P, T <: AbstractVector, R <: Real, I <: Integer, TRM <: AbstractRetractionMethod,
        MSGS <: NamedTuple, IG, BB <: BarzileiBorweinStepsize,
    } <: Linesearch
    bb_stepsize::BB
    candidate_point::P
    initial_guess::IG
    last_stepsize::R
    messages::MSGS
    old_costs::T
    retraction_method::TRM
    stepsize_reduction::R
    stop_decreasing_at_step::I
    stop_increasing_at_step::I
    stop_when_stepsize_exceeds::R
    stop_when_stepsize_less::R
    sufficient_decrease::R
    # This constructor is semi-legacy, since it passes down a lot of parameters to BB now
    function NonmonotoneLinesearchStepsize(
            M::AbstractManifold;
            bb_min_stepsize::R = 1.0e-3,
            bb_max_stepsize::R = 1.0e3,
            p::P = allocate_result(M, rand),
            initial_guess::IG = (problem, state, k, last_stepsize, η) -> k == 0 ? 1.0 : last_stepsize,
            inverse_retraction_method = default_inverse_retraction_method(M, typeof(p)),
            memory_size::I = 10,
            retraction_method::TRM = default_retraction_method(M),
            stepsize_reduction::R = 0.5,
            stop_when_stepsize_less::R = 0.0,
            stop_when_stepsize_exceeds::R = real(max_stepsize(M)),
            stop_increasing_at_step::I = 100,
            stop_decreasing_at_step::I = 1000,
            storage::Union{Nothing, StoreStateAction} = StoreStateAction(
                M; store_fields = [:Iterate, :Gradient]
            ),
            strategy::Symbol = :direct,
            sufficient_decrease::R = 1.0e-4,
            vector_transport_method = default_vector_transport_method(M),
        ) where {TRM, P, R <: Real, I <: Integer, IG}
        stop_when_stepsize_exceeds = R(stop_when_stepsize_exceeds)
        bb = BarzileiBorweinStepsize(
            M; p = p, min_stepsize = bb_min_stepsize, max_stepsize = bb_max_stepsize,
            inverse_retraction_method = inverse_retraction_method, retraction_method = retraction_method,
            vector_transport_method = vector_transport_method,
            storage = storage, strategy = strategy
        )
        return NonmonotoneLinesearchStepsize(
            M, bb;
            p = p,
            initial_guess = initial_guess, memory_size = memory_size,
            retraction_method = retraction_method,
            stepsize_reduction = stepsize_reduction,
            stop_when_stepsize_less = stop_when_stepsize_less,
            stop_when_stepsize_exceeds = stop_when_stepsize_exceeds,
            stop_increasing_at_step = stop_increasing_at_step,
            stop_decreasing_at_step = stop_decreasing_at_step,
            sufficient_decrease = sufficient_decrease,
        )
    end
    function NonmonotoneLinesearchStepsize(
            M::AbstractManifold, stepsize::BBS;
            p::P = rand(M),
            initial_guess::IG = (problem, state, k, last_stepsize, η) -> k == 0 ? 1.0 : last_stepsize,
            memory_size::I = 10,
            retraction_method::TRM = default_retraction_method(M),
            stepsize_reduction::R = 0.5,
            stop_when_stepsize_less::R = 0.0, stop_when_stepsize_exceeds::R = real(max_stepsize(M)),
            stop_increasing_at_step::I = 100, stop_decreasing_at_step::I = 1000,
            sufficient_decrease::R = 1.0e-4,
        ) where {BBS <: Stepsize, P, IG, TRM, R, I}
        if memory_size <= 0
            throw(DomainError(memory_size, "The memory_size has to be greater than zero."))
        end
        old_costs = zeros(memory_size)
        msgs = (;
            non_descent_direction = StepsizeMessage{R, R}(),
            stop_decreasing = StepsizeMessage{Int, R}(),
            stop_increasing = StepsizeMessage{Int, R}(),
            stepsize_less = StepsizeMessage{R, R}(),
            stepsize_exceeds = StepsizeMessage{R, R}(),
        )
        p_ = maybe_wrap_variable(p)
        return new{typeof(p_), typeof(old_costs), R, I, TRM, typeof(msgs), IG, BBS}(
            stepsize, p_, initial_guess, 1.0, msgs, old_costs,
            retraction_method,
            stepsize_reduction,
            stop_decreasing_at_step, stop_increasing_at_step, stop_when_stepsize_exceeds, stop_when_stepsize_less,
            sufficient_decrease,
        )
    end
end
function NonmonotoneLinesearchStepsize(M::AbstractManifold, p; kwargs...)
    return NonmonotoneLinesearchStepsize(M; p = p, kwargs...)
end
function (a::NonmonotoneLinesearchStepsize)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int, η = (-get_gradient(mp, get_iterate(s)));
        gradient = nothing, kwargs...,
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    X = isnothing(gradient) ? get_gradient(mp, p) : gradient
    f(M, p) = get_cost(M, get_objective(mp), p)
    reset_messages!(a.messages)
    initial_stepsize = a.initial_guess(M, p, k, a.last_stepsize, η)

    αBB = a.bb_stepsize(mp, s, k, η; gradient = X, last_stepsize = initial_stepsize, kwargs...)

    memory_size = length(a.old_costs)
    if k <= memory_size
        a.old_costs[k] = f(M, p)
    else
        a.old_costs[1:(memory_size - 1)] = a.old_costs[2:memory_size]
        a.old_costs[memory_size] = f(M, p)
    end

    #compute the new step size with the help of the Barzilai-Borwein step size
    l = norm(M, p, η)
    local swse # COV_EXCL_LINE
    if :stop_when_stepsize_exceeds in keys(kwargs)
        swse = kwargs[:stop_when_stepsize_exceeds]
    else
        swse = (a.stop_when_stepsize_exceeds / l)
    end
    a.last_stepsize = linesearch_backtrack!(
        M, a.candidate_point, f, p, αBB, a.sufficient_decrease, a.stepsize_reduction, η;
        lf0 = maximum(view(a.old_costs, 1:min(k, memory_size))),
        gradient = X,
        retraction_method = a.retraction_method,
        stop_when_stepsize_less = (a.stop_when_stepsize_less / l),
        stop_when_stepsize_exceeds = swse,
        stop_increasing_at_step = a.stop_increasing_at_step,
        stop_decreasing_at_step = a.stop_decreasing_at_step,
        report_messages_in = a.messages,
    )
    return a.last_stepsize
end
function Base.show(io::IO, a::NonmonotoneLinesearchStepsize)
    return print(
        io,
        "NonmonotoneLinesearch(; last_stepsize = $(a.last_stepsize), memory_size = $(length(a.old_costs)), stepsize_reduction = $(a.stepsize_reduction), sufficient_decrease = $(a.sufficient_decrease), retraction_method = $(a.retraction_method))",
    )
end
function get_message(a::NonmonotoneLinesearchStepsize)
    s = [get_message(kv[1], kv[2]) for kv in pairs(a.messages)]
    return join([m for m in s if length(m) > 0], "\n")
end

@doc """
    NonmonotoneLinesearch(; kwargs...)
    NonmonotoneLinesearch(M::AbstractManifold; kwargs...)

A functor representing a nonmonotone line search using the Barzilai-Borwein step size [IannazzoPorcelli:2017](@cite).

Base on the step size from the [`BazileinBorweinStepsize`](@ref) `α_k^{$(_tex(:text, "BB"))}`
Then find the smallest ``h = 0, 1, 2, …`` such that

```math
f($(_tex(:retr))_{p_k}(- σ^h α_k^{$(_tex(:text, "BB"))} $(_tex(:grad))f(p_k)))  ≤
$(_tex(:max))_{1 ≤ j ≤ $(_tex(:max))(k+1,m)} f(p_{k+1-j}) - γ σ^h α_k^{$(_tex(:text, "BB"))} ⟨$(_tex(:grad))f(p_k), $(_tex(:grad))f(p_k)⟩_{p_k},
```

where ``σ ∈ (0,1)`` is a step length reduction factor, ``m`` is the number of iterations
after which the function value has to be lower than the current one
and ``γ ∈ (0,1)`` is the sufficient decrease parameter. Finally the step size is computed as

```math
α_k = σ^h α_k^{$(_tex(:text, "BB"))}.
```

# Keyword arguments

$(_kwargs(:p)) to store an interim result
* `initial_guess = (problem, state, k, last_stepsize, η) -> k == 0 ? 1.0 : last_stepsize`:
  a function to provide an initial guess for the step size
* `memory_size=10`: number of iterations after which the cost value needs to be lower than the current one
* `bb_min_stepsize=1e-3`: lower bound for the Barzilai-Borwein step size, greater than zero
* `bb_max_stepsize=1e3`: upper bound for the Barzilai-Borwein step size, greater than `bb_min_stepsize`
$(_kwargs(:retraction_method))
* `strategy=:direct`: defines if the new step size is computed using the `:direct`, `:inverse` or `:alternating` strategy
* `storage=`[`StoreStateAction`](@ref)`(M; store_fields=[:Iterate, :Gradient])`: increase efficiency by using a [`StoreStateAction`](@ref) for `:Iterate` and `:Gradient`.
* `stepsize_reduction=0.5`:  step size reduction factor contained in the interval ``(0,1)``
* `sufficient_decrease=1e-4`: sufficient decrease parameter contained in the interval ``(0,1)``
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
* `stop_when_stepsize_exceeds=`[`max_stepsize`](@ref)`(M)`: largest stepsize when to stop to avoid leaving the injectivity radius
* `stop_increasing_at_step=100`:  last step to increase the stepsize (phase 1),
* `stop_decreasing_at_step=1000`: last step size to decrease the stepsize (phase 2),
"""
function NonmonotoneLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(NonmonotoneLinesearchStepsize, args...; requires_point = true, kwargs...)
end

@doc """
    PolyakStepsize <: Stepsize

A functor `(problem, state, ...) -> s` to provide a step size due to Polyak, cf. Section 3.2 of [Bertsekas:2015](@cite).

# Fields

* `γ`               : a function `k -> ...` representing a sequence.
* `best_cost_value` : storing the best cost value

# Constructor

    PolyakStepsize(; γ = k -> 1/k,  initial_cost_estimate=0.0)

Construct a stepsize of Polyak type.

# See also
[`Polyak`](@ref)
"""
mutable struct PolyakStepsize{F, R} <: Stepsize
    γ::F
    best_cost_value::R
end
function PolyakStepsize(; γ = (i) -> 1 / i, initial_cost_estimate = 0.0)
    return PolyakStepsize(γ, initial_cost_estimate)
end
function (ps::PolyakStepsize)(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k::Int, args...; kwargs...
    )
    M = get_manifold(amp)
    p = get_iterate(ams)
    X = get_subgradient(amp, p)
    # Evaluate the cost
    c = get_cost(M, get_objective(amp), p)
    (c < ps.best_cost_value) && (ps.best_cost_value = c)
    α = (c - ps.best_cost_value + ps.γ(k)) / (norm(M, p, X)^2)
    return α
end
function Base.show(io::IO, ps::PolyakStepsize)
    return print(io, "Polyak(; γ = $(ps.γ))")
end
function status_summary(ps::PolyakStepsize; context::Symbol = :default)
    (context === :short) && return repr(ps)
    return "Polyak step size with γ = $(ps.γ) and current best minimum estimate $(ps.best_cost_value)"
end
"""
    Polyak(; kwargs...)
    Polyak(M::AbstractManifold; kwargs...)

Compute a step size according to a method proposed by Polyak, cf. the Dynamic step size
discussed in Section 3.2 of [Bertsekas:2015](@cite).
This has been generalized here to both the Riemannian case and to approximate the minimum cost value.

Let ``f_{$(_tex(:text, "best"))}`` be the best cost value seen until now during some iterative
optimization algorithm and let ``γ_k`` be a sequence of numbers that is square summable, but not summable.

Then the step size computed here reads

```math
s_k = $(_tex(:frac, "f(p^{(k)}) - f_{$(_tex(:text, "best"))} + γ_k", _tex(:norm, "∂f(p^{(k)})"))),
```

where ``∂f`` denotes a nonzero-subgradient of ``f`` at the current iterate ``p^{(k)}``.


# Constructor

    Polyak(; γ = k -> 1/k, initial_cost_estimate=0.0)

initialize the Polyak stepsize to a certain sequence and an initial estimate of ``f_{$(_tex(:text, "best"))}``.

$(_note(:ManifoldDefaultsFactory, "PolyakStepsize"))
"""
function Polyak(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.PolyakStepsize, args...; requires_manifold = false, kwargs...)
end

@doc """
    WolfePowellLinesearchStepsize{R<:Real,TRM,VTM,P,T,I,TMSG} <: Linesearch

Do a backtracking line search to find a step size ``α`` that fulfils the
Wolfe conditions along a search direction ``X`` starting from ``p``.
See [`WolfePowellLinesearch`](@ref) for the math details.

# Fields

* `sufficient_decrease::R`, `sufficient_curvature::R`: two constants in the line search
$(_fields(:X; name = "candidate_direction"))
$(_fields(:p; name = "candidate_point"))
  as temporary storage for candidates
* `last_stepsize::R`: the last computed stepsize
* `max_stepsize::R`: the largest stepsize allowed
$(_fields(:retraction_method))
* `stop_when_stepsize_less::R`: a safeguard to stop when the stepsize gets too small
$(_fields(:vector_transport_method))
* `stop_increasing_at_step::I`: last step to increase the stepsize
* `stop_decreasing_at_step::I`: last step to decrease the stepsize
* `messages::TMSG`: a named tuple of [`StepsizeMessage`](@ref)s about the stepsize search

# Constructor

    WolfePowellLinesearchStepsize(M::AbstractManifold; kwargs...)
    WolfePowellLinesearchStepsize(M::AbstractManifold, p; kwargs...)

## Keyword arguments

* `sufficient_decrease=1e-4`
* `sufficient_curvature=0.999`
$(_kwargs(:p)) to store an interim result
$(_kwargs(:X)) as type of memory allocated for the candidate direction
* `max_stepsize=`[`max_stepsize`](@ref)`(M)`: largest stepsize allowed here.
$(_kwargs(:retraction_method))
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
* `stop_increasing_at_step=100`: for the initial increase test (s_plus), stop after these many steps
* `stop_decreasing_at_step=1000`: for the initial decrease test (s_minus), stop after these many steps
$(_kwargs(:vector_transport_method))
"""
mutable struct WolfePowellLinesearchStepsize{
        R <: Real, TRM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod, P, T, I, TMSG <: NamedTuple,
    } <: Linesearch
    sufficient_decrease::R
    sufficient_curvature::R
    candidate_direction::T
    candidate_point::P
    last_stepsize::R
    max_stepsize::R
    retraction_method::TRM
    stop_when_stepsize_less::R
    vector_transport_method::VTM
    stop_increasing_at_step::I
    stop_decreasing_at_step::I
    messages::TMSG
    function WolfePowellLinesearchStepsize(;
            sufficient_decrease::R, sufficient_curvature::R, candidate_direction::T, candidate_point::P,
            last_stepsize::R, max_stepsize::R, retraction_method::TRM, stop_when_stepsize_less::R,
            vector_transport_method::VTM, stop_increasing_at_step::I, stop_decreasing_at_step::I,
            messages::TMSG
        ) where {R <: Real, TRM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod, P, T, I <: Integer, TMSG}
        p_ = maybe_wrap_variable(candidate_point)
        X_ = maybe_wrap_variable(candidate_direction)
        return new{R, TRM, VTM, typeof(p_), typeof(X_), I, TMSG}(
            sufficient_decrease, sufficient_curvature,
            X_, p_, last_stepsize, max_stepsize, retraction_method,
            stop_when_stepsize_less, vector_transport_method, stop_increasing_at_step, stop_decreasing_at_step, messages
        )
    end
    function WolfePowellLinesearchStepsize(
            M::AbstractManifold;
            p::P = allocate_result(M, rand),
            X::T = zero_vector(M, p),
            max_stepsize::Real = max_stepsize(M),
            retraction_method::TRM = default_retraction_method(M),
            sufficient_decrease::Real = 1.0e-4,
            sufficient_curvature::Real = 0.999,
            vector_transport_method::VTM = default_vector_transport_method(M),
            stop_when_stepsize_less::Real = 0.0,
            stop_increasing_at_step::Integer = 100,
            stop_decreasing_at_step::Integer = 1000,
        ) where {TRM, VTM, P, T}
        R = promote_type(
            typeof(max_stepsize), typeof(sufficient_curvature), typeof(sufficient_decrease),
            typeof(stop_when_stepsize_less),
        )
        I = promote_type(typeof(stop_decreasing_at_step), typeof(stop_increasing_at_step))
        msgs = (;
            non_descent_direction = StepsizeMessage{R, R}(),
            stop_decreasing = StepsizeMessage{I, R}(),
            stop_increasing = StepsizeMessage{I, R}(),
            stepsize_less = StepsizeMessage{R, R}(),
            stepsize_exceeds = StepsizeMessage{R, R}(),
        )
        return WolfePowellLinesearchStepsize(;
            sufficient_decrease = convert(R, sufficient_decrease), sufficient_curvature = convert(R, sufficient_curvature),
            candidate_direction = X, candidate_point = p, last_stepsize = convert(R, 0.0),
            max_stepsize = convert(R, max_stepsize), retraction_method = retraction_method,
            stop_when_stepsize_less = convert(R, stop_when_stepsize_less),
            vector_transport_method = vector_transport_method,
            stop_increasing_at_step = convert(I, stop_increasing_at_step), stop_decreasing_at_step = convert(I, stop_decreasing_at_step),
            messages = msgs
        )
    end
end
function WolfePowellLinesearchStepsize(M::AbstractManifold, p; kwargs...)
    candidate_point = allocate(p)
    candidate_direction = zero_vector(M, candidate_point)
    return WolfePowellLinesearchStepsize(
        M; p = candidate_point, X = candidate_direction, kwargs...
    )
end
function (a::WolfePowellLinesearchStepsize)(
        mp::AbstractManoptProblem, ams::AbstractManoptSolverState, k::Int, η = (-get_gradient(mp, get_iterate(ams)));
        kwargs...,
    )
    # For readability extract a few variables
    M = get_manifold(mp)
    p = get_iterate(ams)
    l = get_differential(mp, p, η)
    grad_norm = norm(M, p, η)
    max_step_increase = ifelse(
        isfinite(a.max_stepsize), min(1.0e9, a.max_stepsize / grad_norm), 1.0e9
    )
    if :stop_when_stepsize_exceeds in keys(kwargs)
        max_step_increase = min(max_step_increase, kwargs[:stop_when_stepsize_exceeds])
    end
    step = ifelse(isfinite(a.max_stepsize), min(1.0, a.max_stepsize / grad_norm), 1.0)
    step = min(step, max_step_increase)
    s_plus = step
    s_minus = step
    # clear messages
    reset_messages!(a.messages)

    f0 = get_cost(mp, p)
    ManifoldsBase.retract_fused!(M, a.candidate_point, p, η, step, a.retraction_method)
    fNew = get_cost(mp, a.candidate_point)
    vector_transport_to!(
        M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method
    )
    # Temp tangent vector
    Y = zero_vector(M, a.candidate_point)
    if fNew > f0 + a.sufficient_decrease * step * l
        i = 0
        while (fNew > f0 + a.sufficient_decrease * step * l) && (s_minus > 10^(-9)) # decrease
            s_minus = s_minus * 0.5
            step = s_minus
            ManifoldsBase.retract_fused!(M, a.candidate_point, p, η, step, a.retraction_method)
            fNew = get_cost(mp, a.candidate_point)
            i += 1
            if i == a.stop_decreasing_at_step
                set_message!(a.messages, :stop_decreasing, at = i, bound = a.stop_decreasing_at_step, value = s_minus)
                break
            end
        end
        s_plus = min(2.0 * s_minus, max_step_increase)
    else
        vector_transport_to!(M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method)
        if get_differential(mp, a.candidate_point, a.candidate_direction; Y = Y) < a.sufficient_curvature * l
            i = 0
            while fNew <= f0 + a.sufficient_decrease * step * l && (s_plus < max_step_increase)
                # increase
                s_plus = min(s_plus * 2.0, max_step_increase)
                step = s_plus
                ManifoldsBase.retract_fused!(M, a.candidate_point, p, η, step, a.retraction_method)
                fNew = get_cost(mp, a.candidate_point)
                i += 1
                if i == a.stop_increasing_at_step
                    set_message!(a.messages, :stop_increasing, at = i, bound = a.stop_increasing_at_step, value = s_plus)
                    break
                end
            end
            s_minus = s_plus / 2.0
        end
    end
    ManifoldsBase.retract_fused!(M, a.candidate_point, p, η, s_minus, a.retraction_method)
    vector_transport_to!(M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method)
    while get_differential(mp, a.candidate_point, a.candidate_direction; Y = Y) < a.sufficient_curvature * l
        step = (s_minus + s_plus) / 2
        ManifoldsBase.retract_fused!(M, a.candidate_point, p, η, step, a.retraction_method)
        fNew = get_cost(mp, a.candidate_point)
        if fNew <= f0 + a.sufficient_decrease * step * l
            s_minus = step
        else
            s_plus = step
        end
        if abs(s_plus - s_minus) <= a.stop_when_stepsize_less
            set_message!(a.messages, :stepsize_less, at = k, bound = a.stop_when_stepsize_less, value = step)
            break
        end
        ManifoldsBase.retract_fused!(M, a.candidate_point, p, η, s_minus, a.retraction_method)
        vector_transport_to!(M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method)
    end
    step = s_minus
    a.last_stepsize = step
    return step
end
function get_last_stepsize(step::WolfePowellLinesearchStepsize, ::Any...)
    return step.last_stepsize
end
function Base.show(io::IO, a::WolfePowellLinesearchStepsize)
    print(io, "WolfePowellLinesearchStepsize(; sufficient_decrease = ", a.sufficient_decrease)
    print(io, ", sufficient_curvature = ", a.sufficient_curvature, ", candidate_direction = ", a.candidate_direction, ", candidate_point = ", a.candidate_point)
    print(io, ", last_stepsize = ", a.last_stepsize, ", max_stepsize = ", a.max_stepsize)
    print(io, ", retraction_method = ", a.retraction_method, ", stop_when_stepsize_less = ", a.stop_when_stepsize_less)
    print(io, ", vector_transport_method = ", a.vector_transport_method)
    print(io, ", stop_increasing_at_step = ", a.stop_increasing_at_step, ", stop_decreasing_at_step = ", a.stop_decreasing_at_step)
    return print(io, ", messages = ", a.messages, ")")
end
function status_summary(a::WolfePowellLinesearchStepsize; context::Symbol = :default)
    (context === :short) && return repr(a)
    (context === :inline) && return "A Wolfe Powell step size (last stepsize: $(a.last_stepsize))"
    return """
    A Wolfe Powell line search based step size
    (last stepsize: $(a.last_stepsize))

    ## Parameters
    * maximal step size:       $(_MANOPT_INDENT)$(a.max_stepsize)
    * retraction method:       $(_MANOPT_INDENT)$(a.retraction_method)
    * vector transport method: $(_MANOPT_INDENT)$(a.vector_transport_method)
    * sufficient decrease:     $(_MANOPT_INDENT)$(a.sufficient_decrease)
    * sufficient curvature:    $(_MANOPT_INDENT)$(a.sufficient_curvature)
    """
end
function get_message(a::WolfePowellLinesearchStepsize)
    s = [get_message(kv[1], kv[2]) for kv in pairs(a.messages)]
    return join([m for m in s if length(m) > 0], "\n")
end
"""
    WolfePowellLinesearch(; kwargs...)
    WolfePowellLinesearch(M::AbstractManifold; kwargs...)

Perform a linesearch to fulfill both the Armijo-Goldstein conditions
```math
f$(_tex(:bigl))( $(_tex(:retr))_{p}(αX) $(_tex(:bigr))) ≤ f(p) + c_1 α_k ⟨$(_tex(:grad)) f(p), X⟩_{p}
```

as well as the Wolfe conditions

```math
$(_tex(:deriv)) f$(_tex(:bigl))($(_tex(:retr))_{p}(tX)$(_tex(:bigr)))
$(_tex(:Big))$(_tex(:vert))_{t=α}
≥ c_2 $(_tex(:deriv)) f$(_tex(:bigl))($(_tex(:retr))_{p}(tX)$(_tex(:bigr)))$(_tex(:Big))$(_tex(:vert))_{t=0}.
```

for some given sufficient decrease coefficient ``c_1`` and some sufficient curvature condition coefficient ``c_2``.

This is adopted from [NocedalWright:2006; Section 3.1](@cite)

# Keyword arguments

* `sufficient_decrease=1e-4`
* `sufficient_curvature=0.999`
$(_kwargs(:p)) as temporary storage for candidates
$(_kwargs(:X)) as type of memory allocated for the candidate direction
* `max_stepsize=`[`max_stepsize`](@ref)`(M)`: largest stepsize allowed here.
$(_kwargs(:retraction_method))
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
* `stop_increasing_at_step=100`: for the initial increase test (s_plus), stop after these many steps
* `stop_decreasing_at_step=1000`: for the initial decrease test (s_minus), stop after these many steps
$(_kwargs(:vector_transport_method))
"""
function WolfePowellLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(WolfePowellLinesearchStepsize, args...; requires_point = true, kwargs...)
end

@doc """
    WolfePowellBinaryLinesearchStepsize{TRM,VTM,F} <: Linesearch

Do a backtracking line search to find a step size ``α`` that fulfils the
Wolfe conditions along a search direction ``X`` starting from ``p``.
See [`WolfePowellBinaryLinesearch`](@ref) for the math details.

# Fields

* `sufficient_decrease::F`, `sufficient_curvature::F`: two constants in the line search
* `last_stepsize::F`: the last computed stepsize
$(_fields(:retraction_method))
* `stop_when_stepsize_less::F`: a safeguard to stop when the stepsize gets too small
$(_fields(:vector_transport_method))

# Constructor

    WolfePowellBinaryLinesearchStepsize(M::AbstractManifold; kwargs...)

## Keyword arguments

* `sufficient_decrease=1e-4`
* `sufficient_curvature=0.999`
$(_kwargs(:retraction_method))
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
$(_kwargs(:vector_transport_method))

"""
mutable struct WolfePowellBinaryLinesearchStepsize{
        TRM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod, F,
    } <: Linesearch
    retraction_method::TRM
    vector_transport_method::VTM
    sufficient_decrease::F
    sufficient_curvature::F
    last_stepsize::F
    stop_when_stepsize_less::F
    function WolfePowellBinaryLinesearchStepsize(
            M::AbstractManifold;
            sufficient_decrease::Real = 10.0^(-4),
            sufficient_curvature::Real = 0.999,
            retraction_method::RTM = default_retraction_method(M),
            vector_transport_method::VTM = default_vector_transport_method(M),
            stop_when_stepsize_less::Real = 0.0,
            last_stepsize::Real = 0.0,
        ) where {VTM <: AbstractVectorTransportMethod, RTM <: AbstractRetractionMethod}
        F = promote_type(typeof(sufficient_decrease), typeof(sufficient_curvature), typeof(stop_when_stepsize_less), typeof(last_stepsize))
        return new{RTM, VTM, F}(
            retraction_method, vector_transport_method,
            convert(F, sufficient_decrease), convert(F, sufficient_curvature),
            convert(F, last_stepsize), convert(F, stop_when_stepsize_less),
        )
    end
end
function (a::WolfePowellBinaryLinesearchStepsize)(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Int, η = (-get_gradient(amp, get_iterate(ams)));
        kwargs...,
    )
    M = get_manifold(amp)
    α = 0.0
    β = Inf
    t = 1.0
    p = get_iterate(ams)
    f0 = get_cost(amp, p)
    xNew = ManifoldsBase.retract_fused(M, p, η, t, a.retraction_method)
    fNew = get_cost(amp, xNew)
    X_tmp = zero_vector(M, p)
    η_xNew = vector_transport_to(M, p, η, xNew, a.vector_transport_method)
    nAt = fNew > f0 + a.sufficient_decrease * t * get_differential(amp, p, η; Y = X_tmp)
    nWt =
        get_differential(amp, xNew, η_xNew; Y = X_tmp) <
        a.sufficient_curvature * get_differential(amp, p, η; Y = X_tmp)
    while (nAt || nWt) &&
            (t > a.stop_when_stepsize_less) &&
            ((α + β) / 2 - 1 > a.stop_when_stepsize_less)
        nAt && (β = t)            # A(t) fails
        (!nAt && nWt) && (α = t)  # A(t) holds but W(t) fails
        t = isinf(β) ? 2 * α : (α + β) / 2
        # Update trial point
        ManifoldsBase.retract_fused!(M, xNew, get_iterate(ams), η, t, a.retraction_method)
        fNew = get_cost(amp, xNew)
        vector_transport_to!(
            M, η_xNew, get_iterate(ams), η, xNew, a.vector_transport_method
        )
        # Update conditions
        nAt = fNew > f0 + a.sufficient_decrease * t * get_differential(amp, p, η; Y = X_tmp)
        nWt =
            get_differential(amp, xNew, η_xNew; Y = X_tmp) <
            a.sufficient_curvature * get_differential(amp, p, η; Y = X_tmp)
    end
    a.last_stepsize = t
    return t
end
function get_last_stepsize(step::WolfePowellBinaryLinesearchStepsize, ::Any...)
    return step.last_stepsize
end
function Base.show(io::IO, a::WolfePowellBinaryLinesearchStepsize)
    print(io, "WolfePowellBinaryLinesearchStepsize(; sufficient_decrease = ", a.sufficient_decrease)
    print(io, ", sufficient_curvature = ", a.sufficient_curvature)
    print(io, ", last_stepsize = ", a.last_stepsize)
    print(io, ", retraction_method = ", a.retraction_method, ", stop_when_stepsize_less = ", a.stop_when_stepsize_less)
    print(io, ", vector_transport_method = ", a.vector_transport_method)
    return print(io, ")")
end
function status_summary(a::WolfePowellBinaryLinesearchStepsize; context::Symbol = :default)
    (context === :short) && return repr(a)
    (context === :inline) && return "A Wolfe Powell bisection dissection step size (last stepsize: $(a.last_stepsize))"
    return """
    A Wolfe Powell bisection line search based step size
    (last stepsize: $(a.last_stepsize))

    ## Parameters
    * retraction method:       $(_MANOPT_INDENT)$(a.retraction_method)
    * vector transport method: $(_MANOPT_INDENT)$(a.vector_transport_method)
    * sufficient decrease:     $(_MANOPT_INDENT)$(a.sufficient_decrease)
    * sufficient curvature:    $(_MANOPT_INDENT)$(a.sufficient_curvature)
    """
end

_doc_WPBL_algorithm = """With
```math
A(t) = f(p_+) ≤ f(p) + c_1 t ⟨$(_tex(:grad))f(p), X⟩_{p}
$(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad))
W(t) = ⟨$(_tex(:grad))f(p_+), $(_math(:VectorTransport, "p", "p_+"))X⟩_{p_+} ≥ c_2 ⟨X, $(_tex(:grad))f(p)⟩_p,
```

where ``p_+ =$(_tex(:retr))_p(tX)`` is the current trial point, and ``$(_math(:VectorTransport))`` denotes a
vector transport.
Then the following Algorithm is performed similar to Algorithm 7 from [Huang:2014](@cite)

1. set ``α=0``, ``β=∞`` and ``t=1``.
2. While either ``A(t)`` does not hold or ``W(t)`` does not hold do steps 3-5.
3. If ``A(t)`` fails, set ``β=t``.
4. If ``A(t)`` holds but ``W(t)`` fails, set ``α=t``.
5. If ``β<∞`` set ``t=$(_tex(:frac, "α+β", "2"))``, otherwise set ``t=2α``.
"""

"""
    WolfePowellBinaryLinesearch(; kwargs...)
    WolfePowellBinaryLinesearch(M::AbstractManifold; kwargs...)

Perform a linesearch to fulfill both the Armijo-Goldstein conditions
for some given sufficient decrease coefficient ``c_1`` and some sufficient curvature condition coefficient ``c_2``.
Compared to [`WolfePowellLinesearch`](@ref Manopt.WolfePowellLinesearch) which tries a simpler method, this linesearch performs the following algorithm

$(_doc_WPBL_algorithm)

# Keyword arguments

* `sufficient_decrease=1e-4`
* `sufficient_curvature=0.999`
$(_kwargs(:retraction_method))
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
$(_kwargs(:vector_transport_method))
"""
function WolfePowellBinaryLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(WolfePowellBinaryLinesearchStepsize, args...; kwargs...)
end


"""
    default_point_distance(::AbstractManifold, p)

The default Hager-Zhang guess for the distance between `p` and the solution to the optimization
problem. The default is 0, which deactivates heuristic I0 (a).
On each manifold with `default_point_distance`, you need to also implement `default_vector_norm`.
"""
default_point_distance(::AbstractManifold, p) = zero(number_eltype(p))

"""
    default_point_distance(::DefaultManifold, p)

Following [HagerZhang:2006:2](@cite), the expected distance to the optimal solution from `p`
on `DefaultManifold` is the `Inf` norm of `p`.
"""
default_point_distance(::DefaultManifold, p) = norm(p, Inf)

"""
    default_vector_norm(M::AbstractManifold, p, X)

The norm used by the Hager-Zhang initial guess to measure the search direction `X` at `p`.
There is no default implementation, because it is only needed on manifolds that also provide
a specific [`default_point_distance`](@ref) method.
"""
default_vector_norm(M::AbstractManifold, p, X)
default_vector_norm(::DefaultManifold, p, X) = norm(X, Inf)


"""
    HagerZhangInitialGuess{TF <: Real, TPN, TVN} <: AbstractInitialLinesearchGuess

Initial line search guess from the paper [HagerZhang:2006:2](@cite), following their
initial-guess procedure `I0`. The line search was adapted to the Riemannian setting by
introducing customizable norms for points and tangent vectors and a maximum stepsize `alphamax`.
"""
struct HagerZhangInitialGuess{TF <: Real, TPN, TVN} <: AbstractInitialLinesearchGuess
    ψ0::TF
    ψ1::TF
    ψ2::TF
    constant_guess::TF
    quadstep::Bool
    point_distance::TPN
    vector_norm::TVN
    zero_abstol::TF
    alphamax::TF
end

HagerZhangInitialGuess() = HagerZhangInitialGuess{Float64}()
function HagerZhangInitialGuess{TF}(;
        ψ0::TF = 0.01,
        ψ1::TF = 0.01,
        ψ2::TF = 2.0,
        constant_guess::TF = NaN,
        quadstep::Bool = true,
        point_distance::TPN = default_point_distance,
        vector_norm::TVN = default_vector_norm,
        zero_abstol::TF = eps(TF),
        alphamax::TF = Inf,
    ) where {TF <: Real, TPN, TVN}
    return HagerZhangInitialGuess{TF, TPN, TVN}(
        ψ0, ψ1, ψ2, constant_guess, quadstep,
        point_distance, vector_norm, zero_abstol, alphamax
    )
end

function (hzi::HagerZhangInitialGuess{TF})(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState,
        k::Int, last_stepsize::Real, η;
        lf0 = get_cost(mp, get_iterate(s)),
        Dlf0 = get_differential(mp, get_iterate(s), η),
        kwargs...
    ) where {TF <: Real}
    M = get_manifold(mp)
    p = get_iterate(s)
    abs_lf0 = abs(lf0)

    alphamax = min(hzi.alphamax, max_stepsize(M, p))

    if :stop_when_stepsize_exceeds in keys(kwargs)
        alphamax = min(
            kwargs[:stop_when_stepsize_exceeds],
            alphamax,
        )
    end

    if k == 1
        point_d = hzi.point_distance(M, p)
        # Step I0
        if isnan(hzi.constant_guess)
            if point_d > hzi.zero_abstol
                # I0.(a)
                ηn = hzi.vector_norm(M, p, η)
                return min(hzi.ψ0 * point_d / ηn, alphamax)
            elseif abs_lf0 > hzi.zero_abstol
                # I0.(b)
                return min(hzi.ψ0 * abs_lf0 / norm(M, p, η)^2, alphamax)
            else
                # I0.(c)
                return one(TF)
            end
        else
            return hzi.constant_guess
        end
    else
        if hzi.quadstep
            # attempt step I1
            step_R = hzi.ψ1 * last_stepsize
            f_R = get_cost(mp, ManifoldsBase.retract_fused(M, p, η, step_R, default_retraction_method(M, typeof(p))))
            # solving quadratic fit to the line given lf0, Dlf0 and cost at f_R
            q_b = Dlf0
            q_a = (f_R - q_b * step_R - lf0) / step_R^2

            if f_R ≤ lf0 && isfinite(q_a) && q_a > hzi.zero_abstol
                # if condition is false, we go to step I2
                a_min = -q_b / (2 * q_a)
                return min(a_min, alphamax)
            end
        end
        # step I2
        return min(hzi.ψ2 * last_stepsize, alphamax)
    end
end

@doc """
    HagerZhangLinesearchStepsize{TF<:Real,TIG,TRM,TVTM,TP,TX} <: Linesearch

Do a bracketing line search to find a step size ``α`` that finds a
local minimum along the search direction ``X`` starting from ``p``,
utilizing cubic polynomial interpolation using the method described in
[HagerZhang:2006:2](@cite). The function [`secant`](@ref) is used to find the minimum of the
cubic polynomial fitted to values of the cost function and its derivative at the endpoints
of the current interval.
See [`HagerZhangLinesearch`](@ref) for the mathematical details.

# Fields

$(_fields(:p; name = "candidate_point"))
  as temporary storage for candidates
$(_fields(:retraction_method))
$(_fields(:vector_transport_method))
* `initial_guess`: see keyword arguments of [`HagerZhangLinesearch`](@ref) for details.
* `stepsize_limit`: see keyword arguments of [`HagerZhangLinesearch`](@ref) for details.
* `max_bracket_iterations`: see keyword arguments of [`HagerZhangLinesearch`](@ref) for details.
* `start_enforcing_wolfe_conditions_at_bracketing_iteration`: see keyword arguments of
  [`HagerZhangLinesearch`](@ref) for details.
* `allow_early_maxstep_termination`: see keyword arguments of [`HagerZhangLinesearch`](@ref) for details.
* `wolfe_condition_mode`: see keyword arguments of [`HagerZhangLinesearch`](@ref) for details.
* `ϵ`, `δ`, `σ`, `ω`, `θ`, `γ`, `ρ`, `Δ`: see keyword arguments of [`HagerZhangLinesearch`](@ref) for details.
* `secant_acceptance_ratio`: see keyword arguments of [`HagerZhangLinesearch`](@ref) for details.
* `candidate_direction`, `temporary_tangent`: as temporary storage for tangent vectors
* `triples`: temporary storage for function and derivative evaluations
* `last_evaluation_index`: to keep track of the number of evaluations performed so far;
  points at the last filled entry of `triples`.
* `Qₖ`, `Cₖ`: to keep track of the parameters of the Wolfe condition when in adaptive mode
* `current_mode`: to keep track of the current Wolfe condition mode when in adaptive mode
* `last_stepsize`: last stepsize computed since reset
* `last_cost`: last cost value computed since reset
* `ϵₖ`: the current ϵ parameter used in the approximate Wolfe condition and bracketing

# Constructor

    HagerZhangLinesearchStepsize(M::AbstractManifold; kwargs...)
"""
mutable struct HagerZhangLinesearchStepsize{
        TF <: Real,
        TIG <: AbstractInitialLinesearchGuess,
        TRM <: AbstractRetractionMethod,
        TVTM <: AbstractVectorTransportMethod,
        TP,
        TX,
    } <: Linesearch
    # parameters
    initial_guess::TIG
    retraction_method::TRM
    vector_transport_method::TVTM
    stepsize_limit::TF
    max_bracket_iterations::Int
    start_enforcing_wolfe_conditions_at_bracketing_iteration::Int
    allow_early_maxstep_termination::Bool
    wolfe_condition_mode::Symbol # :standard, :approximate, :adaptive
    ϵ::TF # approximate Wolfe termination parameter
    δ::TF # used in approximate Wolfe condition
    σ::TF # used in curvature condition
    ω::TF
    θ::TF # update rule parameter
    γ::TF
    ρ::TF
    Δ::TF
    secant_acceptance_ratio::TF
    # storage for candidates
    candidate_point::TP
    candidate_direction::TX
    temporary_tangent::TX
    # storage for function evaluations
    triples::Vector{UnivariateTriple{TF}}
    last_evaluation_index::Int
    # storage to be kept between outer solver iterations
    Qₖ::TF
    Cₖ::TF
    current_mode::Symbol
    # other storage
    last_stepsize::TF
    last_cost::TF
    ϵₖ::TF
    function HagerZhangLinesearchStepsize(
            M::AbstractManifold;
            initial_guess::TIG = HagerZhangInitialGuess(),
            retraction_method::TRM = default_retraction_method(M),
            vector_transport_method::TVTM = default_vector_transport_method(M),
            initial_last_stepsize::TF = NaN,
            initial_last_cost::TF = NaN,
            stepsize_limit::TF = Inf,
            candidate_point = allocate_result(M, rand),
            candidate_direction = zero_vector(M, candidate_point),
            max_bracket_iterations::Int = 10,
            start_enforcing_wolfe_conditions_at_bracketing_iteration::Int = initial_guess isa ConstantInitialGuess ? 2 : 1,
            max_function_evaluations::Int = 20,
            wolfe_condition_mode::Symbol = :adaptive,
            allow_early_maxstep_termination::Bool = true,
            ϵ::TF = 1.0e-6,
            δ::TF = 0.1,
            σ::TF = 0.9,
            ω::TF = 1.0e-3,
            θ::TF = 0.5,
            γ::TF = 0.66,
            ρ::TF = 5.0,
            Δ::TF = 0.7,
            secant_acceptance_ratio::TF = 1.0e-8,
        ) where {
            TIG <: AbstractInitialLinesearchGuess, TRM <: AbstractRetractionMethod,
            TVTM <: AbstractVectorTransportMethod, TF <: Real,
        }

        # check parameters
        @assert δ > 0 && δ < 0.5
        @assert δ <= σ
        @assert σ < 1
        @assert ϵ >= 0
        @assert ω >= 0 && ω <= 1
        @assert Δ >= 0 && Δ <= 1
        @assert θ > 0 && θ < 1
        @assert γ > 0 && γ < 1
        @assert ρ > 1
        @assert stepsize_limit > 0
        @assert wolfe_condition_mode in (:standard, :approximate, :adaptive)
        @assert secant_acceptance_ratio >= 0

        # allocate storage
        triples = Vector{UnivariateTriple{TF}}(undef, max_function_evaluations)

        initial_wolfe_mode = wolfe_condition_mode == :adaptive ? :standard : wolfe_condition_mode

        return new{TF, TIG, TRM, TVTM, typeof(candidate_point), typeof(candidate_direction)}(
            initial_guess, retraction_method, vector_transport_method, stepsize_limit,
            max_bracket_iterations, start_enforcing_wolfe_conditions_at_bracketing_iteration,
            allow_early_maxstep_termination, wolfe_condition_mode,
            ϵ, δ, σ, ω, θ, γ, ρ, Δ, secant_acceptance_ratio,
            candidate_point, candidate_direction, zero_vector(M, candidate_point),
            triples, 0,
            0.0, 0.0, # Qₖ, Cₖ
            initial_wolfe_mode,
            initial_last_stepsize, initial_last_cost, ϵ,
        )
    end
end

function initialize_stepsize!(hzls::HagerZhangLinesearchStepsize)
    hzls.Qₖ = 0.0
    hzls.Cₖ = 0.0
    hzls.last_stepsize = NaN
    hzls.last_cost = NaN
    hzls.ϵₖ = hzls.ϵ
    hzls.current_mode = hzls.wolfe_condition_mode
    if hzls.current_mode === :adaptive
        hzls.current_mode = :standard
    end
    hzls.last_evaluation_index = 0
    return hzls
end

"""
    _hz_evaluate_next_step(
        hzls::HagerZhangLinesearchStepsize, M::AbstractManifold,
        mp::AbstractManoptProblem, p, η, α::Real
    )

Evaluate and store the next trial step for the Hager-Zhang linesearch.

Given the current iterate `p`, search direction `η` (in the tangent space at `p`), and a
candidate step size `α`, this function

1. Retracts from `p` along `η` by step `α` into `hzls.candidate_point` (using
     `hzls.retraction_method`),
2. Vector-transports `η` to the candidate point into `hzls.candidate_direction` (using
     `hzls.vector_transport_method`),
3. Evaluates the objective and directional derivative via
     `get_cost_and_differential(mp, hzls.candidate_point, hzls.candidate_direction)`,
4. Stores the resulting triple `(α, f, df)` in `hzls.triples` and increments
     `hzls.last_evaluation_index`.

This helper is side-effecting by design; it mutates `hzls`' internal storage.

# Return value

By default return a tuple with three values:
- the index `i_k::Int` at which the new evaluation was stored.
- `evaluation_limit_termination`: `true` iff the maximum number of stored evaluations
    has been reached.
- `wolfe_termination` is `true` iff the (standard or approximate) Wolfe conditions are
    satisfied for the current candidate, according to `hzls.current_mode`.

# Errors

Throws an error if called more often than the maximum number of allocated function
evaluations (i.e. if `hzls.triples` would overflow).
"""
function _hz_evaluate_next_step(
        hzls::HagerZhangLinesearchStepsize,
        M::AbstractManifold,
        mp::AbstractManoptProblem,
        p,
        η,
        α::Real
    )
    triples = hzls.triples
    max_evals = length(triples)
    if hzls.last_evaluation_index + 1 > max_evals
        # this should never happen if the calling code is correct
        error("Hager-Zhang linesearch exceeded maximum number of function evaluations $(length(hzls.triples)).")
    end
    ManifoldsBase.retract_fused!(M, hzls.candidate_point, p, η, α, hzls.retraction_method)
    vector_transport_to!(
        M, hzls.candidate_direction, p, η, hzls.candidate_point, hzls.vector_transport_method
    )
    f, df = get_cost_and_differential(mp, hzls.candidate_point, hzls.candidate_direction; gradient = hzls.temporary_tangent)
    hzls.last_evaluation_index += 1
    triples[hzls.last_evaluation_index] = UnivariateTriple(α, f, df)

    wolfe_termination = false
    evaluation_limit_termination = hzls.last_evaluation_index == max_evals
    i_k = hzls.last_evaluation_index
    if hzls.current_mode === :standard
        # Eq (22) in HagerZhang:2006:2
        # equivalent to the (T1) condition
        wolfe_termination = (α * hzls.δ * triples[1].df >= (triples[i_k].f - triples[1].f)) &&
            (triples[i_k].df >= hzls.σ * triples[1].df)
    elseif hzls.current_mode === :approximate
        # Eq (23) in HagerZhang:2006:2 + additional criterion in the (T2) condition
        wolfe_termination = ((2 * hzls.δ - 1) * triples[1].df >= triples[i_k].df) &&
            (triples[i_k].df >= hzls.σ * triples[1].df) && triples[i_k].f <= triples[1].f + hzls.ϵₖ
    else
        error("Unknown Wolfe condition mode $(hzls.current_mode).")
    end

    return hzls.last_evaluation_index, evaluation_limit_termination, wolfe_termination
end

"""
    _hz_bracket(
        hzls::HagerZhangLinesearchStepsize, M::AbstractManifold,
        mp::AbstractManoptProblem, p, η, c::Real, max_alpha::Real
    )

Perform the bracketing phase of the Hager-Zhang linesearch starting from an initial
stepsize `c` and not exceeding `max_alpha`.

Returns a tuple `(i_a, i_b, f_eval, f_wolfe, f_early_maxstep)` where `i_a` and `i_b` are
the indices in the stored function evaluations such that the minimum is bracketed between
`triples[i_a].t` and `triples[i_b].t`. `f_eval` is `true` if the maximum number of function
evaluations has been reached during the bracketing phase. `f_wolfe` is `true` if the Wolfe
conditions have been satisfied. `f_early_maxstep` is `true` if the maximum stepsize was
reached early with negative slope and an improvement over the initial point.
"""
function _hz_bracket(
        hzls::HagerZhangLinesearchStepsize, M::AbstractManifold,
        mp::AbstractManoptProblem, p, η, c::Real, max_alpha::Real
    )
    # B0
    current_step = c
    local c_index, f_eval, f_wolfe # COV_EXCL_LINE
    ls_early_exit = false
    for j in 1:hzls.max_bracket_iterations
        c_index, f_eval, f_wolfe = _hz_evaluate_next_step(hzls, M, mp, p, η, current_step)
        if f_eval || (f_wolfe && j >= hzls.start_enforcing_wolfe_conditions_at_bracketing_iteration)
            break
        end
        if hzls.triples[c_index].df >= 0
            # B1 -- detecting a positive slope
            # handled after the loop
            break
        else
            if hzls.triples[c_index].f > hzls.triples[1].f + hzls.ϵₖ
                # B2 -- function value gets sufficiently larger than at 0
                # perform main bracketing loop (we can skip U0-U2 checks here)
                (i_a_bar, i_b_bar, f_eval, f_wolfe) = _hz_u3(hzls, M, mp, p, η, 1, c_index)
                return (i_a_bar, i_b_bar, f_eval, f_wolfe, false)
            else
                if current_step == max_alpha
                    # we've reached maximum alpha so we can't expand anymore
                    # we handle this case after the loop
                    ls_early_exit = hzls.allow_early_maxstep_termination
                    break
                end
                # B3 -- widen the bracket
                current_step *= hzls.ρ
                if current_step > max_alpha
                    current_step = max_alpha
                end
            end
        end
    end
    # we detected positive slope, ran out of iterations or reached max stepsize
    # B1 seems to be the best choice for all three cases

    if ls_early_exit
        # additional termination condition: we reached the maximum stepsize with negative
        # slope and an improvement over the initial point, so we can exit early with this step
        return (1, c_index, f_eval, f_wolfe, true)
    end

    i_min = 1
    for i in 2:(hzls.last_evaluation_index - 1)
        if hzls.triples[i].f <= hzls.triples[1].f + hzls.ϵₖ
            i_min = i
        end
    end
    return (i_min, c_index, f_eval, f_wolfe, false)
end

"""
    _hz_update(
        hzls::HagerZhangLinesearchStepsize, M::AbstractManifold,
        mp::AbstractManoptProblem, p, η, i_a::Int, i_b::Int, c::Real
    )

Perform an update procedure of the Hager-Zhang linesearch given the current bracketing
indices `i_a` and `i_b` and a candidate stepsize `c`.

Returns indices and termination information `(i_A, i_B, i_c, f_eval, f_wolfe)` where the
minimum is now bracketed between `alpha_values[i_A]` and `alpha_values[i_B]`. Index `i_c`
indicates the position at which evaluation of the candidate `c` was stored. If the
candidate `c` is outside of the current bracket, the last index is returned as `-1`.
`f_eval` is `true` if the maximum number of function evaluations has been reached.
`f_wolfe` is `true` if the Wolfe conditions have been satisfied at the candidate `i_c`.
"""
function _hz_update(
        hzls::HagerZhangLinesearchStepsize, M::AbstractManifold,
        mp::AbstractManoptProblem, p, η, i_a::Int, i_b::Int, c::Real
    )
    # U0
    if c < hzls.triples[i_a].t || c > hzls.triples[i_b].t
        return (i_a, i_b, -1, false, false)
    end
    i_c, f_eval, f_wolfe = _hz_evaluate_next_step(hzls, M, mp, p, η, c)
    if hzls.triples[i_c].df >= 0
        # U1
        return (i_a, i_c, i_c, f_eval, f_wolfe)
    else
        if hzls.triples[i_c].f <= hzls.triples[1].f + hzls.ϵₖ
            # U2
            return (i_c, i_b, i_c, f_eval, f_wolfe)
        else
            if f_eval || f_wolfe
                # termination condition met
                return (i_a, i_b, i_c, f_eval, f_wolfe)
            else
                # U3
                i_a_bar, i_b_bar, f_eval, f_wolfe = _hz_u3(hzls, M, mp, p, η, i_a, i_c)
                return (i_a_bar, i_b_bar, i_c, f_eval, f_wolfe)
            end
        end
    end
end

function _hz_u3(
        hzls::HagerZhangLinesearchStepsize, M::AbstractManifold,
        mp::AbstractManoptProblem, p, η, i_a::Int, i_b::Int
    )
    i_a_bar = i_a
    i_b_bar = i_b
    # the loop should typically terminate before exceeding the number of evaluations
    f_eval = false
    f_wolfe = false
    while hzls.last_evaluation_index < length(hzls.triples)
        # U3 (a)
        d = (1 - hzls.θ) * hzls.triples[i_a_bar].t + hzls.θ * hzls.triples[i_b_bar].t
        i_d, f_eval, f_wolfe = _hz_evaluate_next_step(hzls, M, mp, p, η, d)
        if hzls.triples[i_d].df >= 0 || f_eval || f_wolfe
            return (i_a_bar, i_d, f_eval, f_wolfe)
        else
            if hzls.triples[i_d].f <= hzls.triples[1].f + hzls.ϵₖ
                # U3 (b)
                i_a_bar = i_d
            else
                # U3 (c)
                i_b_bar = i_d
            end
        end
    end
    return (i_a_bar, i_b_bar, f_eval, f_wolfe)
end

"""
    _hz_secant2(
        hzls::HagerZhangLinesearchStepsize, M::AbstractManifold,
        mp::AbstractManoptProblem, p, η, i_a::Int, i_b::Int
    )

Perform the secant-based update in the Hager-Zhang linesearch.

Computes a trial step using a secant interpolation of the bracketing
endpoints. If the trial step is too close to an endpoint, falls back to a
bisection step. Returns the updated bracketing indices and termination flags
from the internal update routine.

# Arguments
- `hzls`: linesearch state and storage.
- `M`: manifold for retractions and transports.
- `mp`: optimization problem providing cost and differential.
- `p`: current iterate.
- `η`: search direction in the tangent space at `p`.
- `i_a`, `i_b`: indices of the current bracketing interval in `hzls.triples`.

# Return value
Returns `(i_A, i_B, i_c, f_eval, f_wolfe)` where
- `i_A`, `i_B`: indices bracketing the minimum after the update,
- `i_c`: index of the most recent evaluation (or `-1` if the candidate was out of range),
- `f_eval`: `true` iff the evaluation limit has been reached,
- `f_wolfe`: `true` iff the Wolfe conditions are satisfied.

# Steps (S1-S4)
- S1: compute a secant trial `c` from the current bracket and accept it unless too close to
  an endpoint (otherwise use a bisection step).
- S2/S3: if the trial becomes a new endpoint, perform an update from that side.
- S4: return the updated bracket and termination flags.
"""
function _hz_secant2(
        hzls::HagerZhangLinesearchStepsize, M::AbstractManifold,
        mp::AbstractManoptProblem, p, η, i_a::Int, i_b::Int
    )
    # S1
    c = secant(hzls.triples[i_a], hzls.triples[i_b])
    width = hzls.triples[i_b].t - hzls.triples[i_a].t
    if abs(c - hzls.triples[i_a].t) < hzls.secant_acceptance_ratio * width ||
            abs(c - hzls.triples[i_b].t) < hzls.secant_acceptance_ratio * width
        # secant too close to an endpoint, use bisection instead
        # this case is not present in the original algorithm, but the following steps don't make much sense in this case
        c = (hzls.triples[i_a].t + hzls.triples[i_b].t) / 2
        return _hz_update(hzls, M, mp, p, η, i_a, i_b, c)
    end
    (i_A, i_B, i_c, f_eval, f_wolfe) = _hz_update(hzls, M, mp, p, η, i_a, i_b, c)
    if f_eval || f_wolfe
        # not present in the original algorithm, but this seems to be the right way to handle this case
        return (i_A, i_B, i_c, f_eval, f_wolfe)
    end
    if i_c == i_B
        # S2
        c_bar = secant(hzls.triples[i_b], hzls.triples[i_B])
        # S4, part 1
        return _hz_update(hzls, M, mp, p, η, i_A, i_B, c_bar)
    elseif i_c == i_A
        # S3
        c_bar = secant(hzls.triples[i_a], hzls.triples[i_A])
        # S4, part 1
        return _hz_update(hzls, M, mp, p, η, i_A, i_B, c_bar)
    else
        # S4, part 2
        return (i_A, i_B, i_c, f_eval, f_wolfe)
    end
end

function (hzls::HagerZhangLinesearchStepsize)(
        mp::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        k::Int,
        η = (-get_gradient(mp, get_iterate(s)));
        fp = get_cost(mp, get_iterate(s)),
        gradient = nothing,
        kwargs...,
    )
    M = get_manifold(mp)
    p = get_iterate(s)

    dphi_0 = if !isnothing(gradient)
        real(inner(M, p, η, gradient))
    else
        get_differential(mp, p, η; Y = hzls.temporary_tangent)
    end
    hzls.triples[1] = UnivariateTriple(0.0, fp, dphi_0)
    hzls.last_evaluation_index = 1

    # update Qₖ, Cₖ
    hzls.Qₖ = 1 + hzls.Qₖ * hzls.Δ
    hzls.Cₖ += (abs(fp) - hzls.Cₖ) / hzls.Qₖ

    if hzls.wolfe_condition_mode == :adaptive
        # Checking the V3 condition
        if abs(hzls.last_cost - fp) <= hzls.ω * hzls.Cₖ
            hzls.current_mode = :approximate
        end
    end

    # L0, initialization

    # handle stepsize limit
    max_alpha = hzls.stepsize_limit
    if :stop_when_stepsize_exceeds in keys(kwargs)
        max_alpha = min(
            kwargs[:stop_when_stepsize_exceeds],
            max_alpha,
        )
    end
    # guess initial alpha
    α0 = hzls.initial_guess(mp, s, k, hzls.last_stepsize, η; lf0 = fp, Dlf0 = dphi_0, stop_when_stepsize_exceeds = max_alpha)

    # in case initial_guess does not take into account the stepsize limit, we enforce it here
    α0 = min(α0, max_alpha)

    # L0, bracket(c)
    local i_a_j, i_b_j, f_eval, f_wolfe # COV_EXCL_LINE
    (i_a_j, i_b_j, f_eval, f_wolfe, f_early_maxstep) = _hz_bracket(hzls, M, mp, p, η, α0, max_alpha)
    !f_early_maxstep && while !(f_eval || f_wolfe)
        # L1
        finite_at_b = isfinite(hzls.triples[i_b_j].f)
        if finite_at_b
            # _hz_secant2 only makes sense if we have finite function values at both ends
            # but _hz_update may still work
            (i_a, i_b, _i_c, f_eval, f_wolfe) = _hz_secant2(hzls, M, mp, p, η, i_a_j, i_b_j)
        else
            (i_a, i_b) = (i_a_j, i_b_j)
        end
        # L2
        # we additionally check that we can continue narrowing the bracket
        if !(f_eval || f_wolfe) &&
                (!finite_at_b || (hzls.triples[i_b].t - hzls.triples[i_a].t) > hzls.γ * (hzls.triples[i_b_j].t - hzls.triples[i_a_j].t))
            # secant2 did not reduce the bracket sufficiently
            # we need to do bisection
            (i_a, i_b, _i_c, f_eval, f_wolfe) = _hz_update(
                hzls, M, mp, p, η,
                i_a, i_b,
                (hzls.triples[i_a].t + hzls.triples[i_b].t) / 2,
            )
        end
        # L3
        i_a_j, i_b_j = i_a, i_b

        # loop terminates when we generate a point satisfying T1 or T2, or when we run out
        # of objective evaluations
    end

    hzls.last_stepsize = hzls.triples[hzls.last_evaluation_index].t
    hzls.last_cost = hzls.triples[hzls.last_evaluation_index].f
    return hzls.last_stepsize
end

function Base.show(io::IO, hzls::HagerZhangLinesearchStepsize)
    return print(
        io,
        """
        HagerZhangLinesearch(;
            initial_guess = $(hzls.initial_guess),
            retraction_method = $(hzls.retraction_method),
            vector_transport_method = $(hzls.vector_transport_method),
            stepsize_limit = $(hzls.stepsize_limit),
            max_bracket_iterations = $(hzls.max_bracket_iterations),
            start_enforcing_wolfe_conditions_at_bracketing_iteration = $(hzls.start_enforcing_wolfe_conditions_at_bracketing_iteration),
            max_function_evaluations = $(length(hzls.triples)),
            wolfe_condition_mode = $(hzls.wolfe_condition_mode),
            ϵ = $(hzls.ϵ), δ = $(hzls.δ), σ = $(hzls.σ),
            ω = $(hzls.ω),
            θ = $(hzls.θ), γ = $(hzls.γ), secant_acceptance_ratio = $(hzls.secant_acceptance_ratio),
            ρ = $(hzls.ρ),
            Δ = $(hzls.Δ),
        )""",
    )
end
function status_summary(hzls::HagerZhangLinesearchStepsize; context::Symbol = :default)
    (context === :short) && (return repr(hzls))
    (context === :inline) && (return "A Hager-Zhang linesearch stepsize method.")
    return "$(hzls)\nand a computed last stepsize of $(hzls.last_stepsize)"
end

@doc """
    HagerZhangLinesearch(; kwargs...)
    HagerZhangLinesearch(M::AbstractManifold; kwargs...)

A functor representing the line search introduced in [HagerZhang:2006:2](@cite).

It finds a step size satisfying the (standard or approximate) Wolfe conditions by bracketing
and then narrowing the bracket with secant and bisection steps.

The following changes were made to the original algorithm from the paper:
1. The algorithm bails out early of a secant update that is too close to one of the end
   points and switches to bisection. Original algorithm performs a similar check at a later
   stage. This precaution prevents a non-productive evaluation of the objective.
2. Added `start_enforcing_wolfe_conditions_at_bracketing_iteration`, since with a very low
   stepsize initialization that satisfies Wolfe conditions we might accept the initial
   stepsize and not notice that bracketing could help us reach the minimum earlier.
   Setting `start_enforcing_wolfe_conditions_at_bracketing_iteration` to 1 reproduces the
   behavior of the original paper. For example a static initial stepsize equal to 1.0 could
   benefit from having this parameter increased.
3. The paper isn't entirely clear on what the final stepsize to return is. This
   implementation returns the last evaluated stepsize.
4. The original algorithm doesn't specify what to do when the maximum stepsize is reached
   during the bracketing phase with a negative slope and an improvement over the initial
   point. This implementation allows for an early termination in this case, which seems
   reasonable since we can't expand the bracket anymore and this point is likely close to
   the minimum. By default this early termination is allowed, but it can be turned off via
   `allow_early_maxstep_termination` in which case the algorithm continues with the main
   loop even in this case.

## Keyword arguments

$(_kwargs(:p; name = "candidate_point")) as temporary storage for candidates
$(_kwargs(:retraction_method))
$(_kwargs(:vector_transport_method))
* `initial_guess::AbstractInitialLinesearchGuess=HagerZhangInitialGuess()`: initial linesearch guess strategy
* `initial_last_stepsize::Real = NaN`: initial value for the stored last stepsize
* `initial_last_cost::Real = NaN`: initial value for the stored last cost
* `stepsize_limit::Real = Inf`: upper bound for trial stepsizes during bracketing
* `candidate_direction = zero_vector(M, candidate_point)`: storage for transported directions
* `max_bracket_iterations::Int = 10`: maximum number of bracketing iterations
* `start_enforcing_wolfe_conditions_at_bracketing_iteration::Int = initial_guess isa ConstantInitialGuess ? 2 : 1`:
  bracketing iteration number at which Wolfe conditions are started to be enforced;
  setting to 1 may cause no bracketing to occur when the initial guess satisfies the Wolfe
  conditions.
* `max_function_evaluations::Int = 20`: maximum number of function evaluations per linesearch
* `allow_early_maxstep_termination::Bool = true`: whether to allow early termination when
  the maximum stepsize is reached with negative slope and an improvement over the initial point.
* `wolfe_condition_mode::Symbol = :adaptive`: one of `:standard`, `:approximate`, or `:adaptive`.
  Selects between (T1) and (T2) conditions in [HagerZhang:2006:2](@cite).
* `ϵ::Real = 1.0e-6`: initial allowed increase in function value in termination condition (T2).
  Allowed range: `ϵ >= 0`.
* `δ::Real = 0.1`: parameter for approximate Wolfe condition.
  Allowed range: `0 < δ < 0.5` and `δ <= σ`.
* `σ::Real = 0.9`: curvature condition parameter. Allowed range: `δ <= σ < 1`.
* `ω::Real = 1.0e-3`: interpolation safeguard parameter. Allowed range: `0 <= ω <= 1`.
* `θ::Real = 0.5`: bisection update parameter. Allowed range: `0 < θ < 1`.
* `γ::Real = 0.66`: determines when a bisection step is performed instead of secant.
  Allowed range: `0 < γ < 1`.
* `ρ::Real = 5.0`: bracketing expansion factor. Allowed range: `ρ > 1`.
* `Δ::Real = 0.7`: Parameter controlling the rate of change of Qₖ.
  Allowed range: `0 <= Δ <= 1`.
* `secant_acceptance_ratio::Real = 1.0e-8`: minimum relative interval length
  for accepting secant step. Allowed range: `secant_acceptance_ratio >= 0`.
  In case of rejection, a bisection step is performed instead.

$(_note(:ManifoldDefaultsFactory, "HagerZhangLinesearch"))
"""
function HagerZhangLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(HagerZhangLinesearchStepsize, args...; kwargs...)
end
