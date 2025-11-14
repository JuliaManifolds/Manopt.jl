@doc """
    ArmijoLinesearchStepsize <: Linesearch

A functor `problem, state, k, X; kwargs...) -> s to provide an Armijo line search to compute step size,
based on the search direction `X`.

This functor accepts the following keyword arguments:


# Fields

* `additional_decrease_condition`: specify a condition a new point has to additionally
  fulfill. The default accepts all points.
* `additional_increase_condition`: specify a condtion that additionally to
  checking a valid increase has to be fulfilled. The default accepts all points.
* `candidate_point`:               to store an interim result
* `initial_stepsize`:              and initial step size
$(_var(:Keyword, :retraction_method))
* `contraction_factor`:            exponent for line search reduction
* `sufficient_decrease`:           gain within Armijo's rule
* `last_stepsize`:                 the last step size to start the search with
* $(_doc_stepsize_initial_guess)
* `messages::NamedTuple`:          a named tuple to store possible [`StepsizeMessage`](@ref) about the stepsize search.
* `stop_when_stepsize_less`:       smallest stepsize when to stop (the last one before is taken)
* `stop_when_stepsize_exceeds`:    largest stepsize when to stop.
* `stop_increasing_at_step`:       last step to increase the stepsize (phase 1),
* `stop_decreasing_at_step`:       last step size to decrease the stepsize (phase 2),

Pass `:Messages` to a `debug=` to see `@info`s when these happen.

# Constructor

    ArmijoLinesearchStepsize(M::AbstractManifold; kwarg...)

with the fields keyword arguments and the retraction is set to the default retraction on `M`.

## Keyword arguments

* `candidate_point=`(`allocate_result(M, rand)`)
* `initial_stepsize=1.0`
$(_var(:Keyword, :retraction_method))
* `contraction_factor=0.95`
* `sufficient_decrease=0.1`
* `initial_stepsize = 1.0` the first step size to plug into the `initial_guess` function.
* `initial_guess=`[`ArmijoInitialGuess`](@ref)
* `stop_when_stepsize_less=0.0`: stop when the stepsize decreased below this version.
* `stop_when_stepsize_exceeds=[`max_step`](@ref)`(M)`: provide an absolute maximal step size.
* `stop_increasing_at_step=100`: for the initial increase test, stop after these many steps
* `stop_decreasing_at_step=1000`: in the backtrack, stop after these many steps
"""
mutable struct ArmijoLinesearchStepsize{TRM <: AbstractRetractionMethod, P, I, F <: Real, IGF <: AbstractInitialLinesearchGuess, DF, IF, MSGS} <:
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
    function ArmijoLinesearchStepsize(
            M::AbstractManifold;
            additional_decrease_condition::DF = (M, p) -> true,
            additional_increase_condition::IF = (M, p) -> true,
            candidate_point::P = allocate_result(M, rand),
            contraction_factor::F = 0.95,
            initial_stepsize::F = 1.0,
            initial_guess::IGF = ArmijoInitialGuess(),
            retraction_method::TRM = default_retraction_method(M),
            stop_when_stepsize_less::F = 0.0,
            stop_when_stepsize_exceeds = max_stepsize(M),
            stop_increasing_at_step::I = 100,
            stop_decreasing_at_step::I = 1000,
            sufficient_decrease = 0.1,
        ) where {TRM, P, I, F, IGF, DF, IF}
        msgs = (;
            non_descent_direction = StepsizeMessage{F, F}(),
            stop_decreasing = StepsizeMessage{Int, F}(),
            stop_increasing = StepsizeMessage{Int, F}(),
            stepsize_less = StepsizeMessage{F, F}(),
            stepsize_exceeds = StepsizeMessage{F, F}(),
        )
        return new{TRM, P, I, F, IGF, DF, IF, typeof(msgs)}(
            candidate_point,
            contraction_factor,
            initial_guess,
            initial_stepsize,
            initial_stepsize,
            retraction_method,
            sufficient_decrease,
            stop_when_stepsize_less,
            stop_when_stepsize_exceeds,
            stop_increasing_at_step,
            stop_decreasing_at_step,
            additional_decrease_condition,
            additional_increase_condition,
            msgs,
        )
    end
end
function (a::ArmijoLinesearchStepsize)(
        mp::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        k::Int,
        η = (-get_gradient(mp, get_iterate(s)));
        gradient = nothing,
        kwargs...,
    )
    p = get_iterate(s)
    grad = isnothing(gradient) ? get_gradient(mp, get_iterate(s)) : gradient
    return a(mp, p, grad, η; initial_guess = a.initial_guess(mp, s, k, a.last_stepsize, η))
end
function (a::ArmijoLinesearchStepsize)(
        mp::AbstractManoptProblem, p, X, η; initial_guess = 1.0, kwargs...
    )
    reset_messages!(a.messages)
    l = norm(get_manifold(mp), p, η)
    a.last_stepsize = linesearch_backtrack!(
        get_manifold(mp),
        a.candidate_point,
        (M, p) -> get_cost_function(get_objective(mp))(M, p),
        p,
        initial_guess,
        a.sufficient_decrease,
        a.contraction_factor,
        η;
        gradient = X,
        retraction_method = a.retraction_method,
        stop_when_stepsize_less = (a.stop_when_stepsize_less / l),
        stop_when_stepsize_exceeds = (a.stop_when_stepsize_exceeds / l),
        stop_increasing_at_step = a.stop_increasing_at_step,
        stop_decreasing_at_step = a.stop_decreasing_at_step,
        additional_decrease_condition = a.additional_decrease_condition,
        additional_increase_condition = a.additional_increase_condition,
        report_messages_in = a.messages,
    )
    return a.last_stepsize
end
get_initial_stepsize(a::ArmijoLinesearchStepsize) = a.initial_stepsize
function show(io::IO, als::ArmijoLinesearchStepsize)
    return print(
        io,
        """
        ArmijoLinesearch(;
            initial_stepsize=$(als.initial_stepsize),
            retraction_method=$(als.retraction_method),
            contraction_factor=$(als.contraction_factor),
            sufficient_decrease=$(als.sufficient_decrease),
        )""",
    )
end
function status_summary(als::ArmijoLinesearchStepsize)
    return "$(als)\nand a computed last stepsize of $(als.last_stepsize)"
end
function get_message(a::ArmijoLinesearchStepsize)
    s = [get_message(kv[1], kv[2]) for kv in pairs(a.messages)]
    return join([m for m in s if length(m) > 0], "\n")
end
function get_parameter(a::ArmijoLinesearchStepsize, s::Val{:DecreaseCondition}, args...)
    return get_parameter(a.additional_decrease_condition, args...)
end
function get_parameter(a::ArmijoLinesearchStepsize, ::Val{:IncreaseCondition}, args...)
    return get_parameter(a.additional_increase_condition, args...)
end
function set_parameter!(a::ArmijoLinesearchStepsize, s::Val{:DecreaseCondition}, args...)
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

Specify a step size that performs an Armijo line search. Given a Function ``f:$(_math(:M))→ℝ``
and its Riemannian Gradient ``$(_tex(:grad))f: $(_math(:M))→$(_math(:TM))``,
the curent point ``p∈$(_math(:M))`` and a search direction ``X∈$(_math(:TpM))``.

Then the step size ``s`` is found by reducing the initial step size ``s`` until

```math
f($(_tex(:retr))_p(sX)) ≤ f(p) - τs ⟨ X, $(_tex(:grad))f(p) ⟩_p
```

is fulfilled. for a sufficient decrease value ``τ ∈ (0,1)``.

To be a bit more optimistic, if ``s`` already fulfils this, a first search is done,
__increasing__ the given ``s`` until for a first time this step does not hold.

Overall, we look for step size, that provides _enough decrease_, see
[Boumal:2023; p. 58](@cite) for more information.

# Keyword arguments

* `additional_decrease_condition=(M, p) -> true`:
  specify an additional criterion that has to be met to accept a step size in the decreasing loop
* `additional_increase_condition::IF=(M, p) -> true`:
  specify an additional criterion that has to be met to accept a step size in the (initial) increase loop
* `candidate_point=allocate_result(M, rand)`:
  speciy a point to be used as memory for the candidate points.
* `contraction_factor=0.95`: how to update ``s`` in the decrease step
* `initial_stepsize=1.0`: specify an initial step size
* `initial_guess=`[`ArmijoInitialGuess`](@ref): Compute the initial step size of
  a line search based on this function.
  The function required is `(p,s,k,l) -> α` and computes the initial step size ``α``
  based on a [`AbstractManoptProblem`](@ref) `p`, [`AbstractManoptSolverState`](@ref) `s`,
  the current iterate `k` and a last step size `l`.
$(_var(:Keyword, :retraction_method))
* `stop_when_stepsize_less=0.0`: a safeguard, stop when the decreasing step is below this (nonnegative) bound.
* `stop_when_stepsize_exceeds=max_stepsize(M)`: a safeguard to not choose a too long step size when initially increasing
* `stop_increasing_at_step=100`: stop the initial increasing loop after this amount of steps. Set to `0` to never increase in the beginning
* `stop_decreasing_at_step=1000`: maximal number of Armijo decreases / tests to perform
* `sufficient_decrease=0.1`: the sufficient decrease parameter ``τ``

For the stop safe guards you can pass `:Messages` to a `debug=` to see `@info` messages when these happen.

$(_note(:ManifoldDefaultFactory, "ArmijoLinesearchStepsize"))
"""
function ArmijoLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.ArmijoLinesearchStepsize, args...; kwargs...)
end

@doc """
    AdaptiveWNGradientStepsize{I<:Integer,R<:Real,F<:Function} <: Stepsize

A functor `problem, state, k, X) -> s to an adaptive gradient method introduced by [GrapigliaStella:2023](@cite).
See [`AdaptiveWNGradient`](@ref) for the mathematical details.

# Fields

* `count_threshold::I`: an `Integer` for ``$(_tex(:hat, "c"))``
* `minimal_bound::R`: the value for ``b_{$(_tex(:text, "min"))}``
* `alternate_bound::F`: how to determine ``$(_tex(:hat, "k"))_k`` as a function of `(bmin, bk, hat_c) -> hat_bk`
* `gradient_reduction::R`: the gradient reduction factor threshold ``α ∈ [0,1)``
* `gradient_bound::R`: the bound ``b_k``.
* `weight::R`: ``ω_k`` initialised to ``ω_0 = ```norm(M, p, X)` if this is not zero, `1.0` otherwise.
* `count::I`: ``c_k``, initialised to ``c_0 = 0``.

# Constructor

    AdaptiveWNGrad(M::AbstractManifold; kwargs...)

## Keyword arguments

* `adaptive=true`: switches the `gradient_reduction ``α`` (if `true`) to `0`.
* `alternate_bound = (bk, hat_c) ->  min(gradient_bound == 0 ? 1.0 : gradient_bound, max(minimal_bound, bk / (3 * hat_c))`
* `count_threshold=4`
* `gradient_reduction::R=adaptive ? 0.9 : 0.0`
* `gradient_bound=norm(M, p, X)`
* `minimal_bound=1e-4`
$(_var(:Keyword, :p; add = "only used to define the `gradient_bound`"))
$(_var(:Keyword, :X; add = "only used to define the `gradient_bound`"))
"""
mutable struct AdaptiveWNGradientStepsize{I <: Integer, R <: Real, F <: Function} <: Stepsize
    count_threshold::I
    minimal_bound::R
    alternate_bound::F
    gradient_reduction::R
    gradient_bound::R
    weight::R
    count::I
end
function AdaptiveWNGradientStepsize(
        M::AbstractManifold;
        p = rand(M),
        X = zero_vector(M, p),
        adaptive::Bool = true,
        count_threshold::I = 4,
        minimal_bound::R = 1.0e-4,
        gradient_reduction::R = adaptive ? 0.9 : 0.0,
        gradient_bound::R = norm(M, p, X),
        alternate_bound::F = (bk, hat_c) -> min(
            gradient_bound == 0 ? 1.0 : gradient_bound, max(minimal_bound, bk / (3 * hat_c))
        ),
        kwargs...,
    ) where {I <: Integer, R <: Real, F <: Function}
    if gradient_bound == 0
        # If the gradient bound defaults to zero, set it to 1
        gradient_bound = 1.0
    end
    return AdaptiveWNGradientStepsize{I, R, F}(
        count_threshold,
        minimal_bound,
        alternate_bound,
        gradient_reduction,
        gradient_bound,
        gradient_bound,
        0,
    )
end
function (awng::AdaptiveWNGradientStepsize)(
        mp::AbstractManoptProblem,
        s::AbstractGradientSolverState,
        i,
        args...;
        gradient = nothing,
        kwargs...,
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
function show(io::IO, awng::AdaptiveWNGradientStepsize)
    s = """
    AdaptiveWNGradient(;
      count_threshold = $(awng.count_threshold),
      minimal_bound = $(awng.minimal_bound),
      alternate_bound = $(awng.alternate_bound),
      gradient_reduction = $(awng.gradient_reduction),
      gradient_bound = $(awng.gradient_bound)
    )

    as well as internally the weight ω_k = $(awng.weight) and current count c_k = $(awng.count).
    """
    return print(io, s)
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
$(_var(:Keyword, :p; add = "only used to define the `gradient_bound`"))
$(_var(:Keyword, :X; add = "only used to define the `gradient_bound`"))
"""
function AdaptiveWNGradient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.AdaptiveWNGradientStepsize, args...; kwargs...)
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
        s=min(1.0, injectivity_radius(M)/2);
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
function show(io::IO, cs::ConstantStepsize)
    return print(io, "ConstantLength($(cs.length); type=:$(cs.type))")
end

"""
    ConstantLength(s; kwargs...)
    ConstantLength(M::AbstractManifold, s; kwargs...)

Specify a [`Stepsize`](@ref) that is constant.

# Input

* `M` (optional)
* `s=min( injectivity_radius(M)/2, 1.0)` : the length to use.

# Keyword argument

* `type::Symbol=:relative` specify the type of constant step size. Possible values are
  * `:relative` – scale the gradient tangent vector ``X`` to ``s*X``
  * `:absolute` – scale the gradient to an absolute step length ``s``, that is ``$(_tex(:frac, "s", _tex(:norm, "X")))X``

$(_note(:ManifoldDefaultFactory, "ConstantStepsize"))
"""
function ConstantLength(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.ConstantStepsize, args...; kwargs...)
end

@doc """
    CubicBracketingLinesearchStepsize{P,T,R<:Real} <: Linesearch

Do a bracketing line search to find a step size ``α`` that finds a
local minimum along the  search direction ``X`` starting from ``p``,
utilizing cubic polynomial interpolation.
See [`CubicBracketingLinesearch`](@ref) for the mathematical details.

# Fields
$(_var(:Field, :p, "candidate_point"; add = " as temporary storage for candidates"))
* `initial_stepsize::R`: the step size to start the search with
* `last_stepsize::R`
$(_var(:Field, :retraction_method))
* `stepsize_increase::R`:  step size increase factor ``>1``
* `max_iterations::I`: maximum number of iterations
* `sufficient_curvature::R`: target reduction of the curvature ``(0,1)``
* `min_bracket_width::R`: minimal size of the bracket ``[a,b]``
* `hybrid::Bool`: use the hybrid strategy
* `max_stepsize::R`: maximal stepsize
$(_var(:Field, :vector_transport_method))

# Constructor

    CubicBracketingLinesearchStepsize(M::AbstractManifold; kwargs...)

## Keyword arguments

$(_var(:Keyword, :p, "candidate_point"; add = " as temporary storage for candidates"))
* `initial_stepsize=1.0`: the step size to start the search with
$(_var(:Keyword, :retraction_method))
* `stepsize_increase=1.1`:  step size increase factor ``>1``
* `max_iterations=100`: maximum number of iterations
* `sufficient_curvature=0.2`: target reduction of the curvature ``(0,1)``
* `min_bracket_width=1e-4`: minimal size of the bracket ``[a,b]``
* `hybrid=true`: use the hybrid strategy
* `max_stepsize= max_stepsize(M)`: maximal stepsize
$(_var(:Keyword, :vector_transport_method))
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
        return new{R, I, TRM, VTM, P, T}(candidate_direction, candidate_point, initial_stepsize, initial_stepsize, retraction_method, stepsize_increase, max_iterations, sufficient_curvature, min_bracket_width, hybrid, vector_transport_method, max_stepsize)
    end
end

"""
    UnivariateTriple{R <: Real}

Triple of stepsize, function value und derivative value

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
* `b::UnivariateTriple{R}`: triple bracket value `b`
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
* `b::UnivariateTriple{R}`: triple bracket value `b`

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

Returns the extremal of the quadratic polynomial ``p`` with
``p'(a.t)=a.df``, ``p'(b.t)=b.df``.

# Input
* `a::UnivariateTriple{R}`: triple of bracket value `a`
* `b::UnivariateTriple{R}`: triple bracket value `b`
"""
function secant(a::UnivariateTriple{R}, b::UnivariateTriple{R}) where {R}
    return (a.t * b.df - b.t * a.df) / (b.df - a.df)
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
Get the `UnivariateTriple` of the problem `mp` related to the step with
stepsize ``t`` from ``p`` in direction ``η``.

# Input
* `mp::AbstractManoptProblem`
* `cbls:::CubicBracketingLinesearchStepsize`: containing `retraction_method`, `vector_transport` and the temporary `candidate_point` and `candidate_direction`
* `p`: point in the manifold of `mp`
* `η`: search direction at `p`
* `t::Real`: step size
"""
function get_univariate_triple!(mp::AbstractManoptProblem, cbls::CubicBracketingLinesearchStepsize, p, η, t)
    M = get_manifold(mp)
    cbls.last_stepsize = t
    ManifoldsBase.retract_fused!(M, cbls.candidate_point, p, η, t, cbls.retraction_method)
    vector_transport_to!(M, cbls.candidate_direction, p, η, cbls.candidate_point, cbls.vector_transport_method)
    f, df = get_cost_and_differential(mp, cbls.candidate_point, cbls.candidate_direction)
    return UnivariateTriple(t, f, df)
end

function (cbls::CubicBracketingLinesearchStepsize)(
        mp::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        k::Int,
        η = (-get_gradient(mp, get_iterate(s)));
        kwargs...,
    )
    M = get_manifold(mp)
    p = get_iterate(s)

    init = UnivariateTriple(0.0, get_cost(M, get_objective(mp), p), get_differential(mp, p, η; gradient = s.X, evaluated = true))

    check_curvature(c::UnivariateTriple) = abs(c.df) < cbls.sufficient_curvature * abs(init.df)

    n_iter = 0
    t = cbls.last_stepsize
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
        (t == cbls.max_stepsize) && return t
        t *= cbls.stepsize_increase
        t = min(t, cbls.max_stepsize)
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
function show(io::IO, cbls::CubicBracketingLinesearchStepsize)
    return print(
        io,
        """
        CubicBracketingLinesearch(;
            initial_stepsize = $(cbls.initial_stepsize),
            stepsize_increase = $(cbls.stepsize_increase),
            sufficient_curvature = $(cbls.sufficient_curvature),
            min_bracket_width = $(cbls.min_bracket_width),
            hybrid = $(cbls.hybrid),
            retraction_method = $(cbls.retraction_method),
            vector_transport_method = $(cbls.vector_transport_method),
            max_stepsize = $(cbls.max_stepsize)
        )""",
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

is satisfied by either ``[a,b] = [t_{k-1},t_k]``, ``[a,b] = [t_k,t_{k-1}]``, ``[a,b] = [0,t_k]``, or ``[a,b] = [t_k,0]``.
Here, ``ϕ(t)`` denotes the cost function when performing
a step with size ``t`` into direction ``η``.
Over the iteration, the bracket ``[a,b]`` is repeatedly
updated using a cubic polynomial using values of ``ϕ, ϕ'`` at ``a,b``.
The update value ``c`` is the local minimum of the polynomial, and the bracket coindition
ensures that it lies inbetween ``a`` and ``b``. We note that the update strategy taken from
[Hager:1989](@cite) ensures that the updated bracket satsifies the bracket condtion.

If the parameter `hybrid` is set to `true`, the hybrid approach from [Hager:1989](@cite)
is activated, which prevents slow convergence in edge cases.

The algorithm terminates if at any point the found candidate stepsize suffices the curvature condition
induced by `sufficient_curvature``
or the bracket ``[a,b]`` is smaller than the `min_bracket_width`.

# Keyword arguments

$(_var(:Keyword, :p; add = "to store an interim result"))
* `p=`[`allocate_result`](@extref ManifoldsBase.allocate_result)`(M, rand)`: to store an interim result
* `initial_stepsize=1.0`: the step size to start the search with
$(_var(:Keyword, :retraction_method))
* `stepsize_increase=1.1`:  step size increase factor ``>1``
* `max_iterations=100`: maximum number of iterations
* `sufficient_curvature=0.2`: target reduction of the curvature ``(0,1)``
* `min_bracket_width=1e-4`: minimal size of the bracket ``[a,b]``
* `hybrid=true`: use the hybrid strategy
$(_var(:Keyword, :vector_transport_method))

$(_note(:ManifoldDefaultFactory, "CubicBracketingLinesearch"))
"""
function CubicBracketingLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(CubicBracketingLinesearchStepsize, args...; kwargs...)
end


@doc """
    DecreasingStepsize()

A functor `(problem, state, ...) -> s` to provide a constant step size `s`.

# Fields

* `exponent`:   a value ``e`` the current iteration numbers ``e``th exponential is
  taken of
* `factor`:     a value ``f`` to multiply the initial step size with every iteration
* `length`:     the initial step size ``l``.
* `subtrahend`: a value ``a`` that is subtracted every iteration
* `shift`:      shift the denominator iterator ``i`` by ``s```.
* `type`:       a symbol that indicates whether the stepsize is relatively (:relative),
    with respect to the gradient norm, or absolutely (:absolute) constant.

In total the complete formulae reads for the ``i``th iterate as

```math
s_i = $(_tex(:frac, "(l - i a)f^i", "(i+s)^e"))
```

and hence the default simplifies to just ``s_i = \frac{l}{i}``

# Constructor

    DecreasingStepsize(M::AbstractManifold;
        length=min(injectivity_radius(M)/2, 1.0),
        factor=1.0,
        subtrahend=0.0,
        exponent=1.0,
        shift=0.0,
        type=:relative,
    )

initializes all fields, where none of them is mandatory and the length is set to
half and to ``1`` if the injectivity radius is infinite.
"""
mutable struct DecreasingStepsize{R <: Real} <: Stepsize
    length::R
    factor::R
    subtrahend::R
    exponent::R
    shift::R
    type::Symbol
end
function DecreasingStepsize(
        M::AbstractManifold;
        length::R = isinf(manifold_dimension(M)) ? 1.0 : manifold_dimension(M) / 2,
        factor::R = 1.0,
        subtrahend::R = 0.0,
        exponent::R = 1.0,
        shift::R = 0.0,
        type::Symbol = :relative,
    ) where {R}
    return DecreasingStepsize(length, factor, subtrahend, exponent, shift, type)
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
function show(io::IO, s::DecreasingStepsize)
    return print(
        io,
        "DecreasingLength(; length=$(s.length),  factor=$(s.factor),  subtrahend=$(s.subtrahend),  shift=$(s.shift), type=$(s.type))",
    )
end
"""
    DegreasingLength(; kwargs...)
    DecreasingLength(M::AbstractManifold; kwargs...)

Specify a [`Stepsize`]  that is decreasing as ``s_k = $(_tex(:frac, "(l - ak)f^i", "(k+s)^e"))
with the following

# Keyword arguments

* `exponent=1.0`:   the exponent ``e`` in the denominator
* `factor=1.0`:     the factor ``f`` in the nominator
* `length=min(injectivity_radius(M)/2, 1.0)`: the initial step size ``l``.
* `subtrahend=0.0`: a value ``a`` that is subtracted every iteration
* `shift=0.0`:      shift the denominator iterator ``k`` by ``s``.
* `type::Symbol=relative` specify the type of constant step size.
* `:relative` – scale the gradient tangent vector ``X`` to ``s_k*X``
* `:absolute` – scale the gradient to an absolute step length ``s_k``, that is ``$(_tex(:frac, "s_k", _tex(:norm, "X")))X``

$(_note(:ManifoldDefaultFactory, "DecreasingStepsize"))
"""
function DecreasingLength(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.DecreasingStepsize, args...; kwargs...)
end

@doc raw"""
    DistanceOverGradientsStepsize{R<:Real} <: Stepsize

# Fields

* `initial_distance::R`: initial distance estimate ``ϵ>0``
* `max_distance::R`: tracked maximum distance ``\bar r_t``
* `gradient_sum::R`: accumulated sum ``G_t``
* `initial_point`: stored start point ``p_0``
* `use_curvature::Bool`: toggle curvature correction ``ζ_κ``
* `sectional_curvature_bound::R`: lower bound ``κ`` used in ``ζ_κ`` when `use_curvature=true`
* `last_stepsize::R`: last computed stepsize

# Constructor

    DistanceOverGradientsStepsize(M::AbstractManifold; kwargs...)

## Keyword arguments

* `initial_distance=1e-3`: initial estimate ``ϵ``
* `use_curvature=false`: whether to use ``ζ_κ``
* `sectional_curvature_bound=0.0`: lower curvature bound ``κ`` (if known)
* `p`: initial point, used to track distance

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
end

function DistanceOverGradientsStepsize(
        M::AbstractManifold;
        p = rand(M),
        initial_distance::R1 = 1.0e-3,
        use_curvature::Bool = false,
        sectional_curvature_bound::R2 = 0.0,
    ) where {R1 <: Real, R2 <: Real}
    R = promote_type(R1, R2)
    id = convert(R, initial_distance)
    κ = convert(R, sectional_curvature_bound)
    return DistanceOverGradientsStepsize{R, typeof(p)}(
        id,
        id,  # max_distance starts at initial_distance
        zero(R),          # gradient_sum starts at 0
        copy(M, p),       # store initial point
        use_curvature,
        κ,
        NaN,              # last_stepsize
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
        mp::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        i,
        args...;
        gradient = nothing,
        kwargs...,
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

function show(io::IO, rdog::DistanceOverGradientsStepsize)
    s = """
    DistanceOverGradients(;
      initial_distance = $(rdog.initial_distance),
      use_curvature = $(rdog.use_curvature),
      sectional_curvature_bound = $(rdog.sectional_curvature_bound)
    )

    Current state:
      max_distance = $(rdog.max_distance)
      gradient_sum = $(rdog.gradient_sum)
      last_stepsize = $(rdog.last_stepsize)
    """
    return print(io, s)
end

doc_DoG_main = raw"""
    DistanceOverGradients(; kwargs...)
    DistanceOverGradients(M::AbstractManifold; kwargs...)

Create a factory for the [`DistanceOverGradientsStepsize`](@ref), the
Riemannian Distance over Gradients (RDoG) learning-rate-free stepsize from
[DoddSharrockNemeth:2024](@cite). It adapts via the maximum distance from the
start point and the accumulated gradient norms, optionally corrected by the
geometric curvature term ``ζ_κ``. It adapts without manual
tuning by combining a distance proxy from the start point with accumulated
gradient norms.

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

$(_note(:ManifoldDefaultFactory, "DistanceOverGradientsStepsize"))
"""
function DistanceOverGradients(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.DistanceOverGradientsStepsize, args...; kwargs...)
end

@doc """
    NonmonotoneLinesearchStepsize{P,T,R<:Real} <: Linesearch

A functor representing a nonmonotone line search using the Barzilai-Borwein step size [IannazzoPorcelli:2017](@cite).

# Fields

* $(_doc_stepsize_initial_guess)
* `memory_size`:           number of iterations after which the cost value needs to be lower than the current one
* `bb_min_stepsize=1e-3`:     lower bound for the Barzilai-Borwein step size greater than zero
* `bb_max_stepsize=1e3`:      upper bound for the Barzilai-Borwein step size greater than min_stepsize
* `last_stepsize`:     the last computed stepsize
$(_var(:Keyword, :retraction_method))
* `strategy=direct`:          defines if the new step size is computed using the `:direct`, `:indirect` or `:alternating` strategy
* `storage`:                  (for `:Iterate` and `:Gradient`) a [`StoreStateAction`](@ref)
* `stepsize_reduction`:       step size reduction factor contained in the interval (0,1)
* `sufficient_decrease`:     sufficient decrease parameter contained in the interval (0,1)
$(_var(:Keyword, :vector_transport_method))
* `candidate_point`:          to store an interim result
* `stop_when_stepsize_less`:    smallest stepsize when to stop (the last one before is taken)
* `stop_when_stepsize_exceeds`: largest stepsize when to stop.
* `stop_increasing_at_step`:    last step to increase the stepsize (phase 1),
* `stop_decreasing_at_step`:    last step size to decrease the stepsize (phase 2),

# Constructor

    NonmonotoneLinesearchStepsize(M::AbstractManifold; kwargs...)

## Keyword arguments

* `p=allocate_result(M, rand)`: to store an interim result
* `initial_guess = (problem, state, k, last_stepsize, η) -> k == 0 ? 1.0 : last_stepsize`
   function to provide an initial guess for the stepsize
* `memory_size=10`
* `bb_min_stepsize=1e-3`
* `bb_max_stepsize=1e3`
$(_var(:Keyword, :retraction_method))
* `strategy=direct`
* `storage=`[`StoreStateAction`](@ref)`(M; store_fields=[:Iterate, :Gradient])`
* `stepsize_reduction=0.5`
* `sufficient_decrease=1e-4`
* `stop_when_stepsize_less=0.0`
* `stop_when_stepsize_exceeds=`[`max_stepsize`](@ref)`(M, p)`)
* `stop_increasing_at_step=100`
* `stop_decreasing_at_step=1000`
$(_var(:Keyword, :vector_transport_method))
"""
mutable struct NonmonotoneLinesearchStepsize{
        P,
        T <: AbstractVector,
        R <: Real,
        I <: Integer,
        TRM <: AbstractRetractionMethod,
        VTM <: AbstractVectorTransportMethod,
        TSSA <: StoreStateAction,
        MSGS <: NamedTuple,
        IG,
    } <: Linesearch
    bb_min_stepsize::R
    bb_max_stepsize::R
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
    storage::TSSA
    strategy::Symbol
    sufficient_decrease::R
    vector_transport_method::VTM
    function NonmonotoneLinesearchStepsize(
            M::AbstractManifold;
            bb_min_stepsize::R = 1.0e-3,
            bb_max_stepsize::R = 1.0e3,
            p::P = allocate_result(M, rand),
            initial_guess::IG = (problem, state, k, last_stepsize, η) -> k == 0 ? 1.0 : last_stepsize,
            memory_size::I = 10,
            retraction_method::TRM = default_retraction_method(M),
            stepsize_reduction::R = 0.5,
            stop_when_stepsize_less::R = 0.0,
            stop_when_stepsize_exceeds = real(max_stepsize(M)),
            stop_increasing_at_step::I = 100,
            stop_decreasing_at_step::I = 1000,
            storage::Union{Nothing, StoreStateAction} = StoreStateAction(
                M; store_fields = [:Iterate, :Gradient]
            ),
            strategy::Symbol = :direct,
            sufficient_decrease::R = 1.0e-4,
            vector_transport_method::VTM = default_vector_transport_method(M),
        ) where {TRM, VTM, P, R <: Real, I <: Integer, IG}
        stop_when_stepsize_exceeds = R(stop_when_stepsize_exceeds)
        if strategy ∉ [:direct, :inverse, :alternating]
            @warn string(
                "The strategy '", strategy, "' is not defined. The 'direct' strategy is used instead.",
            )
            strategy = :direct
        end
        if bb_min_stepsize <= 0.0
            throw(
                DomainError(
                    bb_min_stepsize, "The lower bound for the step size min_stepsize has to be greater than zero.",
                ),
            )
        end
        if bb_max_stepsize <= bb_min_stepsize
            throw(
                DomainError(
                    bb_max_stepsize, "The upper bound for the step size max_stepsize has to be greater its lower bound min_stepsize.",
                ),
            )
        end
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
        return new{P, typeof(old_costs), R, I, TRM, VTM, typeof(storage), typeof(msgs), IG}(
            bb_min_stepsize,
            bb_max_stepsize,
            p,
            initial_guess,
            1.0,
            msgs,
            old_costs,
            retraction_method,
            stepsize_reduction,
            stop_decreasing_at_step,
            stop_increasing_at_step,
            stop_when_stepsize_exceeds,
            stop_when_stepsize_less,
            storage,
            strategy,
            sufficient_decrease,
            vector_transport_method,
        )
    end
end
function (a::NonmonotoneLinesearchStepsize)(
        mp::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        k::Int,
        η = (-get_gradient(mp, get_iterate(s)));
        gradient = nothing,
        kwargs...,
    )
    grad = isnothing(gradient) ? get_gradient(mp, get_iterate(s)) : gradient
    if !has_storage(a.storage, PointStorageKey(:Iterate)) ||
            !has_storage(a.storage, VectorStorageKey(:Gradient))
        # first time call: get old grad/iterate and store.
        p_old = get_iterate(s)
        X_old = grad
    else
        #fetch
        p_old = get_storage(a.storage, PointStorageKey(:Iterate))
        X_old = get_storage(a.storage, VectorStorageKey(:Gradient))
    end
    update_storage!(a.storage, mp, s)
    return a(
        get_manifold(mp),
        get_iterate(s),
        (M, p) -> get_cost(M, get_objective(mp), p),
        grad,
        η,
        p_old,
        X_old,
        k,
    )
end
function (a::NonmonotoneLinesearchStepsize)(
        M::mT, p, f::TF, X::T, η::T, old_p, old_X, iter::Int; kwargs...
    ) where {mT <: AbstractManifold, TF, T}
    reset_messages!(a.messages)
    #find the difference between the current and previous gradient after the previous gradient is transported to the current tangent space
    grad_diff = X - vector_transport_to(M, old_p, old_X, p, a.vector_transport_method)
    #transport the previous step into the tangent space of the current manifold point

    initial_stepsize = a.initial_guess(M, p, iter, a.last_stepsize, η)

    x_diff =
        -initial_stepsize *
        vector_transport_to(M, old_p, old_X, p, a.vector_transport_method)

    #compute the new Barzilai-Borwein step size
    s1 = real(inner(M, p, x_diff, grad_diff))
    s2 = real(inner(M, p, grad_diff, grad_diff))
    s2 = s2 == 0 ? 1.0 : s2
    s3 = real(inner(M, p, x_diff, x_diff))
    #indirect strategy
    if a.strategy == :inverse
        if s1 > 0
            BarzilaiBorwein_stepsize = min(
                a.bb_max_stepsize, max(a.bb_min_stepsize, s1 / s2)
            )
        else
            BarzilaiBorwein_stepsize = a.bb_max_stepsize
        end
        #alternating strategy
    elseif a.strategy == :alternating
        if s1 > 0
            if iter % 2 == 0
                BarzilaiBorwein_stepsize = min(
                    a.bb_max_stepsize, max(a.bb_min_stepsize, s1 / s2)
                )
            else
                BarzilaiBorwein_stepsize = min(
                    a.bb_max_stepsize, max(a.bb_min_stepsize, s3 / s1)
                )
            end
        else
            BarzilaiBorwein_stepsize = a.bb_max_stepsize
        end
        #direct strategy
    else
        if s1 > 0
            BarzilaiBorwein_stepsize = min(
                a.bb_max_stepsize, max(a.bb_min_stepsize, s2 / s1)
            )
        else
            BarzilaiBorwein_stepsize = a.bb_max_stepsize
        end
    end

    memory_size = length(a.old_costs)
    if iter <= memory_size
        a.old_costs[iter] = f(M, p)
    else
        a.old_costs[1:(memory_size - 1)] = a.old_costs[2:memory_size]
        a.old_costs[memory_size] = f(M, p)
    end

    #compute the new step size with the help of the Barzilai-Borwein step size
    a.last_stepsize = linesearch_backtrack!(
        M,
        a.candidate_point,
        f,
        p,
        BarzilaiBorwein_stepsize,
        a.sufficient_decrease,
        a.stepsize_reduction,
        η;
        lf0 = maximum([a.old_costs[j] for j in 1:min(iter, memory_size)]),
        gradient = X,
        retraction_method = a.retraction_method,
        stop_when_stepsize_less = (a.stop_when_stepsize_less / norm(M, p, η)),
        stop_when_stepsize_exceeds = (a.stop_when_stepsize_exceeds / norm(M, p, η)),
        stop_increasing_at_step = a.stop_increasing_at_step,
        stop_decreasing_at_step = a.stop_decreasing_at_step,
        report_messages_in = a.messages,
    )
    return a.last_stepsize
end
function show(io::IO, a::NonmonotoneLinesearchStepsize)
    return print(
        io,
        """
        NonmonotoneLinesearch(;
            last_stepsize = $(a.last_stepsize),
            bb_max_stepsize = $(a.bb_max_stepsize),
            bb_min_stepsize = $(a.bb_min_stepsize),
            memory_size = $(length(a.old_costs)),
            stepsize_reduction = $(a.stepsize_reduction),
            strategy = :$(a.strategy),
            sufficient_decrease = $(a.sufficient_decrease),
            retraction_method = $(a.retraction_method),
            vector_transport_method = $(a.vector_transport_method)
        )""",
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

This method first computes

(x -> p, F-> f)
```math
y_{k} = $(_tex(:grad))f(p_{k}) - $(_math(:vector_transport, :symbol, "p_k", "p_{k-1}"))$(_tex(:grad))f(p_{k-1})
```

and
```math
s_{k} = - α_{k-1} ⋅ $(_math(:vector_transport, :symbol, "p_k", "p_{k-1}"))$(_tex(:grad))f(p_{k-1}),
```

where ``α_{k-1}`` is the step size computed in the last iteration and ``$(_math(:vector_transport, :symbol))`` is a vector transport.
Then the Barzilai—Borwein step size is

```math
α_k^{$(_tex(:text, "BB"))} = $(
    _tex(
        :cases,
        "$(_tex(:min))(α_{$(_tex(:text, "max"))}, $(_tex(:max))(α_{$(_tex(:text, "min"))}, τ_{k})), & $(_tex(:text, "if")) ⟨s_{k}, y_{k}⟩_{p_k} > 0,",
        "α_{$(_tex(:text, "max"))}, & $(_tex(:text, "else,"))"
    )
)
```

where

```math
τ_{k} = $(_tex(:frac, "⟨s_{k}, s_{k}⟩_{p_k}", "⟨s_{k}, y_{k}⟩_{p_k}")),
```

if the direct strategy is chosen, or

```math
τ_{k} =  $(_tex(:frac, "⟨s_{k}, y_{k}⟩_{p_k}", "⟨y_{k}, y_{k}⟩_{p_k}")),
```

in case of the inverse strategy or an alternation between the two in cases for
the alternating strategy. Then find the smallest ``h = 0, 1, 2, …`` such that

```math
f($(_tex(:retr))_{p_k}(- σ^h α_k^{$(_tex(:text, "BB"))} $(_tex(:grad))f(p_k)))  ≤
$(_tex(:max))_{1 ≤ j ≤ $(_tex(:max))(k+1,m)} f(p_{k+1-j}) - γ σ^h α_k^{$(_tex(:text, "BB"))} ⟨$(_tex(:grad))F(p_k), $(_tex(:grad))F(p_k)⟩_{p_k},
```

where ``σ ∈ (0,1)`` is a step length reduction factor , ``m`` is the number of iterations
after which the function value has to be lower than the current one
and ``γ ∈ (0,1)`` is the sufficient decrease parameter. Finally the step size is computed as

```math
α_k = σ^h α_k^{$(_tex(:text, "BB"))}.
```

# Keyword arguments

$(_var(:Keyword, :p; add = "to store an interim result"))
* `p=allocate_result(M, rand)`: to store an interim result
* `initial_stepsize=1.0`: the step size to start the search with
* `memory_size=10`: number of iterations after which the cost value needs to be lower than the current one
* `bb_min_stepsize=1e-3`: lower bound for the Barzilai-Borwein step size greater than zero
* `bb_max_stepsize=1e3`: upper bound for the Barzilai-Borwein step size greater than min_stepsize
$(_var(:Keyword, :retraction_method))
* `strategy=direct`: defines if the new step size is computed using the `:direct`, `:indirect` or `:alternating` strategy
* `storage=`[`StoreStateAction`](@ref)`(M; store_fields=[:Iterate, :Gradient])`: increase efficiency by using a [`StoreStateAction`](@ref) for `:Iterate` and `:Gradient`.
* `stepsize_reduction=0.5`:  step size reduction factor contained in the interval ``(0,1)``
* `sufficient_decrease=1e-4`: sufficient decrease parameter contained in the interval ``(0,1)``
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
* `stop_when_stepsize_exceeds=`[`max_stepsize`](@ref)`(M, p)`): largest stepsize when to stop to avoid leaving the injectivity radius
* `stop_increasing_at_step=100`:  last step to increase the stepsize (phase 1),
* `stop_decreasing_at_step=1000`: last step size to decrease the stepsize (phase 2),
"""
function NonmonotoneLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(NonmonotoneLinesearchStepsize, args...; kwargs...)
end

@doc """
    PolyakStepsize <: Stepsize

A functor `(problem, state, ...) -> s` to provide a step size due to Polyak, cf. Section 3.2 of [Bertsekas:2015](@cite).

# Fields

* `γ`               : a function `k -> ...` representing a seuqnce.
* `best_cost_value` : storing the best cost value

# Constructor

    PolyakStepsize(;
        γ = i -> 1/i,
        initial_cost_estimate=0.0
    )

Construct a stepsize of Polyak type.

# See also
[`Polyak`](@ref)
"""
mutable struct PolyakStepsize{F, R} <: Stepsize
    γ::F
    best_cost_value::R
end
function PolyakStepsize(; γ::F = (i) -> 1 / i, initial_cost_estimate::R = 0.0) where {F, R}
    return PolyakStepsize{F, R}(γ, initial_cost_estimate)
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
function show(io::IO, ps::PolyakStepsize)
    return print(
        io,
        """
        Polyak()
        A stepsize with keyword parameters
           * initial_cost_estimate = $(ps.best_cost_value)
        """,
    )
end
"""
    Polyak(; kwargs...)
    Polyak(M::AbstractManifold; kwargs...)

Compute a step size according to a method propsed by Polyak, cf. the Dynamic step size
discussed in Section 3.2 of [Bertsekas:2015](@cite).
This has been generalised here to both the Riemannian case and to approximate the minimum cost value.

Let ``f_{$(_tex(:text, "best"))`` be the best cost value seen until now during some iterative
optimisation algorithm and let ``γ_k`` be a sequence of numbers that is square summable, but not summable.

Then the step size computed here reads

```math
s_k = $(_tex(:frac, "f(p^{(k)}) - f_{$(_tex(:text, "best")) + γ_k", _tex(:norm, "∂f(p^{(k)})}"))),
```

where ``∂f`` denotes a nonzero-subgradient of ``f`` at the current iterate ``p^{(k)}``.


# Constructor

    Polyak(; γ = k -> 1/k, initial_cost_estimate=0.0)

initialize the Polyak stepsize to a certain sequence and an initial estimate of ``f_{\text{best}}``.

$(_note(:ManifoldDefaultFactory, "PolyakStepsize"))
"""
function Polyak(args...; kwargs...)
    return ManifoldDefaultsFactory(
        Manopt.PolyakStepsize, args...; requires_manifold = false, kwargs...
    )
end

@doc """
    WolfePowellLinesearchStepsize{R<:Real} <: Linesearch

Do a backtracking line search to find a step size ``α`` that fulfils the
Wolfe conditions along a search direction ``X`` starting from ``p``.
See [`WolfePowellLinesearch`](@ref) for the math details

# Fields

* `sufficient_decrease::R`, `sufficient_curvature::R` two constants in the line search
$(_var(:Field, :X, "candidate_direction"))
$(_var(:Field, :p, "candidate_point"; add = "as temporary storage for candidates"))
$(_var(:Field, :X, "candidate_tangent"))
* `last_stepsize::R`
* `max_stepsize::R`
$(_var(:Field, :retraction_method))
* `stop_when_stepsize_less::R`: a safeguard to stop when the stepsize gets too small
$(_var(:Field, :vector_transport_method))

# Constructor

    WolfePowellLinesearchStepsize(M::AbstractManifold; kwargs...)

## Keyword arguments

* `sufficient_decrease=10^(-4)`
* `sufficient_curvature=0.999`
$(_var(:Keyword, :p; add = "as temporary storage for candidates"))
$(_var(:Keyword, :X; add = "as type of memory allocated for the candidates direction and tangent"))
* `max_stepsize=`[`max_stepsize`](@ref)`(M, p)`: largest stepsize allowed here.
$(_var(:Keyword, :retraction_method))
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
* `stop_increasing_at_step=100`: for the initial increase test (s_plus), stop after these many steps
* `stop_decreasing_at_step=1000`: for the initial decrease test (s_minus), stop after these many steps
$(_var(:Keyword, :vector_transport_method))
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
    stop_increasing_at_step::Int
    stop_decreasing_at_step::Int
    messages::TMSG
    function WolfePowellLinesearchStepsize(
            M::AbstractManifold;
            p::P = allocate_result(M, rand),
            X::T = zero_vector(M, p),
            max_stepsize::Real = max_stepsize(M),
            retraction_method::TRM = default_retraction_method(M),
            sufficient_decrease::R = 1.0e-4,
            sufficient_curvature::R = 0.999,
            vector_transport_method::VTM = default_vector_transport_method(M),
            stop_when_stepsize_less::R = 0.0,
            stop_increasing_at_step::I = 100,
            stop_decreasing_at_step::I = 1000,
        ) where {TRM, VTM, P, T, R, I}
        msgs = (;
            non_descent_direction = StepsizeMessage{R, R}(),
            stop_decreasing = StepsizeMessage{Int, R}(),
            stop_increasing = StepsizeMessage{Int, R}(),
            stepsize_less = StepsizeMessage{R, R}(),
            stepsize_exceeds = StepsizeMessage{R, R}(),
        )
        return new{R, TRM, VTM, P, T, I, typeof(msgs)}(
            sufficient_decrease,
            sufficient_curvature,
            X,
            p,
            0.0,
            max_stepsize,
            retraction_method,
            stop_when_stepsize_less,
            vector_transport_method,
            stop_increasing_at_step,
            stop_decreasing_at_step,
            msgs,
        )
    end
end
function (a::WolfePowellLinesearchStepsize)(
        mp::AbstractManoptProblem,
        ams::AbstractManoptSolverState,
        k::Int,
        η = (-get_gradient(mp, get_iterate(ams)));
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
    step = ifelse(isfinite(a.max_stepsize), min(1.0, a.max_stepsize / grad_norm), 1.0)
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
        s_plus = 2.0 * s_minus
    else
        vector_transport_to!(M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method)
        if get_differential(mp, a.candidate_point, a.candidate_direction; Y = Y) < a.sufficient_curvature * l
            i = 0
            while fNew <= f0 + a.sufficient_decrease * step * l && (s_plus < max_step_increase)
                # increase
                s_plus = s_plus * 2.0
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
function show(io::IO, a::WolfePowellLinesearchStepsize)
    return print(
        io,
        """
        WolfePowellLinesearch(;
            sufficient_descrease = $(a.sufficient_decrease),
            sufficient_curvature = $(a.sufficient_curvature),
            retraction_method = $(a.retraction_method),
            vector_transport_method = $(a.vector_transport_method),
            stop_when_stepsize_less = $(a.stop_when_stepsize_less),
            stop_increasing_at_step = $(a.stop_increasing_at_step),
            stop_decreasing_at_step = $(a.stop_decreasing_at_step),
        )""",
    )
end
function status_summary(a::WolfePowellLinesearchStepsize)
    s = (a.last_stepsize > 0) ? "\nand the last stepsize used was $(a.last_stepsize)." : ""
    return "$a$s"
end
function get_message(a::WolfePowellLinesearchStepsize)
    s = [get_message(kv[1], kv[2]) for kv in pairs(a.messages)]
    return join([m for m in s if length(m) > 0], "\n")
end
"""
    WolfePowellLinesearch(; kwargs...)
    WolfePowellLinesearch(M::AbstractManifold; kwargs...)

Perform a lineseach to fulfull both the Armijo-Goldstein conditions
```math
f$(_tex(:bigl))( $(_tex(:retr))_{p}(αX) $(_tex(:bigr))) ≤ f(p) + c_1 α_k ⟨$(_tex(:grad)) f(p), X⟩_{p}
```

as well as the Wolfe conditions

```math
$(_tex(:deriv)) f$(_tex(:bigl))($(_tex(:retr))_{p}(tX)$(_tex(:bigr)))
$(_tex(:Big))$(_tex(:vert))_{t=α}
≥ c_2 $(_tex(:deriv)) f$(_tex(:bigl))($(_tex(:retr))_{p}(tX)$(_tex(:bigr)))$(_tex(:Big))$(_tex(:vert))_{t=0}.
```

for some given sufficient decrease coefficient ``c_1`` and some sufficient curvature condition coefficient``c_2``.

This is adopted from [NocedalWright:2006; Section 3.1](@cite)

# Keyword arguments

* `sufficient_decrease=10^(-4)`
* `sufficient_curvature=0.999`
$(_var(:Keyword, :p; add = "as temporary storage for candidates"))
$(_var(:Keyword, :X; add = "as type of memory allocated for the candidates direction and tangent"))
* `max_stepsize=`[`max_stepsize`](@ref)`(M, p)`: largest stepsize allowed here.
$(_var(:Keyword, :retraction_method))
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
* `stop_increasing_at_step=100`: for the initial increase test (s_plus), stop after these many steps
* `stop_decreasing_at_step=1000`: for the initial decrease test (s_minus), stop after these many steps
$(_var(:Keyword, :vector_transport_method))
"""
function WolfePowellLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(WolfePowellLinesearchStepsize, args...; kwargs...)
end

@doc """
    WolfePowellBinaryLinesearchStepsize{R} <: Linesearch

Do a backtracking line search to find a step size ``α`` that fulfils the
Wolfe conditions along a search direction ``X`` starting from ``p``.
See [`WolfePowellBinaryLinesearch`](@ref) for the math details.

# Fields

* `sufficient_decrease::R`, `sufficient_curvature::R` two constants in the line search
* `last_stepsize::R`
* `max_stepsize::R`
$(_var(:Field, :retraction_method))
* `stop_when_stepsize_less::R`: a safeguard to stop when the stepsize gets too small
$(_var(:Field, :vector_transport_method))

# Constructor

    WolfePowellBinaryLinesearchStepsize(M::AbstractManifold; kwargs...)

## Keyword arguments

* `sufficient_decrease=10^(-4)`
* `sufficient_curvature=0.999`
* `max_stepsize=`[`max_stepsize`](@ref)`(M, p)`: largest stepsize allowed here.
$(_var(:Keyword, :retraction_method))
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
$(_var(:Keyword, :vector_transport_method))

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
            M::AbstractManifold = DefaultManifold();
            sufficient_decrease::F = 10^(-4),
            sufficient_curvature::F = 0.999,
            retraction_method::RTM = default_retraction_method(M),
            vector_transport_method::VTM = default_vector_transport_method(M),
            stop_when_stepsize_less::F = 0.0,
        ) where {VTM <: AbstractVectorTransportMethod, RTM <: AbstractRetractionMethod, F}
        return new{RTM, VTM, F}(
            retraction_method,
            vector_transport_method,
            sufficient_decrease,
            sufficient_curvature,
            0.0,
            stop_when_stepsize_less,
        )
    end
end
function (a::WolfePowellBinaryLinesearchStepsize)(
        amp::AbstractManoptProblem,
        ams::AbstractManoptSolverState,
        ::Int,
        η = (-get_gradient(amp, get_iterate(ams)));
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
function show(io::IO, a::WolfePowellBinaryLinesearchStepsize)
    return print(
        io,
        """
        WolfePowellBinaryLinesearch(;
            sufficient_descrease = $(a.sufficient_decrease),
            sufficient_curvature = $(a.sufficient_curvature),
            retraction_method = $(a.retraction_method),
            vector_transport_method = $(a.vector_transport_method),
            stop_when_stepsize_less = $(a.stop_when_stepsize_less),
        )""",
    )
end
function status_summary(a::WolfePowellBinaryLinesearchStepsize)
    s = (a.last_stepsize > 0) ? "\nand the last stepsize used was $(a.last_stepsize)." : ""
    return "$a$s"
end

_doc_WPBL_algorithm = """With
```math
A(t) = f(p_+) ≤ c_1 t ⟨$(_tex(:grad))f(p), X⟩_{x}
$(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad))
W(t) = ⟨$(_tex(:grad))f(x_+), $(_math(:vector_transport, :symbol, "p_+", "p"))X⟩_{p_+} ≥ c_2 ⟨X, $(_tex(:grad))f(x)⟩_x,
```

where ``p_+ =$(_tex(:retr))_p(tX)`` is the current trial point, and ``$(_math(:vector_transport, :symbol))`` denotes a
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

Perform a lineseach to fulfull both the Armijo-Goldstein conditions
for some given sufficient decrease coefficient ``c_1`` and some sufficient curvature condition coefficient``c_2``.
Compared to [`WolfePowellLinesearch`](@ref Manopt.WolfePowellLinesearch) which tries a simpler method, this linesearch performs the following algorithm

$(_doc_WPBL_algorithm)

# Keyword arguments

* `sufficient_decrease=10^(-4)`
* `sufficient_curvature=0.999`
* `max_stepsize=`[`max_stepsize`](@ref)`(M, p)`: largest stepsize allowed here.
$(_var(:Keyword, :retraction_method))
* `stop_when_stepsize_less=0.0`: smallest stepsize when to stop (the last one before is taken)
$(_var(:Keyword, :vector_transport_method))
"""
function WolfePowellBinaryLinesearch(args...; kwargs...)
    return ManifoldDefaultsFactory(WolfePowellBinaryLinesearchStepsize, args...; kwargs...)
end

@doc """
    get_stepsize(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...)

return the stepsize stored within [`AbstractManoptSolverState`](@ref) `ams` when solving the
[`AbstractManoptProblem`](@ref) `amp`.
This method also works for decorated options and the [`Stepsize`](@ref) function within
the options, by default stored in `ams.stepsize`.
"""
function get_stepsize(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...; kwargs...
    )
    return _get_stepsize(amp, ams, dispatch_state_decorator(ams), vars...; kwargs...)
end
function _get_stepsize(
        amp::AbstractManoptProblem,
        ams::AbstractManoptSolverState,
        ::Val{true},
        vars...;
        kwargs...,
    )
    return get_stepsize(amp, ams.state, vars...; kwargs...)
end
function _get_stepsize(
        amp::AbstractManoptProblem,
        ams::AbstractManoptSolverState,
        ::Val{false},
        vars...;
        kwargs...,
    )
    return ams.stepsize(amp, ams, vars...; kwargs...)
end

function get_initial_stepsize(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...; kwargs...
    )
    return _get_initial_stepsize(
        amp::AbstractManoptProblem,
        ams::AbstractManoptSolverState,
        dispatch_state_decorator(ams),
        vars...;
        kwargs...,
    )
end
function _get_initial_stepsize(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Val{true}, vars...
    )
    return get_initial_stepsize(amp, ams.state)
end
function _get_initial_stepsize(
        ::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Val{false}, vars...
    )
    return get_initial_stepsize(ams.stepsize)
end

@doc """
    get_last_stepsize(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...)

return the last computed stepsize stored within [`AbstractManoptSolverState`](@ref) `ams`
when solving the [`AbstractManoptProblem`](@ref) `amp`.

This method takes into account that `ams` might be decorated.
In case this returns `NaN`, a concrete call to the stored stepsize is performed.
For this, usually, the first of the `vars...` should be the current iterate.
"""
function get_last_stepsize(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...
    )
    return _get_last_stepsize(amp, ams, dispatch_state_decorator(ams), vars...)
end
function _get_last_stepsize(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Val{true}, vars...
    )
    return get_last_stepsize(amp, ams.state, vars...)
end
function _get_last_stepsize(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Val{false}, vars...
    )
    s = get_last_stepsize(ams.stepsize) # if it stores the stepsize itself -> return
    !isnan(s) && return s
    # if not -> call step.
    return ams.stepsize(amp, ams, vars...)
end
@doc """
    get_last_stepsize(::Stepsize, vars...)

return the last computed stepsize from within the stepsize.
If no last step size is stored, this returns `NaN`.
"""
get_last_stepsize(::Stepsize, ::Any...) = NaN
function get_last_stepsize(step::ArmijoLinesearchStepsize, ::Any...)
    return step.last_stepsize
end
function get_last_stepsize(step::WolfePowellLinesearchStepsize, ::Any...)
    return step.last_stepsize
end
function get_last_stepsize(step::WolfePowellBinaryLinesearchStepsize, ::Any...)
    return step.last_stepsize
end
