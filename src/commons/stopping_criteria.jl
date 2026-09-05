#
# Meta Stopping Criteria
# ---


"""
    evaluate_all_criteria(criteria::Tuple, problem, state, k)

Evaluate stopping criteria for a [`StopWhenAll`](@ref). Once one criterion returns `false`,
only criteria not eligible for short-circuiting are evaluated.
"""
@inline evaluate_all_criteria(
    ::Tuple{}, ::AbstractManoptProblem, ::AbstractManoptSolverState, ::Int
) = true
@inline function evaluate_all_criteria(
        criteria::Tuple{C, Vararg{StoppingCriterion}},
        p::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        k::Int,
    ) where {C <: StoppingCriterion}
    criterion = first(criteria)
    result = criterion(p, s, k)::Bool
    return result ? evaluate_all_criteria(Base.tail(criteria), p, s, k) :
        (_evaluate_update_criteria(Base.tail(criteria), p, s, k); false)
end

"""
    evaluate_any_criteria(criteria::Tuple, problem, state, k)

Evaluate stopping criteria for a [`StopWhenAny`](@ref). Once one criterion returns `true`,
only criteria not eligible for short-circuiting are evaluated.
"""
@inline evaluate_any_criteria(
    ::Tuple{}, ::AbstractManoptProblem, ::AbstractManoptSolverState, ::Int
) = false
@inline function evaluate_any_criteria(
        criteria::Tuple{C, Vararg{StoppingCriterion}},
        p::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        k::Int,
    ) where {C <: StoppingCriterion}
    criterion = first(criteria)
    result = criterion(p, s, k)::Bool
    return result ? (_evaluate_update_criteria(Base.tail(criteria), p, s, k); true) :
        evaluate_any_criteria(Base.tail(criteria), p, s, k)
end

"""
    _evaluate_update_criteria(criteria::Tuple, problem, state, k)

Evaluate stopping criteria for updating after the truth value has been established by
either [`evaluate_all_criteria`](@ref) or [`evaluate_any_criteria`](@ref).
"""
@inline _evaluate_update_criteria(
    ::Tuple{}, ::AbstractManoptProblem, ::AbstractManoptSolverState, ::Int
) = nothing
@inline function _evaluate_update_criteria(
        criteria::Tuple{C, Vararg{StoppingCriterion}},
        p::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        k::Int,
    ) where {C <: StoppingCriterion}
    if requires_update(C)
        first(criteria)(p, s, k)::Bool
    end
    return _evaluate_update_criteria(Base.tail(criteria), p, s, k)
end

@doc """
    StopWhenAll <: StoppingCriterionSet

Store an array of [`StoppingCriterion`](@ref) elements and indicate to stop
when _all_ of them indicate to stop. The `reason` is given by the concatenation of all
reasons.
All criteria that [`requires_update`](@ref) return `true` for are evaluated in every
call, since some internal criteria might keep an internal status.

# Fields

* `criteria`: the tuple of [`StoppingCriterion`](@ref)s that are combined
* `at_iteration`: the iteration at which this criterion last indicated to stop, `-1` otherwise

# Constructor

    StopWhenAll(c::Vector{<:StoppingCriterion})
    StopWhenAll(c::StoppingCriterion...)
"""
mutable struct StopWhenAll{TCriteria <: Tuple} <: StoppingCriterionSet
    criteria::TCriteria
    at_iteration::Int
    StopWhenAll(c::Vector{<:StoppingCriterion}) = new{typeof(tuple(c...))}(tuple(c...), -1)
    StopWhenAll(c::StoppingCriterion...) = new{typeof(c)}(c, -1)
end
function (c::StopWhenAll)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    if (k <= 0) # reset on init
        c.at_iteration = -1
        map(ci -> ci(p, s, k), c.criteria) #reset internals as well
    end
    if evaluate_all_criteria(c.criteria, p, s, k)
        c.at_iteration = k
        return true
    end
    return false
end
get_stopping_criteria(c::StopWhenAll) = c.criteria
function get_reason(c::StopWhenAll)
    if c.at_iteration >= 0
        return string([get_reason(subC) for subC in c.criteria]...)
    end
    return ""
end
function status_summary(c::StopWhenAll; context::Symbol = :default)
    if context == :short
        return join(
            [
                s isa StoppingCriterionSet ? "($(status_summary(s; context = :short)))" : status_summary(s; context = :short) for s in c.criteria
            ],
            " & "
        )
    end
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    r = "Stop when _all_ of the following are fulfilled:\n"
    for cs in c.criteria
        r = "$r  * $(_in_str(status_summary(cs; context = :inline); indent = 1, headers = 0))\n"
    end
    return "$(r)Overall: $s"
end
function indicates_convergence(c::StopWhenAll)
    return any(indicates_convergence(ci) for ci in c.criteria)
end
function has_converged(c::StopWhenAll)
    # (a) all are active (have converged) and at least one of them indicates convergence
    return is_active_stopping_criterion(c) && any(has_converged(ci) for ci in c.criteria)
end
function get_count(c::StopWhenAll, v::Val{:Iterations})
    iters = [get_count(ci, v) for ci in c.criteria]
    any(x -> x < 0, iters) && (return -1) # Not all indicated to stop yet, so this one did not either
    return maximum(iters; init = -1)
end
function set_parameter!(c::StopWhenAll, e::Val, v)
    for d in c.criteria
        set_parameter!(d, e, v)
    end
    return c
end
function Base.show(io::IO, c::StopWhenAll)
    print(io, "StopWhenAll([")
    first = true
    for cs in c.criteria
        if !first
            print(io, ", ")
        else
            first = false
        end
        show(io, cs)
    end
    return print(io, "])")
end
function requires_update(::Type{StopWhenAll{TC}}) where {TC <: Tuple}
    return any(map(requires_update, Tuple(TC.parameters)))
end

"""
    &(s1,s2)
    s1 & s2

Combine two [`StoppingCriterion`](@ref) within a [`StopWhenAll`](@ref).
If either `s1` (or `s2`) is already a [`StopWhenAll`](@ref), then `s2` (or `s1`) is
appended to the list of [`StoppingCriterion`](@ref) within `s1` (or `s2`).

# Example
    a = StopAfterIteration(200) & StopWhenChangeLess(M, 1e-6)
    b = a & StopWhenGradientNormLess(1e-6)

Is the same as

    a = StopWhenAll(StopAfterIteration(200), StopWhenChangeLess(M, 1e-6))
    b = StopWhenAll(StopAfterIteration(200), StopWhenChangeLess(M, 1e-6), StopWhenGradientNormLess(1e-6))
"""
function Base.:&(s1::S, s2::T) where {S <: StoppingCriterion, T <: StoppingCriterion}
    return StopWhenAll(s1, s2)
end
function Base.:&(s1::S, s2::StopWhenAll) where {S <: StoppingCriterion}
    return StopWhenAll(s1, s2.criteria...)
end
function Base.:&(s1::StopWhenAll, s2::T) where {T <: StoppingCriterion}
    return StopWhenAll(s1.criteria..., s2)
end
function Base.:&(s1::StopWhenAll, s2::StopWhenAll)
    return StopWhenAll(s1.criteria..., s2.criteria...)
end

@doc """
    StopWhenAny <: StoppingCriterionSet

Store an array of [`StoppingCriterion`](@ref) elements and indicate to stop
when _any_ single one indicates to stop. The `reason` is given by the
concatenation of all reasons (assuming that all non-indicating return `""`).
All criteria that [`requires_update`](@ref) return `true` for are evaluated in every
call, since some internal criteria might keep an internal status.

# Fields

* `criteria`: the tuple of [`StoppingCriterion`](@ref)s that are combined
* `at_iteration`: the iteration at which this criterion last indicated to stop, `-1` otherwise

# Constructor
    StopWhenAny(c::Vector{<:StoppingCriterion})
    StopWhenAny(c::StoppingCriterion...)
"""
mutable struct StopWhenAny{TCriteria <: Tuple} <: StoppingCriterionSet
    criteria::TCriteria
    at_iteration::Int
    StopWhenAny(c::Vector{<:StoppingCriterion}) = new{typeof(tuple(c...))}(tuple(c...), -1)
    StopWhenAny(c::StoppingCriterion...) = new{typeof(c)}(c, -1)
end
function (c::StopWhenAny)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    if (k <= 0) # reset on init
        c.at_iteration = -1
        for ci in c.criteria #reset internals as well
            ci(p, s, k)
        end
    end
    if evaluate_any_criteria(c.criteria, p, s, k)
        c.at_iteration = k
        return true
    end
    return false
end
get_stopping_criteria(c::StopWhenAny) = c.criteria
function get_reason(c::StopWhenAny)
    if (c.at_iteration >= 0)
        return string((get_reason(subC) for subC in c.criteria)...)
    end
    return ""
end
function set_parameter!(c::StopWhenAny, e::Val, v)
    for d in c.criteria
        set_parameter!(d, e, v)
    end
    return c
end
function status_summary(c::StopWhenAny; context::Symbol = :default)
    if context == :short
        return join(
            [
                s isa StoppingCriterionSet ? "($(status_summary(s; context = :short)))" : status_summary(s; context = :short) for s in c.criteria
            ],
            " | "
        )
    end
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    r = "Stop when _one_ of the following is fulfilled:\n"
    for cs in c.criteria
        r = "$r  * $(_in_str(status_summary(cs; context = :inline); indent = 1, headers = 0))\n"
    end
    return "$(r)Overall: $s"
end
function indicates_convergence(c::StopWhenAny)
    # Statically we can only indicate convergence in general if all indicate convergence,
    # so that independent of “which one fires” we conclude with convergence
    return all(indicates_convergence(ci) for ci in c.criteria)
end
function has_converged(c::StopWhenAny)
    # (a) we are active and (b) at least one of the active ones indicates convergence If any of the active ones has_converged – we stop due to convergence
    return is_active_stopping_criterion(c) && any(is_active_stopping_criterion(ci) && has_converged(ci) for ci in c.criteria)
end
function get_count(c::StopWhenAny, v::Val{:Iterations})
    iters = filter(x -> x >= 0, [get_count(ci, v) for ci in c.criteria])
    (length(iters) == 0) && (return -1) # None indicated to stop yet, so we also do not
    return minimum(iters)
end
function Base.show(io::IO, c::StopWhenAny)
    print(io, "StopWhenAny([")
    first = true
    for cs in c.criteria
        if !first
            print(io, ", ")
        else
            first = false
        end
        show(io, cs)
    end
    return print(io, "])")
end
function requires_update(::Type{StopWhenAny{TC}}) where {TC <: Tuple}
    return any(map(requires_update, Tuple(TC.parameters)))
end
"""
    |(s1,s2)
    s1 | s2

Combine two [`StoppingCriterion`](@ref) within a [`StopWhenAny`](@ref).
If either `s1` (or `s2`) is already a [`StopWhenAny`](@ref), then `s2` (or `s1`) is
appended to the list of [`StoppingCriterion`](@ref) within `s1` (or `s2`).

# Example
    a = StopAfterIteration(200) | StopWhenChangeLess(M, 1e-6)
    b = a | StopWhenGradientNormLess(1e-6)

Is the same as

    a = StopWhenAny(StopAfterIteration(200), StopWhenChangeLess(M, 1e-6))
    b = StopWhenAny(StopAfterIteration(200), StopWhenChangeLess(M, 1e-6), StopWhenGradientNormLess(1e-6))
"""
function Base.:|(s1::S, s2::T) where {S <: StoppingCriterion, T <: StoppingCriterion}
    return StopWhenAny(s1, s2)
end
function Base.:|(s1::S, s2::StopWhenAny) where {S <: StoppingCriterion}
    return StopWhenAny(s1, s2.criteria...)
end
function Base.:|(s1::StopWhenAny, s2::T) where {T <: StoppingCriterion}
    return StopWhenAny(s1.criteria..., s2)
end
function Base.:|(s1::StopWhenAny, s2::StopWhenAny)
    return StopWhenAny(s1.criteria..., s2.criteria...)
end

#
#
# ---
"""
    StopAfter <: StoppingCriterion

Store a threshold when to stop looking at the complete runtime. It uses
`time_ns()` to measure the time and you provide a `Period` as a time limit,
for example `Minute(15)`.

# Fields

* `threshold`: stores the `Period` after which to stop
* `start`: stores the starting time when the algorithm is started, that is a call with `k=0`.
* `time`: stores the elapsed time
* `at_iteration`: indicates at which iteration (including `k=0`) the stopping criterion
  was fulfilled and is `-1` while it is not fulfilled.

# Constructor

    StopAfter(t::Period)

initialize the stopping criterion to a `Period` `t` to stop after.
"""
mutable struct StopAfter <: StoppingCriterion
    threshold::Period
    start::Nanosecond
    time::Nanosecond
    at_iteration::Int
    function StopAfter(t::Period)
        return if value(t) < 0
            error("You must provide a positive time period")
        else
            new(t, Nanosecond(0), Nanosecond(0), -1)
        end
    end
end
function (c::StopAfter)(::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int)
    if value(c.start) == 0 || k <= 0 # (re)start timer
        c.at_iteration = -1
        c.start = Nanosecond(time_ns())
        c.time = Nanosecond(0)
    else
        c.time = Nanosecond(time_ns()) - c.start
        if k > 0 && (c.time > Nanosecond(c.threshold))
            c.at_iteration = k
            return true
        end
    end
    return false
end
indicates_convergence(c::StopAfter) = false
function get_reason(c::StopAfter)
    if (c.at_iteration >= 0)
        return "The algorithm ran for $(floor(c.time, typeof(c.threshold))) (threshold: $(c.threshold)).\n"
    end
    return ""
end
function status_summary(c::StopAfter; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = (has_stopped ? "reached" : "not reached")
    return (_is_inline(context) ? "stopped after $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop after $(c.threshold)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopAfter)
    return print(io, "StopAfter($(repr(c.threshold)))")
end
requires_update(::Type{StopAfter}) = false
@doc """
    set_parameter!(c::StopAfter, :MaxTime, v::Period)

Update the time period after which an algorithm shall stop.
"""
function set_parameter!(c::StopAfter, ::Val{:MaxTime}, v::Period)
    (value(v) < 0) && error("You must provide a positive time period")
    c.threshold = v
    return c
end
@doc """
    StopAfterIteration <: StoppingCriterion

A functor for a stopping criterion to stop after a maximal number of iterations.

# Fields

* `max_iterations`: stores the maximal iteration number where to stop at
* `at_iteration`: indicates at which iteration (including `k=0`) the stopping criterion
  was fulfilled and is `-1` while it is not fulfilled.

# Constructor

    StopAfterIteration(max_iterations)

initialize the functor to indicate to stop after `max_iterations` iterations.
"""
mutable struct StopAfterIteration <: StoppingCriterion
    max_iterations::Int
    at_iteration::Int
    StopAfterIteration(k::Int) = new(k, -1)
end
function (c::StopAfterIteration)(
        ::P, ::S, k::Int
    ) where {P <: AbstractManoptProblem, S <: AbstractManoptSolverState}
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if k >= c.max_iterations
        c.at_iteration = k
        return true
    end
    return false
end
indicates_convergence(c::StopAfterIteration) = false
function get_reason(c::StopAfterIteration)
    if c.at_iteration >= c.max_iterations
        return "At iteration $(c.at_iteration) the algorithm reached its maximal number of iterations ($(c.max_iterations)).\n"
    end
    return ""
end
function status_summary(c::StopAfterIteration; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "stopped after $(c.max_iterations) iterations:$(_MANOPT_INDENT)" : "A stopping criterion to stop after $(c.max_iterations) iterations\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopAfterIteration)
    return print(io, "StopAfterIteration($(c.max_iterations))")
end
requires_update(::Type{StopAfterIteration}) = false

"""
    set_parameter!(c::StopAfterIteration, :MaxIteration, v::Int)

Update the number of iterations after which the algorithm should stop.
"""
function set_parameter!(c::StopAfterIteration, ::Val{:MaxIteration}, v::Int)
    c.max_iterations = v
    return c
end

"""
    StopWhenChangeLess <: StoppingCriterion

Store a threshold when to stop looking at the norm of the change of the
optimization variable from within a [`AbstractManoptSolverState`](@ref) `s`.
That is, by accessing `get_iterate(s)` and comparing successive iterates.
For the storage a [`StoreStateAction`](@ref) is used.

# Fields

$(_fields([:at_iteration, :last_change, :inverse_retraction_method, :storage]))
* `threshold`: the threshold for the change to check (run under to stop)
* `outer_norm`: if `M` is a manifold with components, this can be used to specify the norm,
  that is used to compute the overall distance based on the element-wise distance.
  You can deactivate this by setting this value to `missing`.

The `inverse_retraction_method` can be used to approximate the distance by that inverse
retraction together with a norm on the tangent space, if neither the distance nor the
logarithmic map are available on `M`.

# Example

On an $(_link(:AbstractPowerManifold)) like ``$(_math(:Manifold)) = $(_math(:Manifold; M = "N"))^n``
any point ``p = (p_1,…,p_n) ∈ $(_math(:Manifold))`` is a vector of length ``n`` of points ``p_i ∈ $(_math(:Manifold; M = "N"))``.
Then, denoting the `outer_norm` by ``r``, the distance of two points ``p,q ∈ $(_math(:Manifold))``
is given by

```math
$(_math(:distance))(p,q) = $(_tex(:Bigl))( $(_tex(:sum))_{k=1}^n $(_math(:distance))(p_k,q_k)^r $(_tex(:Bigr)))^{$(_tex(:frac, "1", "r"))},
```

where the sum turns into a maximum for the case ``r=∞``.
The `outer_norm` has no effect on manifolds that do not consist of components.

# Constructor

    StopWhenChangeLess(
        M::AbstractManifold,
        threshold::Float64;
        storage::StoreStateAction=StoreStateAction(M; store_points=Tuple{:Iterate}),
        inverse_retraction_method::IRT=default_inverse_retraction_method(M),
        outer_norm::Union{Missing,Real}=missing
    )

initialize the stopping criterion to a threshold `ε` using the
[`StoreStateAction`](@ref) `storage`, which is initialized to just store `:Iterate` by
default. You can also provide an `inverse_retraction_method` for the `distance`, or a manifold
to use its default inverse retraction.
"""
mutable struct StopWhenChangeLess{
        F, IRT <: AbstractInverseRetractionMethod, TSSA <: StoreStateAction, N <: Union{Missing, Real},
    } <: StoppingCriterion
    threshold::F
    last_change::F
    storage::TSSA
    inverse_retraction_method::IRT
    at_iteration::Int
    outer_norm::N
end
function StopWhenChangeLess(
        M::AbstractManifold, ε::F;
        storage::StoreStateAction = StoreStateAction(M; store_points = Tuple{:Iterate}),
        inverse_retraction_method::IRT = default_inverse_retraction_method(M),
        outer_norm::N = missing,
    ) where {F, N <: Union{Missing, Real}, IRT <: AbstractInverseRetractionMethod}
    e = float(ε)
    return StopWhenChangeLess{typeof(e), IRT, typeof(storage), N}(
        e, zero(e), storage, inverse_retraction_method, -1, outer_norm
    )
end
function StopWhenChangeLess(
        ε::R; storage::StoreStateAction = StoreStateAction([:Iterate]), kwargs...
    ) where {R <: Real}
    return StopWhenChangeLess(DefaultManifold(), ε; storage = storage, kwargs...)
end
function (c::StopWhenChangeLess)(mp::AbstractManoptProblem, s::AbstractManoptSolverState, k)
    if k == 0 # reset on init
        c.at_iteration = -1
        c.last_change = Inf
    end
    if has_storage(c.storage, PointStorageKey(:Iterate))
        M = get_manifold(mp)
        p_old = get_storage(c.storage, PointStorageKey(:Iterate))
        r = (has_components(M) && !ismissing(c.outer_norm)) ? (c.outer_norm,) : ()
        c.last_change = distance(
            M, get_iterate(s), p_old, c.inverse_retraction_method, r...
        )
        if c.last_change < c.threshold && k > 0
            c.at_iteration = k
            c.storage(mp, s, k)
            return true
        end
    end
    c.storage(mp, s, k)
    return false
end
function get_reason(c::StopWhenChangeLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "At iteration $(c.at_iteration) the algorithm performed a step with a change ($(c.last_change)) less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenChangeLess; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "|Δp| < $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the change of the iterate is less than $(c.threshold)\n using the $(repr(c.inverse_retraction_method))\n$(_MANOPT_INDENT)") * "$s"
end
indicates_convergence(c::StopWhenChangeLess) = false
function Base.show(io::IO, c::StopWhenChangeLess)
    print(io, "StopWhenChangeLess($(c.threshold); inverse_retraction_method=$(repr(c.inverse_retraction_method))")
    !ismissing(c.outer_norm) && print(io, ", outer_norm = ", c.outer_norm)
    return print(io, ")")
end

"""
    set_parameter!(c::StopWhenChangeLess, :MinIterateChange, v)

Update the minimal change below which an algorithm shall stop.
"""
function set_parameter!(c::StopWhenChangeLess, ::Val{:MinIterateChange}, v)
    c.threshold = v
    return c
end

"""
    StopWhenCostChangeLess <: StoppingCriterion

A stopping criterion to stop when the change of the cost function is less than a certain threshold.

# Fields
$(_fields([:at_iteration, :last_change]))
* `last_cost`: the last cost value
* `tolerance`: the threshold for the change of the cost

# Constructor

    StopWhenCostChangeLess(tolerance::F)

Initialize the stopping criterion to a threshold `tolerance` for the change of the cost function.
"""
mutable struct StopWhenCostChangeLess{F <: Real} <: StoppingCriterion
    tolerance::F
    at_iteration::Int
    last_cost::F
    last_change::F
end
function StopWhenCostChangeLess(tol::Real)
    t = float(tol)
    return StopWhenCostChangeLess{typeof(t)}(t, -1, zero(t), 2 * t)
end
function (c::StopWhenCostChangeLess)(
        problem::AbstractManoptProblem, state::AbstractManoptSolverState, k::Int
    )
    if k <= 0 # reset on init
        c.at_iteration = -1
        c.last_cost = Inf
        c.last_change = 2 * c.tolerance
    end
    c.last_change = c.last_cost
    c.last_cost = get_cost(problem, state)
    c.last_change = c.last_change - c.last_cost
    if abs(c.last_change) < c.tolerance
        c.at_iteration = k
        return true
    end
    return false
end
indicates_convergence(c::StopWhenCostChangeLess) = false
function get_reason(c::StopWhenCostChangeLess)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration) the algorithm performed a step with an absolute cost change ($(abs(c.last_change))) less than $(c.tolerance).\n"
    end
    return ""
end
function status_summary(c::StopWhenCostChangeLess; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "|Δf(p)| = $(abs(c.last_change)) < $(c.tolerance):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the change of the cost function is less than $(c.tolerance)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopWhenCostChangeLess)
    return print(io, "StopWhenCostChangeLess($(c.tolerance))")
end

"""
    StopWhenCostLess <: StoppingCriterion

store a threshold when to stop looking at the cost function of the
optimization problem from within a [`AbstractManoptProblem`](@ref), i.e `get_cost(p, s)`.

# Constructor

    StopWhenCostLess(ε::Real)

initialize the stopping criterion to a threshold `ε`.
"""
mutable struct StopWhenCostLess{F} <: StoppingCriterion
    threshold::F
    last_cost::F
    at_iteration::Int
    function StopWhenCostLess(ε::Real)
        e = float(ε)
        return new{typeof(e)}(e, zero(e), -1)
    end
end
function (c::StopWhenCostLess)(
        p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    c.last_cost = get_cost(p, s)
    if c.last_cost < c.threshold
        c.at_iteration = k
        return true
    end
    return false
end
indicates_convergence(c::StopWhenCostLess) = false
function get_reason(c::StopWhenCostLess)
    if (c.last_cost < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached a cost function value ($(c.last_cost)) less than the threshold ($(c.threshold)).\n"
    end
    return ""
end
function status_summary(c::StopWhenCostLess; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "f(x) < $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the cost function is less than $(c.threshold)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopWhenCostLess)
    return print(io, "StopWhenCostLess($(c.threshold))")
end
requires_update(::Type{<:StopWhenCostLess}) = false

"""
    set_parameter!(c::StopWhenCostLess, :MinCost, v)

Update the minimal cost below which the algorithm shall stop.
"""
function set_parameter!(c::StopWhenCostLess, ::Val{:MinCost}, v)
    c.threshold = v
    return c
end

#
#
# ---
"""
    StopWhenCostNaN <: StoppingCriterion

Stop the solver when the cost function of the optimization problem
[`AbstractManoptProblem`](@ref) is `NaN`. The value is obtained using `get_cost(p, s)`.

# Constructor

    StopWhenCostNaN()

initialize the stopping criterion with `at_iteration` equal to -1.
"""
mutable struct StopWhenCostNaN <: StoppingCriterion
    at_iteration::Int
    StopWhenCostNaN() = new(-1)
end
function (c::StopWhenCostNaN)(
        p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    # but still verify whether it yields NaN
    if isnan(get_cost(p, s))
        c.at_iteration = k
        return true
    end
    return false
end
indicates_convergence(::StopWhenCostNaN) = false
function get_reason(c::StopWhenCostNaN)
    if c.at_iteration >= 0
        return "The algorithm reached a cost function value of NaN.\n"
    end
    return ""
end
function status_summary(c::StopWhenCostNaN; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "f(x) is NaN:$(_MANOPT_INDENT)" : "A stopping criterion to stop when the cost function is NaN\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, ::StopWhenCostNaN)
    return print(io, "StopWhenCostNaN()")
end
requires_update(::Type{StopWhenCostNaN}) = false

#
#
# ---
@doc """
    StopWhenCriterionWithIterationCondition <: StoppingCriterion

A stopping criterion that only evaluates a certain (inner) stopping criterion based on a
condition on the iteration `k`.
The condition is a function `comp(k) -> Bool`.

# Example

`comp = >(n)` would only activate the wrapped stopping criterion after `n` iterations.

# Fields

* `stopping_criterion`: the [`StoppingCriterion`](@ref) to wrap
* `comp`: the condition on the iteration `k` that decides whether the wrapped criterion is checked
* `at_iteration`: the iteration at which this criterion indicated to stop, `-1` otherwise

# Constructor

    StopWhenCriterionWithIterationCondition(criterion::StoppingCriterion, n=0; comp = (>(n)))

Create a stopping criterion that only checks the inner `criterion` in those iterations `k`
for which `comp(k)` is `true`. The `n` is ignored if you provide a manual functor `comp`.

# Examples

A stopping criterion that indicates to stop when the gradient norm is small but only after the third iteration

    StopWhenCriterionWithIterationCondition(StopWhenGradientNormLess(1e-6), 3)

You can also use the infix operators `≟` (`\\questeq` on REPL), `⩻` (`\\ltquest`), `⩼` (`\\gtquest`), and `≞` (`\\measeq`) to create such a criterion:

    StopWhenGradientNormLess(1e-6) ≟ 3
    StopWhenGradientNormLess(1e-6) ⩻ 3
    StopWhenGradientNormLess(1e-6) ⩼ 3
    StopWhenGradientNormLess(1e-6) ≞ 3

These are equivalent to specifying `comp = (==(3))`, `comp = (<(3))`, `comp = (>(3))`, and `comp = rem(k,n)==0` respectively.
Their interpretation is “the stopping criterion is only checked (asked) if the condition is met”:
* `≟` is only checked exactly at iteration 3,
* `⩻` is only checked up to (but not including) iteration 3
* `⩼` is only checked after (but not including) iteration 3
* `≞` is only checked on iterations that are zero modulo 3
"""
mutable struct StopWhenCriterionWithIterationCondition{SC <: StoppingCriterion, F} <:
    StoppingCriterion
    stopping_criterion::SC
    comp::F
    at_iteration::Int
end
function StopWhenCriterionWithIterationCondition(
        sc::SC, n::Int = 0; comp::F = (>(n))
    ) where {SC <: StoppingCriterion, F}
    return StopWhenCriterionWithIterationCondition{SC, F}(sc, comp, -1)
end
function ⩻(sc::StoppingCriterion, n::Int)
    return StopWhenCriterionWithIterationCondition(sc; comp = (<(n)))
end
function ⩼(sc::StoppingCriterion, n::Int)
    return StopWhenCriterionWithIterationCondition(sc; comp = (>(n)))
end
function ≟(sc::StoppingCriterion, n::Int)
    return StopWhenCriterionWithIterationCondition(sc; comp = (==(n)))
end
function ≞(sc::StoppingCriterion, n::Int)
    return StopWhenCriterionWithIterationCondition(sc; comp = k -> rem(k, n) == 0)
end
function (c::StopWhenCriterionWithIterationCondition)(
        p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    if k <= 0 # reset on init
        c.at_iteration = -1
        c.stopping_criterion(p, s, k) # reset the criterion
        return false
    end
    if c.comp(k)
        # evaluate the inner stopping criterion
        stop = c.stopping_criterion(p, s, k)
        if stop # if we indicated to stop
            c.at_iteration = k
            return true
        end
    end
    # Else: do not even check the other one.
    return false
end
function get_reason(sc::StopWhenCriterionWithIterationCondition)
    has_stopped = (sc.at_iteration >= 0)
    if has_stopped
        r = "At iteration $(sc.at_iteration), the stopping criterion $(typeof(sc.stopping_criterion)) has indicated to stop together with $(sc.comp), since $(status_summary(sc.stopping_criterion))\n"
        return r
    end
    return ""
end
function indicates_convergence(sc::StopWhenCriterionWithIterationCondition)
    return indicates_convergence(sc.stopping_criterion)
end
function has_converged(sc::StopWhenCriterionWithIterationCondition)
    # When the inner one indicates convergence, this does as well
    return has_converged(sc.stopping_criterion)
end
function Base.show(io::IO, sc::StopWhenCriterionWithIterationCondition)
    return print(io, "StopWhenCriterionWithIterationCondition($(typeof(sc.stopping_criterion)), $(sc.comp))")
end
function status_summary(sc::StopWhenCriterionWithIterationCondition; context::Symbol = :default)
    (context == :short) && return repr(sc)
    has_stopped = (sc.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    is = replace("$(status_summary(sc.stopping_criterion; context = context))", "\n" => "\n    ") #increase indent
    return (_is_inline(context) ? "$(sc.comp) && $(is):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the inner criterion is met and $(sc.comp)\n$(_MANOPT_INDENT)$(is)\n$(_MANOPT_INDENT)$(_MANOPT_INDENT)") * "$s"
end


#
#
# ---
@doc """
    StopWhenEntryChangeLess

Evaluate whether a certain field's change is less than a certain threshold.

# Fields

* `field`:     a symbol addressing the corresponding field in a certain subtype of [`AbstractManoptSolverState`](@ref) to track
* `distance`:  a function `(problem, state, v1, v2) -> R` that computes the distance between two possible values of the `field`
* `storage`:   a [`StoreStateAction`](@ref) to store the previous value of the `field`
* `threshold`: the threshold to indicate to stop when the distance is below this value

# Internal fields

* `at_iteration`: store the iteration at which the stop indication happened
* `last_change`:  the last change recorded in this stopping criterion

# Constructor

    StopWhenEntryChangeLess(
        field::Symbol,
        distance,
        threshold;
        storage::StoreStateAction=StoreStateAction([field]),
    )

"""
mutable struct StopWhenEntryChangeLess{F, TF, TSSA <: StoreStateAction} <: StoppingCriterion
    at_iteration::Int
    distance::F
    field::Symbol
    storage::TSSA
    threshold::TF
    last_change::TF
end
function StopWhenEntryChangeLess(
        field::Symbol, distance::F, threshold; storage::TSSA = StoreStateAction([field])
    ) where {F, TSSA <: StoreStateAction}
    t = float(threshold)
    return StopWhenEntryChangeLess{F, typeof(t), TSSA}(
        -1, distance, field, storage, t, zero(t)
    )
end

function (sc::StopWhenEntryChangeLess)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k
    )
    if k == 0 # reset on init
        sc.at_iteration = -1
    end
    if has_storage(sc.storage, sc.field)
        old_field_value = get_storage(sc.storage, sc.field)
        sc.last_change = sc.distance(mp, s, old_field_value, getproperty(s, sc.field))
        if (k > 0) && (sc.last_change < sc.threshold)
            sc.at_iteration = k
            sc.storage(mp, s, k)
            return true
        end
    end
    sc.storage(mp, s, k)
    return false
end
indicates_convergence(sc::StopWhenEntryChangeLess) = false
function get_reason(sc::StopWhenEntryChangeLess)
    if (sc.last_change < sc.threshold) && (sc.at_iteration >= 0)
        return "At iteration $(sc.at_iteration) the algorithm performed a step with a change ($(sc.last_change)) in $(sc.field) less than $(sc.threshold).\n"
    end
    return ""
end
function status_summary(sc::StopWhenEntryChangeLess; context::Symbol = :default)
    (context == :short) && return repr(sc)
    has_stopped = (sc.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "|Δ:$(sc.field)| < $(sc.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the change of $(sc.field) is less than $(sc.threshold)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, sc::StopWhenEntryChangeLess)
    return print(io, "StopWhenEntryChangeLess($(sc.field), $(sc.distance), $(sc.threshold))")
end

"""
    set_parameter!(c::StopWhenEntryChangeLess, :Threshold, v)

Update the threshold for the change of the tracked field below which the algorithm shall stop.
"""
function set_parameter!(c::StopWhenEntryChangeLess, ::Val{:Threshold}, v)
    c.threshold = v
    return c
end

#
#
# ---
@doc """
    StopWhenGradientChangeLess <: StoppingCriterion

A stopping criterion based on the change of the gradient.

# Fields

$(_fields([:at_iteration, :last_change, :vector_transport_method, :storage]))
* `threshold`: the threshold for the change to check (run under to stop)
* `outer_norm`: if `M` is a manifold with components, this can be used to specify the norm,
  that is used to compute the overall distance based on the element-wise distance.
  You can deactivate this by setting this value to `missing`.

# Example

On an $(_link(:AbstractPowerManifold)) like ``$(_math(:Manifold)) = $(_math(:Manifold; M = "N"))^n``
any point ``p = (p_1,…,p_n) ∈ $(_math(:Manifold))`` is a vector of length ``n`` of points ``p_i ∈ $(_math(:Manifold; M = "N"))``.
Then, denoting the `outer_norm` by ``r``, the norm of the difference of tangent vectors like the last and current gradient ``X,Y ∈ $(_math(:TangentSpace))``
is given by

```math
$(_tex(:norm, "X-Y"; index = "p")) = $(_tex(:Bigl))( $(_tex(:sum))_{k=1}^n $(_tex(:norm, "X_k-Y_k"; index = "p_k"))^r $(_tex(:Bigr)))^{$(_tex(:frac, "1", "r"))},
```

where the sum turns into a maximum for the case ``r=∞``.
The `outer_norm` has no effect on manifolds that do not consist of components.

# Constructor

    StopWhenGradientChangeLess(
        M::AbstractManifold,
        ε::Float64;
        storage::StoreStateAction=StoreStateAction(M; store_points=Tuple{:Iterate}, store_vectors=Tuple{:Gradient}),
        vector_transport_method::VTM=default_vector_transport_method(M),
        outer_norm::N=missing
    )

Create a stopping criterion with threshold `ε` for the change of the gradient, that is, this
criterion indicates to stop when the norm of the change of [`get_gradient`](@ref) is less than
`ε`, where `vector_transport_method` denotes the vector transport ``$(_tex(:Cal, "T"))`` used.
"""
mutable struct StopWhenGradientChangeLess{
        F, VTM <: AbstractVectorTransportMethod, TSSA <: StoreStateAction, N <: Union{Missing, Real},
    } <: StoppingCriterion
    threshold::F
    last_change::F
    storage::TSSA
    vector_transport_method::VTM
    at_iteration::Int
    outer_norm::N
end
function StopWhenGradientChangeLess(
        M::AbstractManifold, ε::F;
        storage::StoreStateAction = StoreStateAction(
            M; store_points = Tuple{:Iterate}, store_vectors = Tuple{:Gradient}
        ),
        vector_transport_method::VTM = default_vector_transport_method(M),
        outer_norm::N = missing,
    ) where {F, N <: Union{Missing, Real}, VTM <: AbstractVectorTransportMethod}
    e = float(ε)
    return StopWhenGradientChangeLess{typeof(e), VTM, typeof(storage), N}(
        e, zero(e), storage, vector_transport_method, -1, outer_norm
    )
end
function StopWhenGradientChangeLess(
        ε::Float64; storage::StoreStateAction = StoreStateAction([:Iterate, :Gradient]), kwargs...
    )
    return StopWhenGradientChangeLess(DefaultManifold(1), ε; storage = storage, kwargs...)
end
function (c::StopWhenGradientChangeLess)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    M = get_manifold(mp)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if has_storage(c.storage, PointStorageKey(:Iterate)) &&
            has_storage(c.storage, VectorStorageKey(:Gradient))
        M = get_manifold(mp)
        p_old = get_storage(c.storage, PointStorageKey(:Iterate))
        X_old = get_storage(c.storage, VectorStorageKey(:Gradient))
        p = get_iterate(s)
        Xt = vector_transport_to(M, p_old, X_old, p, c.vector_transport_method)
        r = (has_components(M) && !ismissing(c.outer_norm)) ? (c.outer_norm,) : ()
        c.last_change = norm(M, p, Xt - get_gradient(s), r...)
        if c.last_change < c.threshold && k > 0
            c.at_iteration = k
            c.storage(mp, s, k)
            return true
        end
    end
    c.storage(mp, s, k)
    return false
end
indicates_convergence(c::StopWhenGradientChangeLess) = false
function get_reason(c::StopWhenGradientChangeLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "At iteration $(c.at_iteration) the change of the gradient ($(c.last_change)) was less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenGradientChangeLess; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "|Δgrad f| < $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the change of the gradient is less than $(c.threshold)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopWhenGradientChangeLess)
    return print(io, "StopWhenGradientChangeLess($(c.threshold); vector_transport_method=$(c.vector_transport_method))")
end

"""
    set_parameter!(c::StopWhenGradientChangeLess, :MinGradientChange, v)

Update the minimal change below which an algorithm shall stop.
"""
function set_parameter!(c::StopWhenGradientChangeLess, ::Val{:MinGradientChange}, v)
    c.threshold = v
    return c
end

#
#
# ---
"""
    StopWhenGradientMappingNormLess <: StoppingCriterion

A stopping criterion based on the gradient mapping norm for proximal gradient methods.

# Fields

$(_fields(:at_iteration))
$(_fields(:last_change))
* `threshold`: the threshold for the change to check (run under to stop)

# Constructor

    StopWhenGradientMappingNormLess(ε)

Create a stopping criterion with threshold `ε` for the gradient mapping for the [`proximal_gradient_method`](@ref).
That is, this criterion indicates to stop when the gradient mapping has a norm less than `ε`.
The gradient mapping is defined as
``G_λ(p) = -$(_tex(:frac, "1", "λ"))$(_tex(:log))_p$(_tex(:bigl))(T_λ(p)$(_tex(:bigr)))``,
where, for ``f = g + h`` with ``g`` smooth and ``h`` (possibly) nonsmooth,
``T_λ(p) = $(_tex(:prox))_{λ h}$(_tex(:bigl))($(_tex(:retr))_p(-λ $(_tex(:grad)) g(p))$(_tex(:bigr)))``
is the proximal mapping.
"""
mutable struct StopWhenGradientMappingNormLess{TF} <: StoppingCriterion
    threshold::TF
    last_change::TF
    at_iteration::Int
    function StopWhenGradientMappingNormLess(ε::Real)
        e = float(ε)
        return new{typeof(e)}(e, zero(e), -1)
    end
end
function get_reason(c::StopWhenGradientMappingNormLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the gradient mapping norm ($(c.last_change)) is less than $(c.threshold).\n"
    end
    return ""
end
indicates_convergence(c::StopWhenGradientMappingNormLess) = true
requires_update(::Type{<:StopWhenGradientMappingNormLess}) = false
function Base.show(io::IO, c::StopWhenGradientMappingNormLess)
    return print(io, "StopWhenGradientMappingNormLess($(c.threshold))")
end
function status_summary(c::StopWhenGradientMappingNormLess; context::Symbol = :default)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "|G| < $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the gradient mapping norm is less than a tolerance.\n$(_MANOPT_INDENT)") * s
end

#
#
# ---
"""
    StopWhenGradientNormLess <: StoppingCriterion

A stopping criterion based on the current gradient norm.

# Fields

* `norm`:      a function `(M::AbstractManifold, p, X) -> ℝ` that computes a norm
  of the gradient `X` in the tangent space at `p` on `M`.
  For manifolds with components provide a function `(M::AbstractManifold, p, X, r) -> ℝ`.
* `threshold`: the threshold to indicate to stop when the distance is below this value
* `outer_norm`: if `M` is a manifold with components, this can be used to specify the norm,
  that is used to compute the overall distance based on the element-wise distance.

# Internal fields

* `last_change`: store the last change
* `at_iteration`: store the iteration at which the stop indication happened

# Example

On an $(_link(:AbstractPowerManifold)) like ``$(_math(:Manifold)) = $(_math(:Manifold; M = "N"))^n``
any point ``p = (p_1,…,p_n) ∈ $(_math(:Manifold))`` is a vector of length ``n`` of points ``p_i ∈ $(_math(:Manifold; M = "N"))``.
Then, denoting the `outer_norm` by ``r``, the norm of a tangent vector like the current gradient ``X ∈ $(_math(:TangentSpace))``
is given by

```math
$(_tex(:norm, "X"; index = "p")) = $(_tex(:Bigl))( $(_tex(:sum))_{k=1}^n $(_tex(:norm, "X_k"; index = "p_k"))^r $(_tex(:Bigr)))^{$(_tex(:frac, "1", "r"))},
```

where the sum turns into a maximum for the case ``r=∞``.
The `outer_norm` has no effect on manifolds that do not consist of components.

If you pass in your individual norm, this can be deactivated on such manifolds
by passing `missing` to `outer_norm`.

# Constructor

    StopWhenGradientNormLess(ε; norm=ManifoldsBase.norm, outer_norm=missing)

Create a stopping criterion with threshold `ε` for the gradient, that is, this criterion
indicates to stop when [`get_gradient`](@ref) returns a gradient vector of norm less than `ε`,
where the norm to use can be specified in the `norm=` keyword.
"""
mutable struct StopWhenGradientNormLess{F, TF <: Real, N <: Union{Missing, Real}} <: StoppingCriterion
    norm::F
    threshold::TF
    last_change::TF
    at_iteration::Int
    outer_norm::N
    function StopWhenGradientNormLess(
            ε::Real; norm::F = norm, outer_norm::N = missing
        ) where {F, N <: Union{Missing, Real}}
        e = float(ε)
        return new{F, typeof(e), N}(norm, e, zero(e), -1, outer_norm)
    end
end

function (sc::StopWhenGradientNormLess)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    M = get_manifold(mp)
    if k == 0 # reset on init
        sc.at_iteration = -1
    end
    if (k > 0)
        r = (has_components(M) && !ismissing(sc.outer_norm)) ? (sc.outer_norm,) : ()
        sc.last_change = sc.norm(M, get_iterate(s), get_gradient(s), r...)
        if sc.last_change < sc.threshold
            sc.at_iteration = k
            return true
        end
    end
    return false
end
function get_reason(c::StopWhenGradientNormLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the gradient norm ($(c.last_change)) is less than $(c.threshold).\n"
    end
    return ""
end
indicates_convergence(c::StopWhenGradientNormLess) = true
requires_update(::Type{<:StopWhenGradientNormLess}) = false
function status_summary(c::StopWhenGradientNormLess; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "|grad f| < $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the gradient norm is less than $(c.threshold)\n$(_MANOPT_INDENT)") * "$s"
end
show(io::IO, c::StopWhenGradientNormLess) = print(io, "StopWhenGradientNormLess($(c.threshold))")
"""
    set_parameter!(c::StopWhenGradientNormLess{F,TF}, :MinGradNorm, v::TF) where {F,TF<:Real}

Update the minimal gradient norm when an algorithm shall stop.
"""
function set_parameter!(c::StopWhenGradientNormLess{F, TF}, ::Val{:MinGradNorm}, v::TF) where {F, TF <: Real}
    c.threshold = v
    return c
end

#
#
# ---
"""
    StopWhenIterateNaN <: StoppingCriterion

Stop the solver when the iterate of the optimization problem from within an
[`AbstractManoptProblem`](@ref) contains `NaN` values.
The value is obtained using [`get_iterate`](@ref)`(s)`.

# Constructor

    StopWhenIterateNaN()

Initialize the stopping criterion.
"""
mutable struct StopWhenIterateNaN <: StoppingCriterion
    at_iteration::Int
    StopWhenIterateNaN() = new(-1)
end
function (c::StopWhenIterateNaN)(::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if (k >= 0) && any(isnan.(get_iterate(s)))
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenIterateNaN)
    if (c.at_iteration >= 0)
        return "The algorithm reached an iterate containing NaNs.\n"
    end
    return ""
end
indicates_convergence(c::StopWhenIterateNaN) = false
function status_summary(c::StopWhenIterateNaN; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "An entry of x is NaN:$(_MANOPT_INDENT)" : "A stopping criterion to stop when an entry of the iterate is NaN\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, ::StopWhenIterateNaN)
    return print(io, "StopWhenIterateNaN()")
end
requires_update(::Type{StopWhenIterateNaN}) = false

#
#
# ---
@doc """
    StopWhenLagrangeMultiplierLess <: StoppingCriterion

A stopping criterion for Lagrange multipliers.

Currently this is meant for the [`convex_bundle_method`](@ref) and [`proximal_bundle_method`](@ref),
where based on the Lagrange multipliers an approximate (sub)gradient ``g`` and an error estimate ``ε``
are computed.

The `mode=:both` requires that both
``ε`` and ``$(_tex(:abs, "g"))`` are smaller than their `tolerance`s for the [`convex_bundle_method`](@ref),
and that
``c`` and ``$(_tex(:abs, "d"))`` are smaller than their `tolerance`s for the [`proximal_bundle_method`](@ref).

The `mode=:estimate` requires that, for the [`convex_bundle_method`](@ref)
``-ξ = $(_tex(:abs, "g"))^2 + ε`` is less than a given `tolerance`.
For the [`proximal_bundle_method`](@ref), the equation reads ``-ν = μ $(_tex(:abs, "d"))^2 + c``.

# Constructors

    StopWhenLagrangeMultiplierLess(tolerance=1e-6; mode::Symbol=:estimate, names=nothing)

Create the stopping criterion for one of the `mode`s mentioned.
Note that tolerance can be a single number for the `:estimate` case,
but a vector of two values is required for the `:both` mode.
Here the first entry specifies the tolerance for ``ε`` (``c``),
the second the tolerance for ``$(_tex(:abs, "g"))`` (``$(_tex(:abs, "d"))``), respectively.

# Fields

* `tolerances`: the tolerances to check against
* `values`: the last values that were compared against the `tolerances`
* `names`: optional names for the `values`, used when reporting the reason
* `mode`: either `:estimate` or `:both`, see above
* `at_iteration`: the iteration at which this criterion indicated to stop, `-1` otherwise
"""
mutable struct StopWhenLagrangeMultiplierLess{
        T <: Real, A <: AbstractVector{<:T}, B <: Union{Nothing, <:AbstractVector{<:String}},
    } <: StoppingCriterion
    tolerances::A
    values::A
    names::B
    mode::Symbol
    at_iteration::Int
    function StopWhenLagrangeMultiplierLess(
            tol::Real = 1.0e-6; mode::Symbol = :estimate, names::B = nothing
        ) where {B <: Union{Nothing, <:AbstractVector{<:String}}}
        t = float(tol)
        return new{typeof(t), Vector{typeof(t)}, B}([t], zero([t]), names, mode, -1)
    end
    function StopWhenLagrangeMultiplierLess(
            tols::AbstractVector{<:Real}; mode::Symbol = :estimate, names::B = nothing
        ) where {B <: Union{Nothing, <:AbstractVector{<:String}}}
        t = float(tols)
        return new{eltype(t), typeof(t), B}(t, zero(t), names, mode, -1)
    end
end
function get_reason(sc::StopWhenLagrangeMultiplierLess)
    if (sc.at_iteration >= 0)
        if isnothing(sc.names)
            tol_str = join(
                ["$ai < $bi" for (ai, bi) in zip(sc.values, sc.tolerances)], ", "
            )
        else
            tol_str = join(
                [
                    "$si = $ai < $bi" for
                        (si, ai, bi) in zip(sc.names, sc.values, sc.tolerances)
                ],
                ", ",
            )
        end
        return "After $(sc.at_iteration) iterations the algorithm reached an approximate critical point with tolerances $tol_str.\n"
    end
    return ""
end
function status_summary(sc::StopWhenLagrangeMultiplierLess; context::Symbol = :default)
    s = (sc.at_iteration >= 0) ? "reached" : "not reached"
    msg = "Lagrange multipliers"
    isnothing(sc.names) && (msg *= " with tolerances $(sc.tolerances)")
    if !isnothing(sc.names)
        msg *= " " * join(["$si < $bi" for (si, bi) in zip(sc.names, sc.tolerances)], ", ")
    end
    return (_is_inline(context) ? "" : "A stopping criterion to stop when the Lagrange multipliers are less than $(sc.tolerances).\n$(_MANOPT_INDENT)") * "$(msg):$(_MANOPT_INDENT)$(s)"
end
function show(io::IO, sc::StopWhenLagrangeMultiplierLess)
    n = isnothing(sc.names) ? "" : ", $(sc.names)"
    return print(
        io,
        "StopWhenLagrangeMultiplierLess($(sc.tolerances); mode=:$(sc.mode)$n)",
    )
end
requires_update(::Type{<:StopWhenLagrangeMultiplierLess}) = false

#
#
# ---
@doc """
    StopWhenRepeated <: StoppingCriterion

A stopping criterion that indicates to stop when the (internal) stopping criterion it wraps
has indicated to stop for `n` (consecutive) times.

# Fields

* `stopping_criterion`: the [`StoppingCriterion`](@ref) to wrap
* `n`: the number of times the criterion has to indicate to stop
* `count`: the number of times the criterion has indicated to stop so far
* `consecutive::Bool`: indicate whether to count consecutive indications to stop or arbitrary.
* `at_iteration`: the iteration at which this criterion indicated to stop, `-1` otherwise

# Constructor

    StopWhenRepeated(criterion::StoppingCriterion, n::Int; consecutive::Bool=true)
    criterion × n
    cross(sc::StoppingCriterion, n::Int)

Create a stopping criterion that indicates to stop when the `criterion` has indicated to stop
`n` times (consecutively, if `consecutive=true` for the first constructor).
Note that the cross product is in general noncommutative, and here only the order `sc × n` is possible.

# Examples

A stopping criterion that indicates to stop whenever the gradient norm is less than `1e-6` for three consecutive iterations:

    StopWhenRepeated(StopWhenGradientNormLess(1e-6), 3)
    StopWhenGradientNormLess(1e-6) × 3

A stopping criterion that indicates to stop whenever the gradient norm is less than `1e-6` at three iterations (not necessarily consecutive):

    StopWhenRepeated(StopWhenGradientNormLess(1e-6), 3; consecutive=false)
"""
mutable struct StopWhenRepeated{SC <: StoppingCriterion} <: StoppingCriterion
    stopping_criterion::SC
    n::Int
    count::Int
    consecutive::Bool
    at_iteration::Int
end
function StopWhenRepeated(
        sc::SC, n::Int; consecutive::Bool = true
    ) where {SC <: StoppingCriterion}
    return StopWhenRepeated{SC}(sc, n, 0, consecutive, -1)
end
function cross(sc::StoppingCriterion, n::Int)
    return StopWhenRepeated(sc, n)
end

function (c::StopWhenRepeated)(
        p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    if k <= 0 # reset on init
        c.count = zero(c.count)
        c.at_iteration = -1
    end
    # evaluate the inner stopping criterion
    stop = c.stopping_criterion(p, s, k)
    if stop # if we indicated to stop
        c.count += 1
        if c.count >= c.n # if it now fired n times (consecutively)
            c.at_iteration = k
            return true
        end
    else
        c.consecutive && (c.count = 0) # reset the count
    end
    return false
end
function get_reason(sc::StopWhenRepeated)
    has_stopped = (sc.at_iteration >= 0)
    if (sc.at_iteration >= 0)
        s = has_stopped ? "reached" : "not reached"
        c = sc.consecutive ? " consecutive" : ""
        # we can only get the last reason, unless we do more allocations
        r = """At iteration $(sc.at_iteration), the stopping criterion $(typeof(sc.stopping_criterion)) has indicated to stop $(sc.n)$(c) times:
        $(sc.count) ≥ $(sc.n): $(s)
        last inner criterion status:
        $(_in_str(status_summary(sc.stopping_criterion); indent = 1, headers = 0))
        """
        return r
    end
    return ""
end
function indicates_convergence(sc::StopWhenRepeated)
    return indicates_convergence(sc.stopping_criterion)
end
function has_converged(sc::StopWhenRepeated)
    # When the inner one indicates convergence, this does as well
    return has_converged(sc.stopping_criterion)
end
function Base.show(io::IO, sc::StopWhenRepeated)
    return print(io, "StopWhenRepeated($(typeof(sc.stopping_criterion)), $(sc.n); consecutive=$(sc.consecutive))")
end
function status_summary(sc::StopWhenRepeated; context::Symbol = :default)
    (context == :short) && return "StopWhenRepeated($(repr(sc.stopping_criterion)))×$(sc.n)"
    has_stopped = (sc.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    c = sc.consecutive ? " consecutive" : ""
    return (_is_inline(context) ? "$(status_summary(sc.stopping_criterion; context = context)) × $(sc.count) ≥ $(sc.n)$(c):$(_MANOPT_INDENT)$(s)" : "A stopping criterion to stop when the inner criterion has indicated to stop $(sc.n)$(c) times.\n$(_in_str(status_summary(sc.stopping_criterion; context = context); indent = 1, headers = 0))\n$(_in_str(s; indent = 2, headers = 0))")
end

#
#
# ---
"""
    StopWhenProjectedNegativeGradientNormLess <: StoppingCriterion

A stopping criterion similar to [`StopWhenGradientNormLess`](@ref), although it checks the
norm of the projected negative gradient. It is primarily useful for optimization involving
[`Hyperrectangle`](@extref Manifolds.Hyperrectangle).

# Fields

* `norm`:       a function `(M::AbstractManifold, p, X) -> ℝ` computing the norm to use
* `threshold`:  the threshold to indicate to stop when the norm is below this value
* `last_change`: the last norm recorded in this stopping criterion
* `at_iteration`: the iteration at which this criterion indicated to stop, `-1` otherwise
* `outer_norm`: for manifolds with components, the norm used to combine the element-wise norms

On manifolds with boundary and manifolds with corners, for a tangent vector ``X``,
``-X`` might not be a valid tangent vector. As an example, consider the objective
``f(x)=x^2`` on the interval ``[1, 2]``. Its gradient at 1 is equal to 2, but because the
point 1 is at the boundary of the interval, the projected negative gradient is equal to 0
because we can't go in the negative direction.
"""
mutable struct StopWhenProjectedNegativeGradientNormLess{F, TF <: Real, N <: Union{Missing, Real}} <: StoppingCriterion
    norm::F
    threshold::TF
    last_change::TF
    at_iteration::Int
    outer_norm::N
    function StopWhenProjectedNegativeGradientNormLess(
            ε::Real; norm::F = norm, outer_norm::N = missing
        ) where {F, N <: Union{Missing, Real}}
        e = float(ε)
        return new{F, typeof(e), N}(norm, e, zero(e), -1, outer_norm)
    end
end
function (sc::StopWhenProjectedNegativeGradientNormLess)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    M = get_manifold(mp)
    if k == 0 # reset on init
        sc.at_iteration = -1
    end
    if (k > 0)
        r = (has_components(M) && !ismissing(sc.outer_norm)) ? (sc.outer_norm,) : ()
        p = get_iterate(s)
        mpg = -get_gradient(s)
        embed_project!(M, mpg, p, mpg)
        sc.last_change = sc.norm(M, p, mpg, r...)
        if sc.last_change < sc.threshold
            sc.at_iteration = k
            return true
        end
    end
    return false
end
function get_reason(c::StopWhenProjectedNegativeGradientNormLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the projected negative gradient norm ($(c.last_change)) is less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenProjectedNegativeGradientNormLess; context::Symbol = :default)
    (context === :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    (context === :inline) && return "|proj (-grad f)| < $(c.threshold): $s"
    return "A stopping criterion to stop when the projected negative gradient norm is less than a threshold of $(c.threshold):\n$(_MANOPT_INDENT)$s"
end
indicates_convergence(c::StopWhenProjectedNegativeGradientNormLess) = true
requires_update(::Type{<:StopWhenProjectedNegativeGradientNormLess}) = false
function Base.show(io::IO, c::StopWhenProjectedNegativeGradientNormLess)
    return print(io, "StopWhenProjectedNegativeGradientNormLess($(c.threshold); norm = $(c.norm))")
end
"""
    set_parameter!(c::StopWhenProjectedNegativeGradientNormLess{F,TF}, :MinGradNorm, v::TF) where {F, TF<:Real}

Update the minimal gradient norm when an algorithm shall stop.
"""
function set_parameter!(c::StopWhenProjectedNegativeGradientNormLess{F, TF}, ::Val{:MinGradNorm}, v::TF) where {F, TF <: Real}
    c.threshold = v
    return c
end

#
#
# ---
"""
    StopWhenRelativeAPosterioriCostChangeLessOrEqual <: StoppingCriterion

A stopping criterion to stop when

````math
\\frac{f_k - f_{k+1}}{\\max(\\lvert f_k \\rvert, \\lvert f_{k+1} \\rvert, 1)} ≤ tol,
````

based on Eq. (1) in [ZhuByrdLuNocedal:1997](@cite).

# Fields
* `threshold`: the threshold `tol` in the above formula.
$(_fields([:at_iteration, :last_change]))
* `last_cost`: the last cost value

# Constructor

    StopWhenRelativeAPosterioriCostChangeLessOrEqual(threshold::F)

Initialize the stopping criterion to a `threshold` for the change of the cost function.

    StopWhenRelativeAPosterioriCostChangeLessOrEqual(; factr::Real=1.0e7)

Initialize `threshold` to `factr * eps(typeof(factr))`, following the convention in [ZhuByrdLuNocedal:1997](@cite).
"""
mutable struct StopWhenRelativeAPosterioriCostChangeLessOrEqual{F <: Real} <: StoppingCriterion
    threshold::F
    at_iteration::Int
    last_cost::F
    last_change::F
end
function StopWhenRelativeAPosterioriCostChangeLessOrEqual(tol::Real)
    t = float(tol)
    return StopWhenRelativeAPosterioriCostChangeLessOrEqual{typeof(t)}(t, -1, zero(t), 2 * t)
end
StopWhenRelativeAPosterioriCostChangeLessOrEqual(; factr::F = 1.0e7) where {F <: Real} = StopWhenRelativeAPosterioriCostChangeLessOrEqual(factr * eps(typeof(factr)))
function (c::StopWhenRelativeAPosterioriCostChangeLessOrEqual)(
        problem::AbstractManoptProblem, state::AbstractManoptSolverState, k::Int
    )
    if k <= 0 # reset on init
        c.at_iteration = -1
        c.last_cost = Inf
        c.last_change = 2 * c.threshold
    end
    current_cost = get_cost(problem, state)
    c.last_change = (c.last_cost - current_cost) / max(abs(c.last_cost), abs(current_cost), 1)
    c.last_cost = current_cost
    if k > 1 && c.last_change <= c.threshold
        c.at_iteration = k
        return true
    end
    return false
end
indicates_convergence(c::StopWhenRelativeAPosterioriCostChangeLessOrEqual) = false
function get_reason(c::StopWhenRelativeAPosterioriCostChangeLessOrEqual)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration) the algorithm performed a step with a relative a posteriori cost change ($(c.last_change)) less than or equal to $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenRelativeAPosterioriCostChangeLessOrEqual; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "(fₖ- fₖ₊₁)/max(|fₖ|, |fₖ₊₁|, 1) = $(c.last_change) ≤ $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the relative posteriori cost change is less than $(c.threshold)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopWhenRelativeAPosterioriCostChangeLessOrEqual)
    return print(io, "StopWhenRelativeAPosterioriCostChangeLessOrEqual($(c.threshold))")
end

#
#
# ---
@doc """
    StopWhenSmallerOrEqual <: StoppingCriterion

A functor for a stopping criterion, where the algorithm is stopped when a field of the solver
state is smaller than or equal to a given minimum value.

# Fields

* `value`:    a `Symbol` naming the field of the solver state that has to fall under the threshold
* `minValue`: the threshold; if the field's value is smaller than or equal to it, the algorithm stops
* `at_iteration`: the iteration at which this criterion indicated to stop, `-1` otherwise

# Constructor

    StopWhenSmallerOrEqual(value::Symbol, minValue)

initialize the functor to indicate to stop as soon as the field `value` is smaller than or
equal to `minValue`.
"""
mutable struct StopWhenSmallerOrEqual{R} <: StoppingCriterion
    value::Symbol
    minValue::R
    at_iteration::Int
    function StopWhenSmallerOrEqual(value::Symbol, mValue::R) where {R <: Real}
        return new{R}(value, mValue, -1)
    end
end
function (c::StopWhenSmallerOrEqual)(
        ::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if getfield(s, c.value) <= c.minValue
        c.at_iteration = k
        return true
    end
    return false
end
indicates_convergence(c::StopWhenSmallerOrEqual) = false
function get_reason(c::StopWhenSmallerOrEqual)
    if (c.at_iteration >= 0)
        return "The value of the variable ($(string(c.value))) is smaller than or equal to its threshold ($(c.minValue)).\n"
    end
    return ""
end
function status_summary(c::StopWhenSmallerOrEqual; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "Field :$(c.value) ≤ $(c.minValue):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the field :$(c.value) is smaller than or equal to $(c.minValue)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopWhenSmallerOrEqual)
    return print(io, "StopWhenSmallerOrEqual(:$(c.value), $(c.minValue))")
end
requires_update(::Type{<:StopWhenSmallerOrEqual}) = false
#
#
# ---
"""
    StopWhenStepsizeLess <: StoppingCriterion

Store a threshold when to stop, looking at the last step size determined or found
during the last iteration from within a [`AbstractManoptSolverState`](@ref).

# Fields

* `threshold`: the threshold below which the algorithm stops
* `last_stepsize`: the last step size recorded in this stopping criterion
* `at_iteration`: the iteration at which this criterion indicated to stop, `-1` otherwise

# Constructor

    StopWhenStepsizeLess(ε)

initialize the stopping criterion to a threshold `ε`.
"""
mutable struct StopWhenStepsizeLess{F} <: StoppingCriterion
    threshold::F
    last_stepsize::F
    at_iteration::Int
    function StopWhenStepsizeLess(ε::Real)
        e = float(ε)
        return new{typeof(e)}(e, zero(e), -1)
    end
end
function (c::StopWhenStepsizeLess)(
        p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    c.last_stepsize = get_last_stepsize(p, s, k)
    if c.last_stepsize < c.threshold && k > 0
        c.at_iteration = k
        return true
    end
    return false
end
indicates_convergence(c::StopWhenStepsizeLess) = false
function get_reason(c::StopWhenStepsizeLess)
    if (c.last_stepsize < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm computed a step size ($(c.last_stepsize)) less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenStepsizeLess; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "Stepsize s < $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the step size is less than $(c.threshold)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopWhenStepsizeLess)
    return print(io, "StopWhenStepsizeLess($(c.threshold))")
end
requires_update(::Type{<:StopWhenStepsizeLess}) = false
"""
    set_parameter!(c::StopWhenStepsizeLess, :MinStepsize, v)

Update the minimal step size below which the algorithm shall stop.
"""
function set_parameter!(c::StopWhenStepsizeLess, ::Val{:MinStepsize}, v)
    c.threshold = v
    return c
end

#
#
# ---
"""
    StopWhenSubgradientNormLess <: StoppingCriterion

A stopping criterion based on the current subgradient norm.

# Fields

* `at_iteration`: the iteration at which this criterion indicated to stop, `-1` otherwise
* `threshold`: the threshold below which the algorithm stops
* `value`: the last subgradient norm recorded in this stopping criterion

# Constructor

    StopWhenSubgradientNormLess(ε::Float64)

Create a stopping criterion with threshold `ε` for the subgradient, that is, this criterion
indicates to stop when [`get_subgradient`](@ref) returns a subgradient vector of norm less than `ε`.
"""
mutable struct StopWhenSubgradientNormLess{R} <: StoppingCriterion
    at_iteration::Int
    threshold::R
    value::R
    function StopWhenSubgradientNormLess(ε::Real)
        e = float(ε)
        return new{typeof(e)}(-1, e, zero(e))
    end
end
function (c::StopWhenSubgradientNormLess)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    M = get_manifold(mp)
    if (k == 0) # reset on init
        c.at_iteration = -1
    end
    c.value = norm(M, get_iterate(s), get_subgradient(s))
    if (c.value < c.threshold) && (k > 0)
        c.at_iteration = k
        return true
    end
    return false
end
indicates_convergence(c::StopWhenSubgradientNormLess) = true
requires_update(::Type{<:StopWhenSubgradientNormLess}) = false
function get_reason(c::StopWhenSubgradientNormLess)
    if (c.value < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the subgradient norm ($(c.value)) is less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenSubgradientNormLess; context::Symbol = :default)
    (context == :short) && return repr(c)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "|∂f| < $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the subgradient norm |∂f| is less than $(c.threshold)\n$(_MANOPT_INDENT)") * "$s"
end
function Base.show(io::IO, c::StopWhenSubgradientNormLess)
    return print(io, "StopWhenSubgradientNormLess($(c.threshold))")
end
"""
    set_parameter!(c::StopWhenSubgradientNormLess, :MinSubgradNorm, v::Float64)

Update the minimal subgradient norm below which an algorithm shall stop.
"""
function set_parameter!(c::StopWhenSubgradientNormLess, ::Val{:MinSubgradNorm}, v::Float64)
    c.threshold = v
    return c
end
