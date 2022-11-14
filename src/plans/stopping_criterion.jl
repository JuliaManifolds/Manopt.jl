#
# Stopping Criteria
#

@doc raw"""
    StopAfterIteration <: StoppingCriterion

A functor for an easy stopping criterion, i.e. to stop after a maximal number
of iterations.

# Fields
* `maxIter` – stores the maximal iteration number where to stop at
* `reason` – stores a reason of stopping if the stopping criterion has one be
  reached, see [`get_reason`](@ref).

# Constructor

    StopAfterIteration(maxIter)

initialize the stopafterIteration functor to indicate to stop after `maxIter`
iterations.
"""
mutable struct StopAfterIteration <: StoppingCriterion
    maxIter::Int
    reason::String
    StopAfterIteration(mIter::Int) = new(mIter, "")
end
function (c::StopAfterIteration)(::P, ::O, i::Int) where {P<:Problem,O<:Options}
    (i == 0) && (c.reason = "") # reset on init
    if i >= c.maxIter
        c.reason = "The algorithm reached its maximal number of iterations ($(c.maxIter)).\n"
        return true
    end
    return false
end

"""
    update_stopping_criterion!(c::StopAfterIteration, :;MaxIteration, v::Int)

Update the number of iterations after which the algorithm should stop.
"""
function update_stopping_criterion!(c::StopAfterIteration, ::Val{:MaxIteration}, v::Int)
    c.maxIter = v
    return c
end

"""
    StopWhenGradientNormLess <: StoppingCriterion

stores a threshold when to stop looking at the norm of the gradient from within
a [`GradientProblem`](@ref).
"""
mutable struct StopWhenGradientNormLess <: StoppingCriterion
    threshold::Float64
    reason::String
    StopWhenGradientNormLess(ε::Float64) = new(ε, "")
end
function (c::StopWhenGradientNormLess)(p::Problem, o::Options, i::Int)
    (i == 0) && (c.reason = "") # reset on init
    if norm(p.M, get_iterate(o), get_gradient(o)) < c.threshold
        c.reason = "The algorithm reached approximately critical point after $i iterations; the gradient norm ($(norm(p.M,get_iterate(o),get_gradient(o)))) is less than $(c.threshold).\n"
        return true
    end
    return false
end

"""
    update_stopping_criterion!(c::StopWhenGradientNormLess, :MinGradNorm, v::Float64)

Update the minimal gradient norm when an algorithm shall stop
"""
function update_stopping_criterion!(
    c::StopWhenGradientNormLess, ::Val{:MinGradNorm}, v::Float64
)
    c.threshold = v
    return c
end

"""
    StopWhenChangeLess <: StoppingCriterion

stores a threshold when to stop looking at the norm of the change of the
optimization variable from within a [`Options`](@ref), i.e `get_iterate(o)`.
For the storage a [`StoreOptionsAction`](@ref) is used

# Constructor

    StopWhenChangeLess(ε[, a])

initialize the stopping criterion to a threshold `ε` using the
[`StoreOptionsAction`](@ref) `a`, which is initialized to just store `:Iterate` by
default.
"""
mutable struct StopWhenChangeLess <: StoppingCriterion
    threshold::Float64
    reason::String
    storage::StoreOptionsAction
    function StopWhenChangeLess(
        ε::Float64, a::StoreOptionsAction=StoreOptionsAction((:Iterate,))
    )
        return new(ε, "", a)
    end
end
function (c::StopWhenChangeLess)(P::Problem, O::Options, i)
    (i == 0) && (c.reason = "") # reset on init
    if has_storage(c.storage, :Iterate)
        x_old = get_storage(c.storage, :Iterate)
        d = distance(P.M, get_iterate(O), x_old, default_inverse_retraction_method(P.M))
        if d < c.threshold && i > 0
            c.reason = "The algorithm performed a step with a change ($d) less than $(c.threshold).\n"
            c.storage(P, O, i)
            return true
        end
    end
    c.storage(P, O, i)
    return false
end

"""
    update_stopping_criterion!(c::StopWhenChangeLess, :MinIterateChange, v::Int)

Update the minimal change blow which an algorithm shall stop.
"""
function update_stopping_criterion!(c::StopWhenChangeLess, ::Val{:MinIterateChange}, v)
    c.threshold = v
    return c
end

"""
    StopWhenStepsizeLess <: StoppingCriterion

stores a threshold when to stop looking at the last step size determined or found
during the last iteration from within a [`Options`](@ref).

# Constructor

    StopWhenStepsizeLess(ε)

initialize the stopping criterion to a threshold `ε`.
"""
mutable struct StopWhenStepsizeLess <: StoppingCriterion
    threshold::Float64
    reason::String
    function StopWhenStepsizeLess(ε::Float64)
        return new(ε, "")
    end
end
function (c::StopWhenStepsizeLess)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    (i == 0) && (c.reason = "") # reset on init
    s = get_last_stepsize(p, o, i)
    if s < c.threshold && i > 0
        c.reason = "The algorithm computed a step size ($s) less than $(c.threshold).\n"
        return true
    end
    return false
end

"""
    update_stopping_criterion!(c::StopWhenStepsizeLess, :MinStepsize, v)

Update the minimal step size below which the slgorithm shall stop
"""
function update_stopping_criterion!(c::StopWhenStepsizeLess, ::Val{:MinStepsize}, v)
    c.threshold = v
    return c
end

"""
    StopWhenCostLess <: StoppingCriterion

store a threshold when to stop looking at the cost function of the
optimization problem from within a [`Problem`](@ref), i.e `get_cost(p,get_iterate(o))`.

# Constructor

    StopWhenCostLess(ε)

initialize the stopping criterion to a threshold `ε`.
"""
mutable struct StopWhenCostLess <: StoppingCriterion
    threshold::Float64
    reason::String
    StopWhenCostLess(ε::Float64) = new(ε, "")
end
function (c::StopWhenCostLess)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    (i == 0) && (c.reason = "") # reset on init
    if i > 0 && get_cost(p, get_iterate(o)) < c.threshold
        c.reason = "The algorithm reached a cost function value ($(get_cost(p,get_iterate(o)))) less than the threshold ($(c.threshold)).\n"
        return true
    end
    return false
end

"""
    update_stopping_criterion!(c::StopWhenCostLess, :MinCost, v)

Update the minimal cost below which the slgorithm shall stop
"""
function update_stopping_criterion!(c::StopWhenCostLess, ::Val{:MinCost}, v)
    c.threshold = v
    return c
end

@doc raw"""
    StopWhenSmallerOrEqual <: StoppingCriterion

A functor for an stopping criterion, where the algorithm if stopped when a variable is smaller than or equal to its minimum value.

# Fields
* `value` – stores the variable which has to fall under a threshold for the algorithm to stop
* `minValue` – stores the threshold where, if the value is smaller or equal to this threshold, the algorithm stops
* `reason` – stores a reason of stopping if the stopping criterion has one be
  reached, see [`get_reason`](@ref).

# Constructor

    StopWhenSmallerOrEqual(value, minValue)

initialize the stopifsmallerorequal functor to indicate to stop after `value` is smaller than or equal to `minValue`.
"""
mutable struct StopWhenSmallerOrEqual <: StoppingCriterion
    value::Symbol
    minValue::Real
    reason::String
    StopWhenSmallerOrEqual(value::Symbol, mValue::Real) = new(value, mValue, "")
end
function (c::StopWhenSmallerOrEqual)(::P, o::O, i::Int) where {P<:Problem,O<:Options}
    (i == 0) && (c.reason = "") # reset on init
    if getfield(o, c.value) <= c.minValue
        c.reason = "The value of the variable ($(string(c.value))) is smaller than or equal to its threshold ($(c.minValue)).\n"
        return true
    end
    return false
end

"""
    StopAfter <: StoppingCriterion

store a threshold when to stop looking at the complete runtime. It uses
`time_ns()` to measure the time and you provide a `Period` as a time limit,
i.e. `Minute(15)`

# Constructor

    StopAfter(t)

initialize the stopping criterion to a `Period t` to stop after.
"""
mutable struct StopAfter <: StoppingCriterion
    threshold::Period
    reason::String
    start::Nanosecond
    function StopAfter(t::Period)
        return if value(t) < 0
            error("You must provide a positive time period")
        else
            new(t, "", Nanosecond(0))
        end
    end
end
function (c::StopAfter)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    if value(c.start) == 0 || i <= 0 # (re)start timer
        c.reason = ""
        c.start = Nanosecond(time_ns())
    else
        cTime = Nanosecond(time_ns()) - c.start
        if i > 0 && (cTime > Nanosecond(c.threshold))
            c.reason = "The algorithm ran for about $(floor(cTime, typeof(c.threshold))) and has hence reached the threshold of $(c.threshold).\n"
            return true
        end
    end
    return false
end

"""
    update_stopping_criterion!(c::StopAfter, :MaxTime, v::Period)

Update the time period after which an algorithm shall stop.
"""
function update_stopping_criterion!(c::StopAfter, ::Val{:MaxTime}, v::Period)
    (value(v) < 0) && error("You must provide a positive time period")
    c.threshold = v
    return c
end

#
# Meta Criteria
#

@doc raw"""
    StopWhenAll <: StoppingCriterion

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _all_ indicate to stop. The `reason` is given by the concatenation of all
reasons.

# Constructor
    StopWhenAll(c::NTuple{N,StoppingCriterion} where N)
    StopWhenAll(c::StoppingCriterion,...)
"""
mutable struct StopWhenAll{TCriteria<:Tuple} <: StoppingCriterionSet
    criteria::TCriteria
    reason::String
    StopWhenAll(c::Vector{StoppingCriterion}) = new{typeof(tuple(c...))}(tuple(c...), "")
    StopWhenAll(c...) = new{typeof(c)}(c, "")
end
function (c::StopWhenAll)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    (i == 0) && (c.reason = "") # reset on init
    if all(subC -> subC(p, o, i), c.criteria)
        c.reason = string([get_reason(subC) for subC in c.criteria]...)
        return true
    end
    return false
end

"""
    &(s1,s2)
    s1 & s2

Combine two [`StoppingCriterion`](@ref) within an [`StopWhenAll`](@ref).
If either `s1` (or `s2`) is already an [`StopWhenAll`](@ref), then `s2` (or `s1`) is
appended to the list of [`StoppingCriterion`](@ref) within `s1` (or `s2`).

# Example
    a = StopAfterIteration(200) & StopWhenChangeLess(1e-6)
    b = a & StopWhenGradientNormLess(1e-6)

Is the same as

    a = StopWhenAll(StopAfterIteration(200), StopWhenChangeLess(1e-6))
    b = StopWhenAll(StopAfterIteration(200), StopWhenChangeLess(1e-6), StopWhenGradientNormLess(1e-6))
"""
function Base.:&(s1::S, s2::T) where {S<:StoppingCriterion,T<:StoppingCriterion}
    return StopWhenAll(s1, s2)
end
function Base.:&(s1::S, s2::StopWhenAll) where {S<:StoppingCriterion}
    return StopWhenAll(s1, s2.criteria...)
end
function Base.:&(s1::StopWhenAll, s2::T) where {T<:StoppingCriterion}
    return StopWhenAll(s1.criteria..., s2)
end

@doc raw"""
    StopWhenAny <: StoppingCriterion

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _any_ single one indicates to stop. The `reason` is given by the
concatenation of all reasons (assuming that all non-indicating return `""`).

# Constructor
    StopWhenAny(c::NTuple{N,StoppingCriterion} where N)
    StopWhenAny(c::StoppingCriterion...)
"""
mutable struct StopWhenAny{TCriteria<:Tuple} <: StoppingCriterionSet
    criteria::TCriteria
    reason::String
    StopWhenAny(c::Vector{StoppingCriterion}) = new{typeof(tuple(c...))}(tuple(c...), "")
    StopWhenAny(c::StoppingCriterion...) = new{typeof(c)}(c)
end
function (c::StopWhenAny)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    (i == 0) && (c.reason = "") # reset on init
    if any(subC -> subC(p, o, i), c.criteria)
        c.reason = string([get_reason(subC) for subC in c.criteria]...)
        return true
    end
    return false
end

"""
    |(s1,s2)
    s1 | s2

Combine two [`StoppingCriterion`](@ref) within an [`StopWhenAny`](@ref).
If either `s1` (or `s2`) is already an [`StopWhenAny`](@ref), then `s2` (or `s1`) is
appended to the list of [`StoppingCriterion`](@ref) within `s1` (or `s2`)

# Example
    a = StopAfterIteration(200) | StopWhenChangeLess(1e-6)
    b = a | StopWhenGradientNormLess(1e-6)

Is the same as

    a = StopWhenAny(StopAfterIteration(200), StopWhenChangeLess(1e-6))
    b = StopWhenAny(StopAfterIteration(200), StopWhenChangeLess(1e-6), StopWhenGradientNormLess(1e-6))
"""
function Base.:|(s1::S, s2::T) where {S<:StoppingCriterion,T<:StoppingCriterion}
    return StopWhenAny(s1, s2)
end
function Base.:|(s1::S, s2::StopWhenAny) where {S<:StoppingCriterion}
    return StopWhenAny(s1, s2.criteria...)
end
function Base.:|(s1::StopWhenAny, s2::T) where {T<:StoppingCriterion}
    return StopWhenAny(s1.criteria..., s2)
end

#
# Functions for Criteria
#
"""
    are_these_stopping_critera_active(c::StoppingCriterion, cond)

Return `true` if any criterion from the given set is both active and fulfils the given
condition `cond` (`cond(c)` returns `true`).
"""
function are_these_stopping_critera_active(cond::Function, c::StoppingCriterionSet)
    return any(cc -> are_these_stopping_critera_active(cond, cc), get_stopping_criteria(c))
end
function are_these_stopping_critera_active(cond::Function, c::StoppingCriterion)
    return !isempty(c.reason) && cond(c)
end

@doc raw"""
    get_active_stopping_criteria(c)

returns all active stopping criteria, if any, that are within a
[`StoppingCriterion`](@ref) `c`, and indicated a stop, i.e. their reason is
nonempty.
To be precise for a simple stopping criterion, this returns either an empty
array if no stop is indicated or the stopping criterion as the only element of
an array. For a [`StoppingCriterionSet`](@ref) all internal (even nested)
criteria that indicate to stop are returned.
"""
function get_active_stopping_criteria(c::sCS) where {sCS<:StoppingCriterionSet}
    c = get_active_stopping_criteria.(get_stopping_criteria(c))
    return vcat(c...)
end
# for non-array containing stopping criteria, the recursion ends in either
# returning nothing or an 1-element array containing itself
function get_active_stopping_criteria(c::sC) where {sC<:StoppingCriterion}
    if c.reason != ""
        return [c] # recursion top
    else
        return []
    end
end

@doc raw"""
    get_reason(c)

return the current reason stored within a [`StoppingCriterion`](@ref) `c`.
This reason is empty if the criterion has never been met.
"""
get_reason(c::sC) where {sC<:StoppingCriterion} = c.reason

@doc raw"""
    get_reason(o)

return the current reason stored within the [`StoppingCriterion`](@ref) from
within the [`Options`](@ref) This reason is empty if the criterion has never
been met.
"""
get_reason(o::Options) = get_reason(get_options(o).stop)

@doc raw"""
    get_stopping_criteria(c)

return the array of internally stored [`StoppingCriterion`](@ref)s for a
[`StoppingCriterionSet`](@ref) `c`.
"""
function get_stopping_criteria(c::S) where {S<:StoppingCriterionSet}
    return error("get_stopping_criteria() not defined for a $(typeof(c)).")
end
get_stopping_criteria(c::StopWhenAll) = c.criteria
get_stopping_criteria(c::StopWhenAny) = c.criteria

@doc raw"""
    update_stopping_criterion!(c::Stoppingcriterion, s::Symbol, v::value)
    update_stopping_criterion!(o::Options, s::Symbol, v::value)
    update_stopping_criterion!(c::Stoppingcriterion, ::Val{Symbol}, v::value)

Update a value within a stopping criterion, specified by the symbol `s`, to `v`.
If a criterion does not have a value assigned that corresponds to `s`, the update is ignored.

For the second signature, the stopping criterion within the [`Options`](@ref) `o` is updated.

To see which symbol updates which value, see the specific stopping criteria. They should
use dispatch per symbol value (the third signature).
"""
update_stopping_criterion!(c, s, v)

function update_stopping_criterion!(o::Options, s::Symbol, v)
    update_stopping_criterion!(o.stop, s, v)
    return o
end
function update_stopping_criterion!(c::StopWhenAll, s::Symbol, v)
    for d in c.criteria
        update_stopping_criterion!(d, s, v)
    end
    return c
end
function update_stopping_criterion!(c::StopWhenAny, s::Symbol, v)
    for d in c.criteria
        update_stopping_criterion!(d, s, v)
    end
    return c
end
function update_stopping_criterion!(c::StoppingCriterion, s::Symbol, v::Any)
    update_stopping_criterion!(c, Val(s), v)
    return c
end
# fallback: do nothing
function update_stopping_criterion!(c::StoppingCriterion, ::Val, v)
    return c
end
