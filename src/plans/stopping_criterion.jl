@doc raw"""
    get_reason(c)

return the current reason stored within a [`StoppingCriterion`](@ref) `c`.
This reason is empty if the criterion has never been met.
"""
get_reason(c::sC) where {sC<:StoppingCriterion} = c.reason

@doc raw"""
    get_stopping_criteria(c)
return the array of internally stored [`StoppingCriterion`](@ref)s for a
[`StoppingCriterionSet`](@ref) `c`.
"""
function get_stopping_criteria(c::S) where {S<:StoppingCriterionSet}
    return error("get_stopping_criteria() not defined for a $(typeof(c)).")
end

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
function (c::StopAfterIteration)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    if i > c.maxIter
        c.reason = "The algorithm reached its maximal number of iterations ($(c.maxIter)).\n"
        return true
    end
    return false
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
function (c::StopWhenGradientNormLess)(
    p::Problem, o::AbstractGradientOptions, iter::Int
)
    if norm(p.M, o.x, o.gradient) < c.threshold
        c.reason = "The algorithm reached approximately critical point after $iter iterations; the gradient norm ($(norm(p.M,o.x,o.gradient))) is less than $(c.threshold).\n"
        return true
    end
    return false
end
"""
    StopWhenChangeLess <: StoppingCriterion

stores a threshold when to stop looking at the norm of the change of the
optimization variable from within a [`Options`](@ref), i.e `o.x`.
For the storage a [`StoreOptionsAction`](@ref) is used

# Constructor

    StopWhenChangeLess(ε[, a])

initialize the stopping criterion to a threshold `ε` using the
[`StoreOptionsAction`](@ref) `a`, which is initialized to just store `:x` by
default.
"""
mutable struct StopWhenChangeLess <: StoppingCriterion
    threshold::Float64
    reason::String
    storage::StoreOptionsAction
    function StopWhenChangeLess(ε::Float64, a::StoreOptionsAction=StoreOptionsAction((:x,)))
        return new(ε, "", a)
    end
end
function (c::StopWhenChangeLess)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    if has_storage(c.storage, :x)
        xOld = get_storage(c.storage, :x)
        if distance(p.M, o.x, xOld) < c.threshold && i > 0
            c.reason = "The algorithm performed a step with a change ($(distance(p.M, o.x, xOld))) less than $(c.threshold).\n"
            c.storage(p, o, i)
            return true
        end
    end
    c.storage(p, o, i)
    return false
end
@doc raw"""
    StopWhenAll <: StoppingCriterion

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _all_ indicate to stop. The `reseason` is given by the concatenation of all
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
    if all(subC -> subC(p, o, i), c.criteria)
        c.reason = string([get_reason(subC) for subC in c.criteria]...)
        return true
    end
    return false
end
get_stopping_criteria(c::StopWhenAll) = c.criteria

@doc raw"""
    StopWhenAny <: StoppingCriterion

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _any_ single one indicates to stop. The `reseason` is given by the
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
    if any(subC -> subC(p, o, i), c.criteria)
        c.reason = string([get_reason(subC) for subC in c.criteria]...)
        return true
    end
    return false
end
get_stopping_criteria(c::StopWhenAny) = c.criteria
"""
    StopWhenCostLess <: StoppingCriterion

store a threshold when to stop looking at the cost function of the
optimization problem from within a [`Problem`](@ref), i.e `get_cost(p,o.x)`.

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
    if i > 0 && get_cost(p, o.x) < c.threshold
        c.reason = "The algorithm reached a cost function value ($(get_cost(p,o.x))) less then the threshold ($(c.threshold)).\n"
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
@doc raw"""
    get_active_stopping_criteria(c)

returns all active stopping criteria, if any, that are within a
[`StoppingCriterion`](@ref) `c`, and indicated a stop, i.e. their reason is
nonempty.
To be precise for a simple stopping criterion, this returns either an empty
array if no stop is incated or the stopping criterion as the only element of
an array. For a [`StoppingCriterionSet`](@ref) all internal (even nested)
criteria that indicate to stop are returned.
"""
function get_active_stopping_criteria(c::sCS) where {sCS<:StoppingCriterionSet}
    c = get_active_stopping_criteria.(get_stopping_criteria(c))
    return vcat(c...)
end
# for non-array containing stopping criteria, the recursion ends in either
# returning nothing or an 1-element array contianing itself
function get_active_stopping_criteria(c::sC) where {sC<:StoppingCriterion}
    if c.reason != ""
        return [c] # recursion top
    else
        return []
    end
end
