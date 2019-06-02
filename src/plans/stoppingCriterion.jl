#
# This file provides a systematic way to state stopping criteria employing functors
#
using Dates: Period, Nanosecond, value
export stopAfterIteration, stopWhenChangeLess, stopWhenGradientNormLess
export stopWhenCostLess, stopAfter
export stopWhenAll, stopWhenAny
export getReason
# defaults
@doc doc"""
    getReason(c)

return the current reason stored within a [`StoppingCriterion`](@ref) `c`.
This reason is empty if the criterion has never been met.
"""
getReason(c::sC) where sC <: StoppingCriterion = c.reason

@doc doc"""
    stopAfterIteration <: StoppingCriterion

A functor for an easy stopping criterion, i.e. to stop after a maximal number
of iterations.

# Fields
* `maxIter` – stores the maximal iteration number where to stop at
* `reason` – stores a reason of stopping if the stopping criterion has one be
  reached, see [`getReason`](@ref).

# Constructor

    stopAfterIteration(maxIter)

initialize the stopafterIteration functor to indicate to stop after `maxIter`
iterations.
"""
mutable struct stopAfterIteration <: StoppingCriterion
    maxIter::Int
    reason::String
    stopAfterIteration(mIter::Int) = new(mIter,"")
end
function (c::stopAfterIteration)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if i > c.maxIter
        c.reason = "The algorithm reached its maximal number of iterations ($(c.maxIter)).\n"
        return true
    end
    return false
end
"""
    stopWhenGradientNormLess <: StoppingCriterion

stores a threshold when to stop looking at the norm of the gradient from within
a [`GradientProblem`](@ref).
"""
mutable struct stopWhenGradientNormLess <: StoppingCriterion
    threshold::Float64
    reason::String
    stopWhenGradientNormLess(ε::Float64) = new(ε,"")
end
function (c::stopWhenGradientNormLess)(p::P,o::O,i::Int) where {P <: GradientProblem, O <: Options}
    if norm(p.M,o.x,getGradient(p,o.x)) < c.threshold
        c.reason = "The algorithm reached approximately critical point; the gradient norm ($(norm(p.M,o.x,getGradient(p,o.x)))) is less than $(c.threshold).\n"
        return true
    end
    return false
end
"""
    stopWhenChangeLess <: StoppingCriterion

stores a threshold when to stop looking at the norm of the change of the
optimization variable from within a [`Options`](@ref), i.e `o.x`.
For the storage a [`StoreOptionsAction`](@ref) is used

# Constructor

    stopWhenChangeLess(ε[, a])

initialize the stopping criterion to a threshold `ε` using the
[`StoreOptionsAction`](@ref) `a`, which is initialized to just store `:x` by
default.
"""
mutable struct stopWhenChangeLess <: StoppingCriterion
    threshold::Float64
    reason::String
    storage::StoreOptionsAction
    stopWhenChangeLess(ε::Float64, a::StoreOptionsAction=StoreOptionsAction( (:x,) )) = new(ε,"",a)
end
function (c::stopWhenChangeLess)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if hasStorage(c.storage,:x)
        xOld = getStorage(c.storage,:x)
        if distance(p.M, o.x, xOld) < c.threshold && i>0
            c.reason = "The algorithm performed a step with a change ($(distance(p.M, o.x, xOld))) less than $(c.threshold).\n"
            c.storage(p,o,i)
            return true
        end
    end
    c.storage(p,o,i)
    return false
end
@doc doc"""
    stopWhenAll <: StoppingCriterion

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _all_ indicate to stop. The `reseason` is given by the concatenation of all
reasons.

# Constructor
    stopWhenAll(c::Array{StoppingCriterion,1})
    stopWhenAll(c::StoppingCriterion,...)
"""
mutable struct stopWhenAll <: StoppingCriterion
    criteria::Array{StoppingCriterion,1}
    reason::String
    stopWhenAll(c::Array{StoppingCriterion,1}) = new(c,"")
    stopWhenAll(c...) = new([c...],"")
end
function (c::stopWhenAll)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if all([ subC(p,o,i) for subC in c.criteria])
        c.reason = string( [ getReason(subC) for subC in c.criteria ]... )
        return true
    end
    return false
end

@doc doc"""
    stopWhenAny <: StoppingCriterion

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _any_ single one indicates to stop. The `reseason` is given by the
concatenation of all reasons (assuming that all non-indicating return `""`).

# Constructor
    stopWhenAny(c::Array{StoppingCriterion,1})
    stopWhenAny(c::StoppingCriterion,...)
"""
mutable struct stopWhenAny <: StoppingCriterion
    criteria::Array{StoppingCriterion,1}
    reason::String
    stopWhenAny(c::Array{StoppingCriterion,1}) = new(c,"")
    stopWhenAny(c::StoppingCriterion...) = stopWhenAny([c...])
end
function (c::stopWhenAny)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if any([ subC(p,o,i) for subC in c.criteria])
        c.reason = string( [ getReason(subC) for subC in c.criteria ]... )
        return true
    end
    return false
end

"""
    stopWhenCostLess <: StoppingCriterion

store a threshold when to stop looking at the cost function of the
optimization problem from within a [`Problem`](@ref), i.e `getCost(p,o.x)`.

# Constructor

    stopWhenCostLess(ε)

initialize the stopping criterion to a threshold `ε`.
"""
mutable struct stopWhenCostLess <: StoppingCriterion
    threshold::Float64
    reason::String
    stopWhenCostLess(ε::Float64) = new(ε,"")
end
function (c::stopWhenCostLess)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if i > 0 && getCost(p,o.x) < c.threshold
        c.reason = "The algorithm reached a cost function value ($(getCost(p,o.x))) less then the threshold ($(c.threshold)).\n"
        return true
    end
    return false
end

"""
    stopAfter <: StoppingCriterion

store a threshold when to stop looking at the complete runtime. It uses
`time_ns()` to measure the time and you provide a `Period` as a time limit,
i.e. `Minute(15)`

# Constructor

    stopAfter(t)

initialize the stopping criterion to a `Period t` to stop after.
"""
mutable struct stopAfter <: StoppingCriterion
    threshold::Period
    reason::String
    start::Nanosecond
    stopAfter(t::Period) = value(t) < 0 ? error("You must provide a positive time period") : new(t,"", Nanosecond(0))
end
function (c::stopAfter)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if value(c.start) == 0 || i <= 0 # (re)start timer
        c.start = Nanosecond(time_ns())
    else
        cTime = Nanosecond(time_ns()) - c.start
        if i > 0 && ( cTime > Nanosecond(c.threshold) )
            c.reason = "The algorithm ran for about $(floor(cTime, typeof(c.threshold))) and has hence reached the threshold of $(c.threshold).\n"
            return true
        end
    end
    return false
end