#
# This file provides a systematic way to state stopping criteria employing functors
#
export stopAfterIteration, stopWhenChangeLess, stopWhenGradientNormLess
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
* `reason` – stores a reason of stopping if the stoppingcriterion has one be reached,
    see [`getReason`](@ref).

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

stores a threshold when to stop looking at the norm of the change of the optimization variable
from within a [`Options`](@ref).
"""
mutable struct stopWhenChangeLess <: StoppingCriterion
    threshold::Float64
    reason::String
    xOld::MPoint
    stopWhenChangeLess(ε::Float64) = new(ε,"")
end
function (c::stopWhenChangeLess)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if i==0 # init
        c.xOld = o.x
    elseif !isdefined(c,:xOld)
        c.xOld = o.xOld
    else
        if distance(p.M, o.x, c.xOld) < c.threshold && i>0
            c.reason = "The algorithm performed a step with a change ($(distance(p.M, o.x, c.xOld))) less than $(c.threshold).\n"
            return true
        end
    end
    return false
end
@doc doc"""
    stopWhenAll <: StoppingCriterion

stores an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
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

stores an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
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
    stopWhenAny(c...) = new([c...],"")
end
function (c::stopWhenAny)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if any([ subC(p,o,i) for subC in c.criteria])
        c.reason = string( [ getReason(subC) for subC in c.criteria ]... )
        return true
    end
    return false
end