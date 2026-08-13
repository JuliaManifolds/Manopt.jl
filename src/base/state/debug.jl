@doc """
    DebugAction

A `DebugAction` is a small functor to print/issue debug output. The usual call is given by
`(p::AbstractManoptProblem, s::AbstractManoptSolverState, k) -> s`, where `i` is
the current iterate.

By convention `i=0` is interpreted as "For Initialization only," only debug
info that prints initialization reacts, `i<0` triggers updates of variables
internally but does not trigger any output.

# Fields (assumed by subtypes to exist)
* `print` method to perform the actual print. Can for example be set to a file export,
or to @info. The default is the `print` function on the default `Base.stdout`.
"""
abstract type DebugAction <: AbstractStateAction end

@doc """
    DebugSolverState <: AbstractManoptSolverState

The debug state appends debug to any state, they act as a decorator pattern.
Internally a dictionary is kept that stores a [`DebugAction`](@ref) for several occasions
using a `Symbol` as reference.

The original options can still be accessed using the [`get_state`](@ref) function.

# Fields

* `options`:         the options that are extended by debug information
* `debug_dictionary`: a `Dict{Symbol,DebugAction}` to keep track of Debug for different actions

# Constructors
    DebugSolverState(o,dA)

construct debug decorated options, where `dD` can be
* a [`DebugAction`](@ref), then it is stored within the dictionary at `:Iteration`
* an `Array` of [`DebugAction`](@ref)s.
* a `Dict{Symbol,DebugAction}`.
* an Array of Symbols, String and an Int for the [`DebugFactory`](@ref)
"""
mutable struct DebugSolverState{S <: AbstractManoptSolverState} <: AbstractManoptSolverState
    state::S
    debug_dictionary::Dict{Symbol, <:DebugAction}
    function DebugSolverState{S}(
            st::S, dA::Dict{Symbol, <:DebugAction}
        ) where {S <: AbstractManoptSolverState}
        return new{S}(st, dA)
    end
end
function DebugSolverState(st::S, dD::D) where {S <: AbstractManoptSolverState, D <: DebugAction}
    return DebugSolverState{S}(st, Dict(:Iteration => dD))
end
function DebugSolverState(
        st::S, dD::Array{<:DebugAction, 1}
    ) where {S <: AbstractManoptSolverState}
    return DebugSolverState{S}(st, Dict(:Iteration => DebugGroup(dD)))
end
function DebugSolverState(
        st::S, dD::Dict{Symbol, <:DebugAction}
    ) where {S <: AbstractManoptSolverState}
    return DebugSolverState{S}(st, dD)
end
function DebugSolverState(
        st::S, format::Array{<:Any, 1}
    ) where {S <: AbstractManoptSolverState}
    return DebugSolverState{S}(st, DebugFactory(format))
end
function DebugSolverState( # a function: callback
        st::S, callback::Function,
    ) where {S <: AbstractManoptSolverState}
    return DebugSolverState{S}(st, DebugFactory([callback]))
end

"""
    set_parameter!(ams::DebugSolverState, ::Val{:Debug}, args...)

Set certain values specified by `args...` into the elements of the `debug_dictionary`
"""
function set_parameter!(dss::DebugSolverState, ::Val{:Debug}, args...)
    for d in values(dss.debug_dictionary)
        set_parameter!(d, args...)
    end
    return dss
end
# all other pass through
function set_parameter!(dss::DebugSolverState, v::Val{T}, args...) where {T}
    return set_parameter!(dss.state, v, args...)
end
function set_parameter!(dss::DebugSolverState, v::Val{:StoppingCriterion}, args...)
    return set_parameter!(dss.state, v, args...)
end
# all other pass through
function get_parameter(dss::DebugSolverState, v::Val{T}, args...) where {T}
    return get_parameter(dss.state, v, args...)
end

function status_summary(dst::DebugSolverState; context::Symbol = :default)
    if length(dst.debug_dictionary) > 0
        s = ""
        for (k, v) in dst.debug_dictionary
            s = "$s\n    :$k = $(status_summary(v; context = context))"
        end
        return "$(status_summary(dst.state; context = context))\n\n## Debug$s"
    else # if the dictionary has no entries, there is no actual debug in pretty print
        return status_summary(dst.state; context = context)
    end
end
function show(io::IO, dst::DebugSolverState)
    return print(io, "DebugSolverState($(dst.state), $(dst.debug_dictionary))")
end

dispatch_state_decorator(::DebugSolverState) = Val(true)

#
# Meta Debug Actions
#
"""
    DebugGroup <: DebugAction

group a set of [`DebugAction`](@ref)s into one action, where the internal prints
are removed by default and the resulting strings are concatenated

# Constructor

    DebugGroup(g)

construct a group consisting of an Array of [`DebugAction`](@ref)s `g`,
that are evaluated `en bloque`; the method does not perform any print itself,
but relies on the internal prints. It still concatenates the result and returns
the complete string
"""
mutable struct DebugGroup{D <: DebugAction} <: DebugAction
    group::Vector{D}
end
function (d::DebugGroup)(p::AbstractManoptProblem, st::AbstractManoptSolverState, k)
    for di in d.group
        di(p, st, k)
    end
    return
end
function status_summary(dg::DebugGroup; context::Symbol = :default)
    (context == :short) && return "[ " * join(["$(status_summary(di; context = context))" for di in dg.group], ", ") * " ]"
    (context == :inline) && return "A DebugAction consisting of a group actions, " * join(["$(status_summary(di; context = context))" for di in dg.group], ", ", ", and ")
    return """
    A DebugAction consisting of a group with the following elements
    $(join(["* $(status_summary(di; context = context))" for di in dg.group], "\n"))"""
end
function show(io::IO, dg::DebugGroup)
    s = join(["$(di)" for di in dg.group], ", ")
    return print(io, "DebugGroup([$s])")
end
function set_parameter!(dg::DebugGroup, v::Val, args...)
    for di in dg.group
        set_parameter!(di, v, args...)
    end
    return dg
end

@doc """
    DebugEvery <: DebugAction

evaluate and print debug only every ``k``th iteration. Otherwise no print is performed.
Whether internal variables are updates is determined by `always_update`.

This method does not perform any print itself but relies on it's children's print.

It also sets the sub solvers active parameter, see |`DebugWhenActive`}(#ref).
Here, the `activation_offset` can be used to specify whether it refers to _this_ iteration,
the `i`th, when this call is _before_ the iteration, then the offset should be 0,
for the _next_ iteration, that is if this is called _after_ an iteration, it has to be set to 1.
Since usual debug is happening after the iteration, 1 is the default.

# Constructor

    DebugEvery(d::DebugAction, every=1, always_update=true, activation_offset=1)
"""
mutable struct DebugEvery <: DebugAction
    debug::DebugAction
    every::Int
    always_update::Bool
    activation_offset::Int
    function DebugEvery(
            d::DebugAction, every::Int = 1, always_update::Bool = true; activation_offset = 1
        )
        return new(d, every, always_update, activation_offset)
    end
end
function (d::DebugEvery)(p::AbstractManoptProblem, st::AbstractManoptSolverState, k)
    if (rem(k, d.every) == 0)
        d.debug(p, st, k)
    elseif d.always_update
        d.debug(p, st, -1)
    end
    # set activity for this iterate in sub solvers
    set_parameter!(
        st, Val(:SubState), Val(:Debug), Val(:Activity),
        !(k < 1) && (rem(k + d.activation_offset, d.every) == 0),
    )
    return nothing
end
function show(io::IO, de::DebugEvery)
    return print(
        io,
        "DebugEvery($(de.debug), $(de.every), $(de.always_update); activation_offset=$(de.activation_offset))",
    )
end
function status_summary(de::DebugEvery; context::Symbol = :default)
    s = ""
    if context == :short
        s = status_summary(de.debug; context = context)
        # If we have a group, remove outer brackets and spaces
        (de.debug isa DebugGroup) && (s = s[3:(end - 2)])
        return "[$s, $(de.every)]"
    end
    (context == :inline) && return "The Debug $(status_summary(de.debug; context = context)) only printed every $(de.every) iteration"
    return "A DebugAction wrapping the following DebugAction to only print it every $(de.every)th iteration.\n$(_in_str(status_summary(de.debug; context = context)))"
end
function set_parameter!(de::DebugEvery, e::Val, args...)
    set_parameter!(de.debug, e, args...)
    return de
end
