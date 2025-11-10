"""
    StepsizeMessage{TBound, TS}

A message struct to hold stepsize information, when e.g.
a step size underflow happens at a certain iteration

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

Given a certain structure `a` from within `Manopt.jl`, retireve its last message of
information, e.g. warnings from a step size.
If no message is available, an empty string is returned.
"""
function get_message end

"""
    reset_messages!(messages::NamedTuple)

Given a named tuple of `[StepsizeMessage`](@ref)s, reset all messages to default values,
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

Given a named tuple of `[StepsizeMessage`](@ref)s, set the message identified by `key` to the provided values,
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

Given a named tuple of `[StepsizeMessage`](@ref)s, set the message identified by `key` to the provided values,
i.e. if they are not `nothing`.
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
# Displaying concrete messages
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
amd the step size `step` chosen instead.
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

Display a message string for stopping the increase of the step size at iteration `k`
and the step size `step` chosen instead.
"""
function get_message(::Val{:stepsize_less}, k::Int = -1, value::Real = NaN, bound::Real = NaN)
    (k < 0) && (return "")
    s = (k == 0) ? "the beginning" : "iteration #$k"
    s_str = isnan(value) ? "" : " Falling back to a stepsize of $value."
    b_str = isnan(bound) ? "" : "($bound)"
    return (k > 0) ? "At $s: Minimal stepsize less than bound $b_str reached.$s_str" : ""
end
