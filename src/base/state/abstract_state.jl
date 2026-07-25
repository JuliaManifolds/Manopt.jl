@doc """
    AbstractManoptSolverState

A general super type for all solver states.

# Fields

The following fields are assumed to be available by default.
If you use different ones, adapt the the access functions
[`get_iterate`](@ref), [`get_stopping_criterion`](@ref),
and [`get_callbacks`](@ref)  accordingly

$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:stopping_criterion; name = "stop"))
"""
abstract type AbstractManoptSolverState end

@doc """
    AbstractGradientSolverState <: AbstractManoptSolverState

A generic [`AbstractManoptSolverState`](@ref) type for gradient based options data.

It assumes that

* the iterate is stored in the field `p`
* the gradient at `p` is stored in `X`.

# See also
[`GradientDescentState`](@ref), [`StochasticGradientDescentState`](@ref), [`SubGradientMethodState`](@ref), [`QuasiNewtonState`](@ref).
"""
abstract type AbstractGradientSolverState <: AbstractManoptSolverState end


"""
    AbstractRestartCondition

A general struct, that indicates then to restart.
It is used within the [`ConjugateGradientDescentState`](@ref).

It is implemented to work as a functor `(problem, state, iteration) -> true|false`
and what is done in the restart case (`true`) is decided by the single solver.
"""
abstract type AbstractRestartCondition end

"""
    get_count(ams::AbstractManoptSolverState, ::Symbol)

Obtain the count for a certain countable size, for example the `:Iterations`.
This function returns 0 if there was nothing to count

Available symbols from within the solver state

* `:Iterations` is passed on to the `stop` field to obtain the
  iteration at which the solver stopped.
"""
function get_count(ams::AbstractManoptSolverState, s::Symbol)
    return get_count(ams, Val(s))
end

function get_count(ams::AbstractManoptSolverState, v::Val{:Iterations})
    return get_count(ams.stop, v)
end
@doc """
    get_gradient(agst::AbstractGradientSolverState)

return the gradient stored within gradient options.
THe default returns `agst.X`.
"""
get_gradient(agst::AbstractGradientSolverState) = agst.X

"""
    get_iterate(state::AbstractManoptSolverState)

return the (last stored) iterate within [`AbstractManoptSolverState`](@ref)` `state`.
This should usually refer to a single point on the manifold the solver is working on

By default this also removes all decorators of the state beforehand.
"""
get_iterate(s::AbstractManoptSolverState) = _get_iterate(s, dispatch_state_decorator(s))
function _get_iterate(s::AbstractManoptSolverState, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide access to an iterate.
        If it has the iterate stored internally, please implement `get_iterate(s::$(typeof(s))).`
        ",
    )
end
_get_iterate(s::AbstractManoptSolverState, ::Val{true}) = get_iterate(s.state)

@doc """
    get_iterate(agst::AbstractGradientSolverState)

return the iterate stored within gradient options.
THe default returns `agst.p`.
"""
get_iterate(agst::AbstractGradientSolverState) = agst.p


_set_iterate!(s::AbstractManoptSolverState, M, p, ::Val{true}) = set_iterate!(s.state, M, p)

@doc """
    get_message(du::AbstractManoptSolverState)

get a message (String) from internal functors, in a summary.
This should return any message a sub-step might have issued as well.
"""
function get_message(s::AbstractManoptSolverState)
    return _get_message(s, dispatch_state_decorator(s))
end
_get_message(s::AbstractManoptSolverState, ::Val{true}) = get_message(s.state)
#INtroduce a default that there is no message
_get_message(s::AbstractManoptSolverState, ::Val{false}) = ""

@doc """
    get_state(s::AbstractManoptSolverState, recursive::Bool=true)

return the (one step) undecorated [`AbstractManoptSolverState`](@ref) of the (possibly) decorated `s`.
As long as your decorated state stores the state within `s.state` and
the [`dispatch_objective_decorator`](@ref) is set to `Val{true}`,
the internal state are extracted automatically.

By default the state that is stored within a decorated state is assumed to be at
`s.state`. Overwrite `_get_state(s, ::Val{true}, recursive) to change this behaviour for your state `s`
for both the recursive and the direct case.

If `recursive` is set to `false`, only the most outer decorator is taken away instead of all.
"""
function get_state(s::AbstractManoptSolverState, recursive::Bool = true)
    return _get_state(s, dispatch_state_decorator(s), recursive)
end
_get_state(s::AbstractManoptSolverState, ::Val{false}, rec = true) = s
function _get_state(s::AbstractManoptSolverState, ::Val{true}, rec = true)
    return rec ? get_state(s.state) : s.state
end

@doc """
    get_stopping_criterion(ams::AbstractManoptSolverState)

Return the [`StoppingCriterion`](@ref) stored within the [`AbstractManoptSolverState`](@ref) `ams`.

For an undecorated state, this is assumed to be in `ams.stop`.
Overwrite `_get_stopping_criterion(yms::YMS)`
to change this for your manopt solver (`yms`) assuming it has type YMS`.
"""
function get_stopping_criterion(ams::AbstractManoptSolverState)
    return _get_stopping_criterion(get_state(ams, true))
end
_get_stopping_criterion(ams::AbstractManoptSolverState) = ams.stop

"""
    has_converged(ams::AbstractManoptSolverState)

Return whether the solver has converged, based on the internal [`StoppingCriterion`](@ref).
"""
has_converged(ams::AbstractManoptSolverState) = has_converged(get_stopping_criterion(ams))

@doc """
    set_gradient!(state::AbstractGradientSolverState, M, p, X)

set the (current) gradient stored within an [`AbstractGradientSolverState`](@ref) to `X`.
The default function modifies `state.X`.
"""
function set_gradient!(state::AbstractGradientSolverState, M, p, X)
    copyto!(M, state.X, p, X)
    return state
end

"""
    set_iterate!(s::AbstractManoptSolverState, M::AbstractManifold, p)

set the iterate within an [`AbstractManoptSolverState`](@ref) to some (start) value `p`.
"""
function set_iterate!(s::AbstractManoptSolverState, M, p)
    return _set_iterate!(s, M, p, dispatch_state_decorator(s))
end
function _set_iterate!(s::AbstractManoptSolverState, ::Any, ::Any, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide (write) access to an iterate",
    )
end

"""
    set_parameter!(ams::AbstractManoptSolverState, element::Symbol, args...)

Set a certain field or semantic element from the [`AbstractManoptSolverState`](@ref) `ams` to `value`.
This function passes to `Val(element)` and specific setters should dispatch on `Val{element}`.

By default, this function just does nothing.
"""
function set_parameter!(ams::AbstractManoptSolverState, e::Symbol, args...)
    return set_parameter!(ams, Val(e), args...)
end
# Default: do nothing
function set_parameter!(ams::AbstractManoptSolverState, ::Val, args...)
    return ams
end

@doc """
    stopped_at(state::AbstractManoptSolverState)

Return the number of iterations the solver represented by the `state` took to stop.
If the solver has not yet stopped, this function returns `-1`.

By default, this function calls `get_count` function on the state's stopping criterion to access its `:Iteration` count.
"""
function stopped_at(state::AbstractManoptSolverState)
    return get_count(get_stopping_criterion(state), Val(:Iterations))
end

function Base.show(io::IO, ::MIME"text/plain", ams::AbstractManoptSolverState)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, ams) : show(io, ams)
end
