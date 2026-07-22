@inline _extract_val(::Val{T}) where {T} = T

"""
    ClosedFormSubSolverState{E<:AbstractEvaluationType} <: AbstractManoptSolverState

Subsolver state indicating that a closed-form solution is available with
[`AbstractEvaluationType`](@ref) `E`.

# Constructor

    ClosedFormSubSolverState(; evaluation=AllocatingEvaluation())
"""
struct ClosedFormSubSolverState{E <: AbstractEvaluationType} <: AbstractManoptSolverState end
function ClosedFormSubSolverState(::E) where {E <: AbstractEvaluationType}
    return ClosedFormSubSolverState{E}()
end
function ClosedFormSubSolverState(;
        evaluation::E = AllocatingEvaluation()
    ) where {E <: AbstractEvaluationType}
    return ClosedFormSubSolverState(evaluation)
end
Base.show(io::IO, cfss::ClosedFormSubSolverState{E}) where {E} = print(io, "ClosedFormSubSolverState(; $(_to_kw(E)))")
status_summary(cfss::ClosedFormSubSolverState; context::Symbol = :default) = repr(cfss)

maybe_wrap_evaluation_type(s::AbstractManoptSolverState) = s
maybe_wrap_evaluation_type(n::Nothing) = n
function maybe_wrap_evaluation_type(::E) where {E <: AbstractEvaluationType}
    return ClosedFormSubSolverState{E}()
end

@doc """
    ReturnSolverState{O<:AbstractManoptSolverState} <: AbstractManoptSolverState

This internal type is used to indicate that the contained [`AbstractManoptSolverState`](@ref) `state`
should be returned at the end of a solver instead of the usual minimizer.

# See also

[`get_solver_result`](@ref)
"""
struct ReturnSolverState{S <: AbstractManoptSolverState} <: AbstractManoptSolverState
    state::S
end
status_summary(rst::ReturnSolverState; context::Symbol = :default) = status_summary(rst.state; context = context)
show(io::IO, rst::ReturnSolverState) = print(io, "ReturnSolverState(", rst.state, ")")
dispatch_state_decorator(::ReturnSolverState) = Val(true)

"""
    get_solver_return(s::AbstractManoptSolverState)
    get_solver_return(o::AbstractManifoldObjective, s::AbstractManoptSolverState)

determine the result value of a call to a solver.
By default this returns the same as [`get_solver_result`](@ref).

    get_solver_return(s::ReturnSolverState)
    get_solver_return(o::AbstractManifoldObjective, s::ReturnSolverState)

return the internally stored state of the [`ReturnSolverState`](@ref) instead of the minimizer.
This means that when the state are decorated like this, the user still has to call [`get_solver_result`](@ref)
on the internal state separately.

    get_solver_return(o::ReturnManifoldObjective, s::AbstractManoptSolverState)

return both the objective and the state as a tuple.
"""
function get_solver_return(s::AbstractManoptSolverState)
    return _get_solver_return(s, dispatch_state_decorator(s))
end
_get_solver_return(s::AbstractManoptSolverState, ::Val{false}) = get_solver_result(s)
_get_solver_return(s::AbstractManoptSolverState, ::Val{true}) = get_solver_return(s.state)
get_solver_return(s::ReturnSolverState) = s.state
function get_solver_return(o::AbstractManifoldObjective, s::AbstractManoptSolverState)
    #resolve objective first
    return _get_solver_return(o, s, dispatch_objective_decorator(o))
end
# remove decorator
function _get_solver_return(o::AbstractManifoldObjective, s, ::Val{true})
    return get_solver_return(get_objective(o, false), s)
end
_get_solver_return(::AbstractManifoldObjective, s, ::Val{false}) = get_solver_return(s)
function get_solver_return(o::ReturnManifoldObjective, s::AbstractManoptSolverState)
    return o.objective, get_solver_return(s)
end

"""
    get_gradient(s::AbstractManoptSolverState)

return the (last stored) gradient within [`AbstractManoptSolverState`](@ref)` `s`.
By default also undecorates the state beforehand
"""
get_gradient(s::AbstractManoptSolverState) = _get_gradient(s, dispatch_state_decorator(s))
function _get_gradient(s::AbstractManoptSolverState, ::Val{false})
    return error("It seems that $s do not provide access to a gradient")
end
_get_gradient(s::AbstractManoptSolverState, ::Val{true}) = get_gradient(s.state)

"""
    set_gradient!(s::AbstractManoptSolverState, M::AbstractManifold, p, X)

set the gradient within an (possibly decorated) [`AbstractManoptSolverState`](@ref)
to some (start) value `X` in the tangent space at `p`.
"""
function set_gradient!(s::AbstractManoptSolverState, M, p, X)
    return _set_gradient!(s, M, p, X, dispatch_state_decorator(s))
end
function _set_gradient!(s::AbstractManoptSolverState, ::Any, ::Any, ::Any, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide (write) access to a gradient",
    )
end
function _set_gradient!(s::AbstractManoptSolverState, M, p, X, ::Val{true})
    return set_gradient!(s.state, M, p, X)
end

"""
    get_solver_result(state::AbstractManoptSolverState)
    get_solver_result(tos::Tuple{AbstractManifoldObjective,AbstractManoptSolverState})
    get_solver_result(objective::AbstractManifoldObjective, state::AbstractManoptSolverState)
    get_solver_result(problem::AbstractManoptProblem, state::AbstractManoptSolverState)

Return the final result after all iterations that is stored within
the [`AbstractManoptSolverState`](@ref) `ams`, which was modified during the iterations.

For the case an [`AbstractManifoldObjective`](@ref) `o` the objective is passed as well
– either as a Tuple or as two parameters –, by default, the objective is ignored,
and the solver result for the state is called; this is due to display reasons in REPL
related to statistics, where such a Tuple might appear

For the case an [`AbstractManoptProblem`](@ref) `p` is passed as well as
a first optional parameter, by default the problem is ignored.
This can be used to change the representation of a result stored in a state, e.g.
when a tangent vector is (part of) the result, changing between representations in
coefficients and different tangent vector representations could be performed as a final step,
depending on which problem was aimed to be solved

Note that the returned value or point might still be aliased to the original `state`.
"""
function get_solver_result(state::AbstractManoptSolverState)
    return _get_solver_result(state, dispatch_state_decorator(state))
end
function get_solver_result(
        tos::Tuple{<:AbstractManifoldObjective, <:AbstractManoptSolverState}
    )
    return get_solver_result(tos...)
end
function get_solver_result(::AbstractManifoldObjective, state::AbstractManoptSolverState)
    return get_solver_result(state)
end
#A problem or – hence untyped – a closed form solution / function
function get_solver_result(pf, state::AbstractManoptSolverState)
    return get_solver_result(state)
end
function get_solver_result(tos::Tuple{<:AbstractManifoldObjective, S}) where {S}
    return tos[2]
end
# if the second one is anything else, assume it is a point/result -> return that
function get_solver_result(::AbstractManifoldObjective, p)
    return p
end
_get_solver_result(state::AbstractManoptSolverState, ::Val{false}) = get_iterate(state)
_get_solver_result(state::AbstractManoptSolverState, ::Val{true}) = get_solver_result(state.state)

# in general, ignore printing the objective by default
function show(io::IO, t::Tuple{<:AbstractManifoldObjective, <:AbstractManoptSolverState})
    return print(io, "$(t[2])")
end
# for decorated ones, default: pass down
function show(
        io::IO, t::Tuple{<:AbstractDecoratedManifoldObjective, <:AbstractManoptSolverState}
    )
    return show(io, (get_objective(t[1], false), t[2]))
end
