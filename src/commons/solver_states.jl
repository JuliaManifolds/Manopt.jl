"""
    ClosedFormSubSolverState{E<:AbstractEvaluationType} <: AbstractManoptSolverState

Subsolver state indicating that a closed-form solution is available

# Constructor

    ClosedFormSubSolverState()
"""
struct ClosedFormSubSolverState{} <: AbstractManoptSolverState end
Base.show(io::IO, ::ClosedFormSubSolverState) = print(io, "ClosedFormSubSolverState()")
status_summary(cfss::ClosedFormSubSolverState; context::Symbol = :default) = repr(cfss)

# TODO: After refactoring, Part I: Replace this by maybe wrapping the closed form (allocating) solution
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
    get_solver_return(s::ReturnSolverState)
    get_solver_return(o::AbstractManifoldObjective, s::ReturnSolverState)

return the internally stored state of the [`ReturnSolverState`](@ref) instead of the minimizer.
This means that when the state are decorated like this, the user still has to call [`get_solver_result`](@ref)
on the internal state separately.
"""
get_solver_return(s::ReturnSolverState) = s.state
