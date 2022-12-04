@doc raw"""
    decorate_state(o)

decorate the [`AbstractManoptSolverState`](@ref)` o` with specific decorators.

# Optional Arguments
optional arguments provide necessary details on the decorators. A specific
one is used to activate certain decorators.

* `debug` – (`Array{Union{Symbol,DebugAction,String,Int},1}()`) a set of symbols
  representing [`DebugAction`](@ref)s, `Strings` used as dividers and a subsampling
  integer. These are passed as a [`DebugGroup`](@ref) within `:All` to the
  [`DebugSolverState`](@ref) decorator dictionary. Only excention is `:Stop` that is passed to `:Stop`.
* `record` – (`Array{Union{Symbol,RecordAction,Int},1}()`) specify recordings
  by using `Symbol`s or [`RecordAction`](@ref)s directly. The integer can again
  be used for only recording every ``i``th iteration.
* `return_state` - (`false`) indicate whether to wrap the options in a [`ReturnSolverState`](@ref),
  indicating that the solver should return options and not (only) the minimizer.

# See also
[`DebugSolverState`](@ref), [`RecordSolverState`](@ref), [`ReturnSolverState`](@ref)
"""
function decorate_state(
    s::O;
    debug::Union{
        Missing, # none
        DebugAction, # single one for :All
        Array{DebugAction,1}, # a group that's put into :All
        Dict{Symbol,DebugAction}, # the most elaborate, a dictionary
        Array{<:Any,1}, # short hand for Factory.
    }=missing,
    record::Union{
        Missing, # none
        Symbol, # single action shortcut by symbol
        RecordAction, # single action
        Array{RecordAction,1}, # a group to be set in :All
        Dict{Symbol,RecordAction}, # a dictionary for precise settings
        Array{<:Any,1}, # a formated string with symbols orAbstractStateActions
    }=missing,
    return_state=false,
) where {O<:AbstractManoptSolverState}
    o = ismissing(debug) ? s : DebugSolverState(s, debug)
    o = ismissing(record) ? o : RecordSolverState(o, record)
    o = (return_state) ? ReturnSolverState(o) : o
    return o
end
"""
    initialize_solver!(p,o)

Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`AbstractManoptSolverState`](@ref)` o`.
"""
function initialize_solver! end

"""
    step_solver!(p,o,iter)

Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`AbstractManoptSolverState`](@ref) `o`.
"""
function step_solver! end

"""
    get_solver_result(o)

Return the final result after all iterations that is stored within the
(modified during the iterations) [`AbstractManoptSolverState`](@ref) `o`. By default it uses[`get_iterate`](@ref)
"""
function get_solver_result end

"""
    stop_solver!(p,o,i)

depending on the current [`Problem`](@ref) `p`, the current state of the solver
stored in [`AbstractManoptSolverState`](@ref) `o` and the current iterate `i` this function determines
whether to stop the solver by calling the [`StoppingCriterion`](@ref).
"""
function stop_solver!(p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int)
    return s.stop(p, s, i)
end

"""
    solve!(p,o)

run the solver implemented for the [`Problem`](@ref)` p` and the
[`AbstractManoptSolverState`](@ref)` o` employing [`initialize_solver!`](@ref), [`step_solver!`](@ref),
as well as the [`stop_solver!`](@ref) of the solver.
"""
function solve!(p::AbstractManoptProblem, s::AbstractManoptSolverState)
    iter::Integer = 0
    initialize_solver!(p, s)
    while !stop_solver!(p, s, iter)
        iter = iter + 1
        step_solver!(p, s, iter)
    end
    return s
end

function initialize_solver!(p::AbstractManoptProblem, s::ReturnSolverState)
    return initialize_solver!(p, s.state)
end
function step_solver!(p::AbstractManoptProblem, s::ReturnSolverState, i)
    return step_solver!(p, s.state, i)
end
function stop_solver!(p::AbstractManoptProblem, s::ReturnSolverState, i::Int)
    return stop_solver!(p, s.state, i)
end
