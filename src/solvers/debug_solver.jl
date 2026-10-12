"""
    initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)

Extend the initialization of the solver by a hook to run the [`DebugAction`](@ref)
that was added to the `:Start` entry of the debug lists. The `:BeforeIteration`
and `:Iteration` entries are additionally triggered with iteration number `0`,
so that they are reset and may already print, the `:Stop` entry is triggered
with `-1`, so that it is only reset without printing.
"""
function initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)
    initialize_solver!(amp, dss.state)
    # Call Start
    get(dss.debug_dictionary, :Start, _EMPTY_DIVIDER)(amp, get_state(dss), 0)
    # Reset / Init (maybe with print at 0) (before) Iteration
    for key in [:BeforeIteration, :Iteration]
        get(dss.debug_dictionary, key, _EMPTY_DIVIDER)(amp, get_state(dss), 0)
    end
    # (just) reset Stop (do not print here)
    for key in [:Stop]
        get(dss.debug_dictionary, key, _EMPTY_DIVIDER)(amp, get_state(dss), -1)
    end
    return dss
end
"""
    step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, k)

Extend the `k`th step of the solver by a hook to run debug prints,
that were added to the `:BeforeIteration` and `:Iteration` entries of the debug lists.
"""
function step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, k)
    get(dss.debug_dictionary, :BeforeIteration, _EMPTY_DIVIDER)(amp, get_state(dss), k)
    step_solver!(amp, dss.state, k)
    get(dss.debug_dictionary, :Iteration, _EMPTY_DIVIDER)(amp, get_state(dss), k)
    return dss
end

"""
    stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, k)

Extend the call to the stopping criterion by a hook to run debug actions
that were added to the `:Stop` entry of the debug lists.
"""
function stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, k::Int)
    stop = stop_solver!(amp, dss.state, k)
    if stop
        get(dss.debug_dictionary, :Stop, _EMPTY_DIVIDER)(amp, get_state(dss), k)
    end
    return stop
end
