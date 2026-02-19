"""
    initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)

Extend the initialization of the solver by a hook to run the [`DebugAction`](@ref)
that was added to the `:Start` entry of the debug lists. All others are
triggered (with iteration number `0`) to trigger possible resets
"""
function initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)
    initialize_solver!(amp, dss.state)
    # Call Start
    get(dss.debugDictionary, :Start, dss.empty_divider)(amp, get_state(dss), 0)
    # Reset / Init (maybe with print at 0) (before) Iteration
    for key in [:BeforeIteration, :Iteration]
        get(dss.debugDictionary, key, dss.empty_divider)(amp, get_state(dss), 0)
    end
    # (just) reset Stop (do not print here)
    for key in [:Stop]
        get(dss.debugDictionary, key, dss.empty_divider)(amp, get_state(dss), -1)
    end
    return dss
end
"""
    step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, k)

Extend the `i`th step of the solver by a hook to run debug prints,
that were added to the `:BeforeIteration` and `:Iteration` entries of the debug lists.
"""
function step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, k)
    get(dss.debugDictionary, :BeforeIteration, dss.empty_divider)(amp, get_state(dss), k)
    step_solver!(amp, dss.state, k)
    get(dss.debugDictionary, :Iteration, dss.empty_divider)(amp, get_state(dss), k)
    return dss
end

"""
    stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, k)

Extend the `stop_solver!`, whether to stop the solver by a hook to run debug,
that were added to the `:Stop` entry of the debug lists.
"""
function stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, k::Int)
    stop = stop_solver!(amp, dss.state, k)
    if stop
        get(dss.debugDictionary, :Stop, dss.empty_divider)(amp, get_state(dss), k)
    end
    return stop
end
