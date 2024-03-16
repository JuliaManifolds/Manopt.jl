"""
    initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)

Extend the initialization of the solver by a hook to run the [`DebugAction`](@ref)
that was added to the `:Start` entry of the debug lists.
"""
function initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)
    initialize_solver!(amp, dss.state)
    # Call Start
    get(dss.debugDictionary, :Start, DebugDivider(""))(amp, get_state(dss), 1)
    # Reset others
    for (key, action) in dss.debugDictionary
        !(key === :Start) && action(amp, get_state(dss), 0)
    end
    return dss
end
"""
    step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)

Extend the `i`th step of the solver by a hook to run debug prints,
that were added to the `:Step` and `:All` entries of the debug lists.
"""
function step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)
    get(dss.debugDictionary, :BeforeIteration, DebugDivider(""))(amp, get_state(dss), i)
    step_solver!(amp, dss.state, i)
    get(dss.debugDictionary, :Iteration, DebugDivider(""))(amp, get_state(dss), i)
    return dss
end

"""
    stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)

Extend the `stop_solver!`, whether to stop the solver by a hook to run debug,
that were added to the `:Stop` and `:All` entries of the debug lists.
"""
function stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i::Int)
    stop = stop_solver!(amp, dss.state, i)
    if stop
        get(dss.debugDictionary, :Stop, DebugDivider(""))(amp, get_state(dss), typemax(Int))
    end
    return stop
end
