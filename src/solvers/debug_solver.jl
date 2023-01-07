"""
    initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)

Extend the initialization of the solver by a hook to run debug
that were added to the `:Start` and `:All` entries of the debug lists.
"""
function initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)
    initialize_solver!(amp, dss.state)
    get(dss.debugDictionary, :Start, DebugDivider(""))(amp, get_state(dss), 0)
    get(dss.debugDictionary, :All, DebugDivider(""))(amp, get_state(dss), 0)
    return dss
end
"""
    step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)

Extend the `i`th step of the solver by a hook to run debug prints,
that were added to the `:Step` and `:All` entries of the debug lists.
"""
function step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)
    step_solver!(amp, dss.state, i)
    get(dss.debugDictionary, :Step, DebugDivider(""))(amp, get_state(dss), i)
    get(dss.debugDictionary, :All, DebugDivider(""))(amp, get_state(dss), i)
    return dss
end

"""
    stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)

Extend the check, whether to stop the solver by a hook to run debug,
that were added to the `:Stop` and `:All` entries of the debug lists.
"""
function stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i::Int)
    stop = stop_solver!(amp, dss.state, i)
    if stop
        get(dss.debugDictionary, :Stop, DebugDivider(""))(amp, get_state(dss), typemin(Int))
        get(dss.debugDictionary, :All, DebugDivider(""))(amp, get_state(dss), typemin(Int))
    end
    return stop
end
