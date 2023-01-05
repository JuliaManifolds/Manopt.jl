"""
    initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)

Initialize the solver to the optimization [`AbstractManoptProblem`](@ref) `amp` by
initializing all values in the [`DebugSolverState`](@ref) `dss`.

Since debug acts as a decorator this also calls the `initialize_solver!`
of the correpsonding internally stored options
"""
function initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)
    initialize_solver!(amp, dss.state)
    get(dss.debugDictionary, :Start, DebugDivider(""))(amp, get_state(dss), 0)
    get(dss.debugDictionary, :All, DebugDivider(""))(amp, get_state(dss), 0)
    return dss
end
"""
    step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)

    Do one iteration step (the `i`th) for [`AbstractManoptProblem`](@ref) `amp` by modifying
the values in the [`AbstractManoptSolverState`](@ref)` s.state` and print the debug specified
"""
function step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)
    step_solver!(amp, dss.state, i)
    get(dss.debugDictionary, :Step, DebugDivider(""))(amp, get_state(dss), i)
    get(dss.debugDictionary, :All, DebugDivider(""))(amp, get_state(dss), i)
    return dss
end

"""
    stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)

determine whether the solver for [`AbstractManoptProblem`](@ref) `amp` and the [`DebugSolverState`](@ref) `dss`
should stop at iteration `i` by calling the function corresponding to the internally stored [`AbstractManoptSolverState`](@ref).
If so, print debug from `:All` and `:Stop`.
"""
function stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i::Int)
    stop = stop_solver!(amp, dss.state, i)
    if stop
        get(dss.debugDictionary, :Stop, DebugDivider(""))(amp, get_state(dss), typemin(Int))
        get(dss.debugDictionary, :All, DebugDivider(""))(amp, get_state(dss), typemin(Int))
    end
    return stop
end
