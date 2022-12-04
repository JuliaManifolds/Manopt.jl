"""
    initialize_solver!(p::AbstractManoptProblem, o::DebugSolverState)

Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`DebugSolverState`](@ref)` o`.

Since debug acts as a decorator this also calls the `initialize_solver!`
of the correpsonding internally stored options
"""
function initialize_solver!(p::AbstractManoptProblem, s::DebugSolverState)
    initialize_solver!(p, s.state)
    get(s.debugDictionary, :Start, DebugDivider(""))(p, get_state(s), 0)
    get(s.debugDictionary, :All, DebugDivider(""))(p, get_state(s), 0)
    return s
end
"""
    step_solver!(p::AbstractManoptProblem, s::DebugSolverState, i)

    Do one iteration step (the `i`th) for [`Problem`](@ref)` p` by modifying
the values in the [`AbstractManoptSolverState`](@ref)` s.state` and print the debug specified
"""
function step_solver!(p::AbstractManoptProblem, s::DebugSolverState, i)
    step_solver!(p, s.state, i)
    get(s.debugDictionary, :Step, DebugDivider(""))(p, get_state(o), i)
    get(s.debugDictionary, :All, DebugDivider(""))(p, get_state(o), i)
    return o
end

"""
    stop_solver!(p,o,i)

determine whether the solver for [`Problem`](@ref) `p` and the [`DebugSolverState`](@ref) `o`
should stop at iteration `i` by calling the function corresponding to the internally stored [`AbstractManoptSolverState`](@ref).
If so, print debug from `:All` and `:Stop`.
"""
function stop_solver!(p::AbstractManoptProblem, s::DebugSolverState, i::Int)
    stop = stop_solver!(p, s.state, i)
    if stop
        get(s.debugDictionary, :Stop, DebugDivider(""))(p, get_state(s), typemin(Int))
        get(s.debugDictionary, :All, DebugDivider(""))(p, get_state(s), typemin(Int))
    end
    return stop
end
