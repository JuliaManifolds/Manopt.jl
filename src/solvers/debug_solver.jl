"""
    initialize_solver!(p::AbstractManoptProblem, o::DebugSolverState)

Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`DebugSolverState`](@ref)` o`.

Since debug acts as a decorator this also calls the `initialize_solver!`
of the correpsonding internally stored options
"""
function initialize_solver!(p::AbstractManoptProblem, o::DebugSolverState)
    initialize_solver!(p, o.options)
    get(o.debugDictionary, :Start, DebugDivider(""))(p, get_state(o), 0)
    get(o.debugDictionary, :All, DebugDivider(""))(p, get_state(o), 0)
    return o
end
"""
    step_solver!(p::AbstractManoptProblem, o::DebugSolverState, i)

    Do one iteration step (the `i`th) for [`Problem`](@ref)` p` by modifying
the values in the [`AbstractManoptSolverState`](@ref)` o.options` and print the debug specified
"""
function step_solver!(p::AbstractManoptProblem, o::DebugSolverState, i)
    step_solver!(p, o.options, i)
    get(o.debugDictionary, :Step, DebugDivider(""))(p, get_state(o), i)
    get(o.debugDictionary, :All, DebugDivider(""))(p, get_state(o), i)
    return o
end

"""
    stop_solver!(p,o,i)

determine whether the solver for [`Problem`](@ref) `p` and the [`DebugSolverState`](@ref) `o`
should stop at iteration `i` by calling the function corresponding to the internally stored [`AbstractManoptSolverState`](@ref).
If so, print debug from `:All` and `:Stop`.
"""
function stop_solver!(p::AbstractManoptProblem, o::DebugSolverState, i::Int)
    s = stop_solver!(p, o.options, i)
    if s
        get(o.debugDictionary, :Stop, DebugDivider(""))(p, get_state(o), typemin(Int))
        get(o.debugDictionary, :All, DebugDivider(""))(p, get_state(o), typemin(Int))
    end
    return s
end
