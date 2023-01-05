"""
    initialize_solver!(p,o)

Initialize the solver to the optimization [`AbstractManoptProblem`](@ref) by initializing
the encapsulated [`AbstractManoptSolverState`](@ref) from within the [`RecordSolverState`](@ref)` o`.
"""
function initialize_solver!(p::AbstractManoptProblem, s::RecordSolverState)
    initialize_solver!(p, s.state)
    get(s.recordDictionary, :Start, RecordGroup())(p, get_state(s), 0)
    return s
end

"""
    step_solver!(p,o,iter)

Do one iteration step (the `iter`th) for [`AbstractManoptProblem`](@ref)` p` by modifying
the values in the [`AbstractManoptSolverState`](@ref)` s.state` and record the result(s).
"""
function step_solver!(p::AbstractManoptProblem, s::RecordSolverState, i)
    step_solver!(p, s.state, i)
    get(s.recordDictionary, :Iteration, RecordGroup())(p, get_state(s), i)
    return s
end

"""
    stop_solver!(amp::AbstractManoptProblem, rss::RecordSolverState, i)

determine whether the solver for [`AbstractManoptProblem`](@ref)  `amp` and the
[`RecordSolverState`](@ref) `rss` should stop at iteration `i`.
If so, do a (final) record to `:All` and `:Stop`.
"""
function stop_solver!(p::AbstractManoptProblem, s::RecordSolverState, i::Int)
    stop = stop_solver!(p, s.state, i)
    stop && get(s.recordDictionary, :Stop, RecordGroup())(p, get_state(s), i)
    return stop
end
