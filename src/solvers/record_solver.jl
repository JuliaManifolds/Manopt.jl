"""
    initialize_solver!(ams::AbstractManoptProblem, rss::RecordSolverState)

Extend the initialization of the solver by a hook to run records
that were added to the `:Start` entry.
"""
function initialize_solver!(amp::AbstractManoptProblem, rss::RecordSolverState)
    initialize_solver!(amp, rss.state)
    get(rss.recordDictionary, :Start, RecordGroup())(amp, get_state(rss), 0)
    # Reset Iteration and Stop
    get(rss.recordDictionary, :Iteration, RecordGroup())(amp, get_state(rss), -1)
    get(rss.recordDictionary, :Stop, RecordGroup())(amp, get_state(rss), -1)
    return rss
end

"""
    step_solver!(amp::AbstractManoptProblem, rss::RecordSolverState, k)

Extend the `i`th step of the solver by a hook to run records,
that were added to the `:Iteration` entry.
"""
function step_solver!(amp::AbstractManoptProblem, rss::RecordSolverState, k)
    step_solver!(amp, rss.state, k)
    get(rss.recordDictionary, :Iteration, RecordGroup())(amp, get_state(rss), k)
    return rss
end

"""
    stop_solver!(amp::AbstractManoptProblem, rss::RecordSolverStatek k)

Extend the call to the stopping criterion by a hook to run records, that were
added to the `:Stop` entry.
"""
function stop_solver!(amp::AbstractManoptProblem, rss::RecordSolverState, k)
    stop = stop_solver!(amp, rss.state, k)
    stop && get(rss.recordDictionary, :Stop, RecordGroup())(amp, get_state(rss), k)
    return stop
end
