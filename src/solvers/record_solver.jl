"""
    initialize_solver!(p,o)

Initialize the solver to the optimization [`Problem`](@ref) by initializing
the encapsulated `options` from within the [`RecordOptions`](@ref)` o`.
"""
function initialize_solver!(p::AbstractManoptProblem, o::RecordOptions)
    initialize_solver!(p, o.options)
    get(o.recordDictionary, :Start, RecordGroup())(p, get_options(o), 0)
    return o
end
"""
    step_solver!(p,o,iter)

Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o.options` and record the result(s).
"""
function step_solver!(p::AbstractManoptProblem, o::RecordOptions, i)
    step_solver!(p, o.options, i)
    get(o.recordDictionary, :Iteration, RecordGroup())(p, get_options(o), i)
    return o
end

"""
    stop_solver!(p,o,i)

determine whether the solver for [`Problem`](@ref) `p` and the
[`RecordOptions`](@ref) `o` should stop at iteration `i`.
If so, do a (final) record to `:All` and `:Stop`.
"""
function stop_solver!(p::AbstractManoptProblem, o::RecordOptions, i::Int)
    s = stop_solver!(p, o.options, i)
    s && get(o.recordDictionary, :Stop, RecordGroup())(p, get_options(o), i)
    return s
end
