"""
    initialize_solver!(p,o)

Initialize the solver to the optimization [`Problem`](@ref) by initializing
the encapsulated `options` from within the [`RecordOptions`](@ref)` o`.
"""
function initialize_solver!(p::Problem,o::RecordOptions)
    initialize_solver!(p,o.options)
    get(o.recordDictionary, :Start, RecordGroup() )(p, get_options(o), 0)
    get(o.recordDictionary, :All,   RecordGroup() )(p, get_options(o), 0)
end
"""
    step_solver!(p,o,iter)

Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o.options` and record the result(s).
"""
function step_solver!(p::Problem,o::RecordOptions, i)
    step_solver!(p,o.options,i)
    get(o.recordDictionary, :Step, RecordGroup() )(p, get_options(o), i)
    get(o.recordDictionary, :All,   RecordGroup() )(p, get_options(o), i)
end
"""
    get_solver_result(o)

Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref)` o`.
"""
function get_solver_result(o::RecordOptions)
    return get_solver_result(o.options)
end
"""
    stop_solver!(p,o,i)

determine whether the solver for [`Problem`](@ref) `p` and the
[`RecordOptions`](@ref) `o` should stop at iteration `i`.
If so, do a (final) record to `:All` and `:Stop`.
"""
function stop_solver!(p::Problem,o::RecordOptions,i::Int)
    s = stop_solver!(p,o.options,i)
    if s
        get(o.recordDictionary, :Stop, RecordGroup() )(p, get_options(o), typemin(Int) )
        get(o.recordDictionary, :All,   RecordGroup() )(p, get_options(o), typemin(Int ))
    end
    return s
end