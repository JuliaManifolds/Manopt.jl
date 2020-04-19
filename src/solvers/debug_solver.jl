"""
    initialize_solver!(p,o)
Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`DebugOptions`](@ref)` o`.
"""
function initialize_solver!(p::Problem, o::DebugOptions)
    initialize_solver!(p,o.options)
    get(o.debugDictionary,:Start,DebugDivider(""))(p,get_options(o),0)
    get(o.debugDictionary,:All,DebugDivider(""))(p,get_options(o),0)
end
"""
    step_solver!(p,o,iter)
Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o.options` and print `Debug`.
"""
function step_solver!(p::Problem, o::DebugOptions, i)
    step_solver!(p,o.options,i)
    get(o.debugDictionary,:Step,DebugDivider(""))(p,get_options(o),i)
    get(o.debugDictionary,:All,DebugDivider(""))(p,get_options(o),i)
end
"""
    get_solver_result(o)
Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref) `o`.
"""
function get_solver_result(o::DebugOptions)
    return get_solver_result(o.options)
end
"""
    stop_solver!(p,o,i)

determine whether the solver for [`Problem`](@ref) `p` and the [`DebugOptions`](@ref) `o`
should stop at iteration `i`. If so, print all debug from `:All` and `:Final`.
"""
function stop_solver!(p::Problem,o::DebugOptions,i::Int)
    s = stop_solver!(p,o.options,i)
    if s
        get(o.debugDictionary,:Stop,DebugDivider(""))(p,get_options(o),typemin(Int))
        get(o.debugDictionary,:All,DebugDivider(""))(p,get_options(o),typemin(Int))
    end
    return s
end