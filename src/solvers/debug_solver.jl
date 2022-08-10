"""
    initialize_solver!(p::Problem, o::DebugOptions)

Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`DebugOptions`](@ref)` o`.

Since debug acts as a decorator this also calls the `initialize_solver!`
of the correpsonding internally stored options
"""
function initialize_solver!(p::Problem, o::DebugOptions)
    initialize_solver!(p, o.options)
    get(o.debugDictionary, :Start, DebugDivider(""))(p, get_options(o), 0)
    return get(o.debugDictionary, :All, DebugDivider(""))(p, get_options(o), 0)
end
"""
    step_solver!(p::Problem, o::DebugOptions, i)

    Do one iteration step (the `i`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o.options` and print the debug specified
"""
function step_solver!(p::Problem, o::DebugOptions, i)
    step_solver!(p, o.options, i)
    get(o.debugDictionary, :Step, DebugDivider(""))(p, get_options(o), i)
    return get(o.debugDictionary, :All, DebugDivider(""))(p, get_options(o), i)
end

"""
    stop_solver!(p,o,i)

determine whether the solver for [`Problem`](@ref) `p` and the [`DebugOptions`](@ref) `o`
should stop at iteration `i` by calling the function corresponding to the internally stored [`Options`](@ref).
If so, print debug from `:All` and `:Stop`.
"""
function stop_solver!(p::Problem, o::DebugOptions, i::Int)
    s = stop_solver!(p, o.options, i)
    if s
        get(o.debugDictionary, :Stop, DebugDivider(""))(p, get_options(o), typemin(Int))
        get(o.debugDictionary, :All, DebugDivider(""))(p, get_options(o), typemin(Int))
    end
    return s
end
