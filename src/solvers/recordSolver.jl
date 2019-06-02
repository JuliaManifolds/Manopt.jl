#
#
# Encapulates arbitrary solvers by a debug solver, which is induced by Decoration Options
#
#
export initializeSolver!, doSolverStep!, getSolverResult, stopSolver!
"""
    initializeSolver!(p,o)

Initialize the solver to the optimization [`Problem`](@ref) by initializing
the encapsulated `options` from within the [`RecordOptions`](@ref)` o`.
"""
function initializeSolver!(p::P,o::O) where {P <: Problem, O <: RecordOptions}
    initializeSolver!(p,o.options)
    get(o.recordDictionary, :Start, RecordGroup() )(p, getOptions(o), 0)
    get(o.recordDictionary, :All,   RecordGroup() )(p, getOptions(o), 0)
end
"""
    doSolverStep!(p,o,iter)

Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o.options` and record the result(s).
"""
function doSolverStep!(p::P,o::O, i) where {P <: Problem, O <: RecordOptions}
    doSolverStep!(p,o.options,i)
    get(o.recordDictionary, :Step, RecordGroup() )(p, getOptions(o), i)
    get(o.recordDictionary, :All,   RecordGroup() )(p, getOptions(o), i)
end
"""
    getSolverResult(p,o)

Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref)` o`.
"""
function getSolverResult(p::P,o::O) where {P <: Problem, O <: RecordOptions}
    return getSolverResult(p, o.options)
end
"""
    stopSolver!(p,o,i)

determine whether the solver for [`Problem`](@ref) `p` and the
[`RecordOptions`](@ref) `o` should stop at iteration `i`. 
If so, do a (final) record to `:All` and `:Stop`.
"""
function stopSolver!(p::P,o::O,i::Int) where {P <: Problem, O <: RecordOptions}
    s = stopSolver!(p,o.options,i)
    if s
        get(o.recordDictionary, :Stop, RecordGroup() )(p, getOptions(o), typemin(Int) )
        get(o.recordDictionary, :All,   RecordGroup() )(p, getOptions(o), typemin(Int ))
    end
    return s
end