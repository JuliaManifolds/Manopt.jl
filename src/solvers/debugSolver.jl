#
#
# Encapulates arbitrary solvers by a debug solver, which is induced by Decoration Options
#
#
#
#
# A general framework for solvers of problems on Manifolds
#
# This file introduces fallbacks for not yet implemented parts and the general
# function to run the solver
export initializeSolver!, doSolverStep!, getSolverResult, stopSolver!
"""
    initializeSolver!(p,o)
Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`DebugOptions`](@ref)` o`.
"""
function initializeSolver!(p::P,o::O) where {P <: Problem, O <: DebugOptions}
    initializeSolver!(p,o.options)
    get(o.debugDictionary,:Start,DebugDivider(""))(p,getOptions(o),0)
    get(o.debugDictionary,:All,DebugDivider(""))(p,getOptions(o),0)
end
"""
    doSolverStep!(p,o,iter)
Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o.options` and print `Debug`.
"""
function doSolverStep!(p::P,o::O, i) where {P <: Problem, O <: DebugOptions}
    doSolverStep!(p,o.options,i)
    get(o.debugDictionary,:Step,DebugDivider(""))(p,getOptions(o),i)
    get(o.debugDictionary,:All,DebugDivider(""))(p,getOptions(o),i)
end
"""
    getSolverResult(p,o)
Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref) `o`.
"""
function getSolverResult(p::P,o::O) where {P <: Problem, O <: DebugOptions}
    return getSolverResult(p, o.options)
end
"""
    stopSolver!(p,o,i)

determine whether the solver for [`Problem`](@ref) `p` and the [`DebugOptions`](@ref) `o`
should stop at iteration `i`. If so, print all debug from `:All` and `:Final`.
"""
function stopSolver!(p::P,o::O,i::Int) where {P <: Problem, O <: DebugOptions}
    s = stopSolver!(p,o.options,i)
    if s
        get(o.debugDictionary,:Stop,DebugDivider(""))(p,getOptions(o),typemin(Int))
        get(o.debugDictionary,:All,DebugDivider(""))(p,getOptions(o),typemin(Int))
    end
    return s
end