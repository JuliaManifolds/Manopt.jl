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
export initializeSolver!, doSolverStep!, evaluateStoppingCriterion, getSolverResult
"""
    initializeSolver!(p,o)
Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`DebugOptions`](@ref)` o`.
"""
function initializeSolver!(p::P,o::O) where {P <: Problem, O <: DebugOptions}
    sig1 = string( typeof(p) )
    sig2 = string( typeof( getOptions(o) ) )
    debug(p,o,:Solver,"Starting solver for the $sig1 and $sig2.\n");
    debug(p,o,:InitialCost,0);
    initializeSolver!(p,o.options)
end
"""
    doSolverStep!(p,o,iter)
Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o.options` and print `Debug`.
"""
function doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: DebugOptions}
    doSolverStep!(p,o.options,iter)
    for dbgType ∈ filter(e -> e ∉ [:stoppingCriterion, :Solver, :InitialCost, :FinalCost], o.debugActivated)
        debug(p,o,dbgType,iter)
    end
    stop, reason = evaluateStoppingCriterion(p,o,iter)
    if stop
        sig1 = string( typeof(p) )
        sig2 = string( typeof( getOptions(o) ) )
        debug(p,o,:stoppingCriterion,reason);
        debug(p,o,:FinalCost,iter);
        debug(p,o,:Solver,"Finished solver for the $sig1 and $sig2.\n");
    end    
end
"""
    evaluateStoppingCriterion(p,o,iter)
Evaluate, whether the stopping criterion for the [`Problem`](@ref)` p`
and the [`Options`](@ref)` o` after `iter`th iteration is met.
"""
function evaluateStoppingCriterion(p::P,o::O, iter) where {P <: Problem, O <: DebugOptions}
    return evaluateStoppingCriterion(p, o.options,iter)
end
"""
    getResult(p,o)
Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref)` o`.
"""
function getSolverResult(p::P,o::O) where {P <: Problem, O <: DebugOptions}
    return getSolverResult(p, o.options)
end