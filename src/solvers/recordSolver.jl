#
#
# Encapulates arbitrary solvers by a debug solver, which is induced by Decoration Options
#
#
export initializeSolver!, doSolverStep!, evaluateStoppingCriterion, getSolverResult
"""
    initializeSolver!(p,o)

Initialize the solver to the optimization [`Problem`](@ref) by initializing
the encapsulated `options` from within the [`RecordOptions`](@ref)` o`.
"""
function initializeSolver!(p::P,o::O) where {P <: Problem, O <: RecordOptions}
    initializeSolver!(p,o.options)
end
"""
    doSolverStep!(p,o,iter)

Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o.options` and [`record!`](@ref) the result
"""
function doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: RecordOptions}
    doSolverStep!(p,o.options,iter)
    record!(p,o,iter)
end
"""
    evaluateStoppingCriterion(p,o,iter)

Evaluate, whether the stopping criterion for the [`Problem`](@ref)` p`
and the [`Options`](@ref)` o` after `iter`th iteration is met.
"""
function evaluateStoppingCriterion(p::P,o::O, iter) where {P <: Problem, O <: RecordOptions}
    return evaluateStoppingCriterion(p, o.options,iter)
end
"""
    getResult(p,o)

Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref)` o`.
"""
function getSolverResult(p::P,o::O) where {P <: Problem, O <: RecordOptions}
    return getSolverResult(p, o.options)
end