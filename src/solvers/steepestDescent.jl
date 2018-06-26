#
# A simple steepest descent algorithm implementation
#
export steepestDescent
"""
    steepestDescent(problem)

    given a problem with at least a gradient and a costFunction are given
"""
function steepestDescent{P <: DescentProblem}(problem::P)
    stop = false
    iter = 0
    x = problem.initX
    s = problem.lineSearchProblem.initialStepsize
    M = problem.manifold

    while !stop
        if getVerbosity(problem) > 2
            global reason
        end
        ξ = getGradient(problem,x)
        s = getStepsize(problem,x,ξ,s)
        xnew = exp(M,x,-s*ξ)
        iter=iter+1
        (stop, reason) = evaluateStoppingCriterion(problem,iter,ξ,x,xnew)
        x=xnew
    end
    if getVerbosity(problem) > 2
        print(reason)
    end
    return x
end
