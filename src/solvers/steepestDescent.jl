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
        # Debug
        if getVerbosity(problem) > 4 && !isnull(problem.debugSettings) && !isnull(problem.debugFunction)
            d = get(problem.debugSettings);
            if haskey(d,"x")
                d["x"] = xnew
            end
            if haskey(d,"xold")
                d["xold"] = x
            end
            if haskey(d,"Iteration")
                d["Iteration"] = iter
            end
            get(problem.debugFunction)(d);
        end
    end
    if getVerbosity(problem) > 2 && length(reason) > 0
        print(reason)
    end
    return x
end
