#
# A simple steepest descent algorithm implementation
#
"""
    steepestDescent(problem)

    given a problem with at least a gradient and a costFunction are given
"""
function steepestDescent{P <: Problem}(problem::P)
    stop=false
    iter = 0
    x = problem.initX
    while !stop
        ξi = getGradient(problem,x)
        xnew = exp.(x,-ξi)
        iter=iter+1
        (stop, reason) = evaluateStoppingCriterion(problem,iter,x,xnew)
        x=xnew
    end
    if getVerbosity(problem) > 2
        print(reason)
    end
end
