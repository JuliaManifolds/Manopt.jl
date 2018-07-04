#
# A simple steepest descent algorithm implementation
#
export steepestDescent
"""
    steepestDescent(M,F,gradF,x)
        perform a steepestDescent
    INPUT
            M – a manifold
            F - a cost function to minimize
        gradF - the gradient of F
            x - an initial value of F

    OPTIONAL
        debug             - a tuple (f,p) of a DebugFunction f that is called
                                with its settings dictionary fS that is updated
                                during iterations (iter, x, xnew, stepSize)
        lineSearch        – a tuple (l,p) a line search function with its
                                lineSeachProblem p. The default is a constant
                                step size 1.
        retraction        - a retraction to use. Set to exp by standard
        stoppingCriterion – a function indicating when to stop. Default is to
            stop if ||gradF(x)||<10^-4 or Iterations > 500
        verbosity         - set console verbosity.
        useCache          - use a cache if available. Set to false by default

    OUTPUT
        xOpt – the resulting point of gradientDescent
"""
function steepestDescent{Mc <: Manifold, MP <: MPoint}(M::Mc,
        F::Function, gradF::Function, x::MP;
        debug=(Nullable{Function}(),Nullable{Dict}()),
        lineSearch=((p,M,gradF,ξ)->1/2,LineSearchProblem(M,F)),
        retraction=exp,
        stoppingCriterion= (iter,ξ,x,xnew) -> norm(M,x,ξ) < 10^-4 || Iterations > 500,
        useCache=false,
        verbosity=0
    )
    # TODO Test Input
    p = DescentProblem(M,F,gradF,x,Retraction,lineSeach[1],lineSeach[2],useCache,Verbosity,debug[1],debug[2])
    return steepestDescent(p)
end
"""
    steepestDescent(problem)
        performs a steepestDescent based on a DescentProblem struct.
        sets “x” “xold” and “Iteration” in a non-null debugSettings Dict.
"""
function steepestDescent{P <: DescentProblem}(problem::P)
    stop = false
    iter = 0
    x = problem.initX
    s = problem.lineSearchProblem.initialStepsize
    M = problem.manifold
    retr = problem.retraction
    while !stop
        if getVerbosity(problem) > 2
            global reason
        end
        ξ = getGradient(problem,x)
        s = getStepsize(problem,x,ξ,s)
        xnew = retr(M,x,-s*ξ)
        iter=iter+1
        (stop, reason) = evaluateStoppingCriterion(problem,iter,ξ,x,xnew)
        # Debug
        if getVerbosity(problem) > 4 && !isnull(problem.debugSettings) && !isnull(problem.debugFunction)
            d = get(problem.debugSettings);
            if haskey(d,"x")
                d["x"] = xnew;
            end
            if haskey(d,"xold")
                d["xold"] = x;
            end
            if haskey(d,"gradient")
                d["gradient"] = ξ;
            end
            if haskey(d,"Iteration")
                d["Iteration"] = iter;
            end
            if haskey(d,"Stepsize")
                d["Stepsize"] = s;
            end
            get(problem.debugFunction)(d);
        end
    end
    if getVerbosity(problem) > 2 && length(reason) > 0
        print(reason)
    end
    return x,reason
end
