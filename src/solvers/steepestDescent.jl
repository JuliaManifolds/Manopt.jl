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
    p = GradientProblem(M,F,gradF)
    o = GradientDescentOptions(x,Retraction,lineSeach[1],lineSeach[2],Verbosity,debug[1],debug[2])
    return steepestDescent(p)
end
"""
    steepestDescent(problem)
        performs a steepestDescent based on a DescentProblem struct.
        sets “x” “xold” and “Iteration” in a non-null debugSettings Dict.
"""
function steepestDescent{P <: GradientProblem, O <: GradientDescentOptions}(problem::P, options::O)
    stop = false
    iter = 0
    x = getOptions(O).initX
    s = getOptions(O).initialStepsize
    while !stop
        xnew,s,iter,stop,reason = gradientStep(problem, options,iter,s,x)
    if getVerbosity(options) > 2 && length(reason) > 0
        print(reason)
    end
    return x,reason
end

function gradientStep{P <: GradientProblem, O <: GradientDescentOptions, MP <: MPoint}(problem::P, options::O,iter::Int,s::Float64,x::MP)
    M = problem.manifold
    ξ = getGradient(problem,x)
    s = getStepsize(problem,x,ξ,s)
    xnew = O.retraction(M,x,-s*ξ)
    iter=iter+1
    (stop, reason) = O.evaluateStoppingCriterion(problem,iter,ξ,x,xnew)
    return xnew,s,iter,stop,reason;
end

function gradientStep{P <: GradientProblem, D <: DebugDecoOptions{O} where O <: GradientDescentOptions, MP <: MPoint}(problem::P, options::D, iter::Int,s::Float64,x::MP)
    # for debug
    ξ = getGradient(problem,x)
    # classical debug
    xnew,s,iter,stop,reason = gradientStep(problem,options.options,iter,s,x);
    # decorate
    d = options.debugSettings;
    # Update values for debug
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
    options.debugFunction(d);
end
