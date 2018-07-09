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
        debug             - a tuple (f,p,v) of a DebugFunction f that is called
                                with its settings dictionary fS that is updated
                                during iterations (iter, x, xnew, stepSize) and
                                a verbosity v
        lineSearch        – a tuple (l,p) a line search function with its
                                lineSeachProblem p. The default is a constant
                                step size 1.
        returnReason      - (false) whether or not to return the reason as second element
        retraction        - a retraction to use. Set to exp by standard
        stoppingCriterion – a function indicating when to stop. Default is to
            stop if ||gradF(x)||<10^-4 or Iterations > 500

    OUTPUT
        xOpt – the resulting point of gradientDescent
"""
function steepestDescent{Mc <: Manifold, MP <: MPoint}(M::Mc,
        F::Function, gradF::Function, x::MP;
        lineSearch::Tuple{Function,Options}= ( (p::GradientProblem{Mc},
            o::GradientLineSearchOptions) -> 0.5, GradientLineSearchOptions(x)),
        retraction::Function = exp,
        stoppingCriterion::Function = (i,ξ,x,xnew) -> (norm(M,x,ξ) < 10.0^-4 || i > 499, (i>499) ? "max Iter $(i) reached.":"critical point reached"),
        returnReason=false,
        kwargs... #especially may contain debug
    )
    # TODO Test Input
    p = GradientProblem(M,F,gradF)
    o = GradientDescentOptions(x,stoppingCriterion,retraction,lineSearch[1],lineSearch[2])
    # create default here to check if the user provided a debug and still have the typecheck
    debug::Tuple{Function,Dict{String,Any},Int}= (x::Dict{String,Any}->print(""),Dict{String,Any}(),0);
    kwargs=Dict(kwargs)
    if haskey(kwargs, :debug) # if a key is given -> decorate Options.
        debug = kwargs[:debug]
        o = DebugDecoOptions(o,debug[1],debug[2],debug[3])
    end
    x,r = steepestDescent(p,o)
    if returnReason
        return x,r;
    else
        return x;
    end
end
"""
    steepestDescent(problem,options)
        performs a steepestDescent based on a DescentProblem struct.
        sets “x” “xold” and “Iteration” in a non-null debugOptions Dict.
"""
function steepestDescent{P <: GradientProblem, O <: Options}(p::P, o::O)
    stop::Bool = false
    reason::String="";
    iter::Integer = 0
    x = getOptions(o).initX
    s = getOptions(o).lineSearchOptions.initialStepsize
    M = p.M
    while !stop
        ξ = getGradient(p,x)
        s = getStepsize(p,getOptions(o),x,s)
        xnew = getOptions(o).retraction(M,x,-s*ξ)
        iter=iter+1
        (stop, reason) = evaluateStoppingCriterion(getOptions(o),iter,ξ,x,xnew)
        gradDescDebug(o,iter,x,xnew,ξ,s,reason);
        x=xnew;
    end
    return x,reason
end
# fallback - do nothing just unpeel
function gradDescDebug{O <: Options, MP <: MPoint, MT <: TVector}(o::O,iter::Int,x::MP,xnew::MP,ξ::MT,s::Float64,reason::String)
    if getOptions(o) != o
        gradDescDebug(getOptions(o),iter,x,xnew,ξ,s,reason)
    end
end
function gradDescDebug{D <: DebugDecoOptions{O} where O<:Options, MT <: TVector, MP <: MPoint}(o::D,iter::Int,x::MP,xnew::MP,ξ::MT,s::Float64,reason::String)
    # decorate
    d = o.debugOptions;
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
    # one could also activate a debug checking stept size to -grad if problem and chekNegative are given?
    o.debugFunction(d);
    if getVerbosity(o) > 2 && length(reason) > 0
        print(reason)
    end
end
