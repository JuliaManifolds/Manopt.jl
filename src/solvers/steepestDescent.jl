#
# A simple steepest descent algorithm implementation
#
export steepestDescent, gradDescDebug
@doc doc"""
    steepestDescent(M, F, ∇F, x)
perform a steepestDescent $x_{k+1} = \exp_{x_k} s_k\nabla f(x_k)$ with different
choices of $s_k$ available (see `lineSearch` option below).

# Input
* `M` : a manifold $\mathcal M$
* `F` : a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∇F`: the gradient $\nabla F\colon\mathcal M\to T\mathcal M$ of F
* `x` : an initial value $x\in\mathcal M$

# Optional
* `debug` : (off) a tuple `(f,p,v)` of a DebugFunction `f`
  that is called with its settings dictionary `p` and a verbosity `v`. Existing
  fields of `p` are updated during the iteration from (iter, x, xnew, stepSize).
* `lineSearch` : (`(p,lO) -> 1, lO::`[`LineSearchOptions`](@ref)`)`) A tuple `(lS,lO)`
  consisting of a line search function `lS` (called with two arguments, the
  problem `p` and the lineSearchOptions `lO`) with its LineSearchOptions `lO`.
  The default is a constant step size 1.
* `retraction` : (`exp`) a retraction(M,x,ξ) to use.
* `returnReason` : (`false`) whether or not to return the reason as second return
   value.
* `stoppingCriterion` : (`(i,ξ,x,xnew) -> ...`) a function indicating when to stop.
  Default is to stop if the norm of the gradient $\lVert \xi\rVert_x$ is less
  than $10^{-4}$ or the iterations `i` exceed 500.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
* `reason` - if activated a String containing the stopping criterion stopping
  reason.
"""
function steepestDescent(M::mT,
        F::Function, ∇F::Function, x::MP;
        lineSearch::Tuple{Function,Options}= ( (p::GradientProblem{mT},
            o::LineSearchOptions) -> 1, SimpleLineSearchOptions() ),
        retraction::Function = exp,
        stoppingCriterion::Function = (i,ξ,x,xnew) -> (norm(M,x,ξ) < 10.0^-4 || i > 499, (i>499) ? "max Iter $(i) reached." : "critical point reached"),
        returnReason=false,
        kwargs... #especially may contain debug
    ) where {mT <: Manifold, MP <: MPoint}
    # TODO Test Input
    p = GradientProblem(M,F,∇F)
    o = GradientDescentOptions(x,stoppingCriterion,retraction,lineSearch[1],lineSearch[2])
    # create default here to check if the user provided a debug and still have the typecheck
    debug::Tuple{Function,Dict{String,Any},Int}= (x::Dict{String,Any}->print(""),Dict{String,Any}(),0);
    kwargs=Dict(kwargs)
    if haskey(kwargs, :debug) # if a key is given -> decorate Options.
        debug = kwargs[:debug]
        o = DebugOptions(o,debug[1],debug[2],debug[3])
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
performs a steepestDescent based on a GradientProblem containing all information
for the `problem <: GradientProblem` (Manifold, costFunction, Gradient)  and
Options for the solver (`x0 <: MPoint`, `lineSearch` and `lineSearchOptions`,
`retraction` and stoppingCriterion` functions); see the general Interface
for details on these parameters.
"""
function steepestDescent(p::P, o::O) where {P <: GradientProblem, O <: Options}
    stop::Bool = false
    reason::String="";
    iter::Integer = 0
    x = getOptions(o).x0
    s = getOptions(o).lineSearchOptions.initialStepsize
    M = p.M
    while !stop
        ξ = getGradient(p,x)
        s = getStepsize(p,getOptions(o),x,s)
        xnew = getOptions(o).retraction(M,x,-s*ξ)
        iter=iter+1
        stop, reason = evaluateStoppingCriterion(getOptions(o),iter,ξ,x,xnew)
        gradDescDebug(o,iter,x,xnew,ξ,s,reason)
        x=xnew
    end
    return x,reason
end
# fallback - do nothing just unpeel
function gradDescDebug(o::O,iter::Int,x::MP,xnew::MP,ξ::MT,s::Float64,reason::String) where {O <: Options, MP <: MPoint, MT <: TVector}
    if getOptions(o) != o
        gradDescDebug(getOptions(o),iter,x,xnew,ξ,s,reason)
    end
end
function gradDescDebug(o::D,iter::Int,x::MP,xnew::MP,ξ::MT,s::Float64,reason::String) where {D <: DebugOptions, MT <: TVector, MP <: MPoint}
    # decorate
    d = o.debugOptions;
    # Update values for debug
    if haskey(d,"x")
        d["x"] = xnew;
    end
    if haskey(d,"xnew")
        d["xnew"] = x;
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
