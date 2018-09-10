#
# A simple steepest descent algorithm implementation
#
export subgradientMethod, subGradDescDebug
@doc doc"""
    subGradientMethod(M, F, ∂F, x)
perform a subgradient method $x_{k+1} = \mathcal R({x_k}, s_k∂F(x_k))$,

where $\mathcal R$ is a retraction, $s_k$ can be specified as a function but is
usually set to a constant value. Though the subgradient might be set valued,
the argument `∂F` should always return _one_ element from the subgradient.

# Input
* `M` : a manifold $\mathcal M$
* `F` : a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∂F`: the (sub)gradient $\partial F\colon\mathcal M\to T\mathcal M$ of F
  restricted to always only returning one value/element from the subgradient
* `x` : an initial value $x\in\mathcal M$

# Optional
* `debug` : (off) a tuple `(f,p,v)` of a DebugFunction `f`
  that is called with its settings dictionary `p` and a verbosity `v`. Existing
  fields of `p` are updated during the iteration from (iter, x, xnew, stepSize).
* `stepSize` : (`(x,ξ) -> 0.001`)
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
function subgradientDescent(M::mT,
        F::Function, ∂F::Function, x::MP;
        retraction::Function = exp,
        stepSize::Function = (i,ξ) -> 0.001,
        stoppingCriterion::Function = (i,ξ,x,xnew) -> (i>4999) ? "max Iter $(i) reached." : "",
        returnReason=false,
        kwargs... #especially may contain debug
    ) where {mT <: Manifold, MP <: MPoint}
    p = SubGradientProblem(M,F,∂F)
    o = SubGradientMethodOptions(x,stoppingCriterion,retraction,stepSize)
    # create default here to check if the user provided a debug and still have the typecheck
    debug::Tuple{Function,Dict{String,Any},Int}= (x::Dict{String,Any}->print(""),Dict{String,Any}(),0);
    kwargs=Dict(kwargs)
    if haskey(kwargs, :debug) # if a key is given -> decorate Options.
        debug = kwargs[:debug]
        o = DebugDecoOptions(o,debug[1],debug[2],debug[3])
    end
    x,r = subGradientMethod(p,o)
    if returnReason
        return x,r;
    else
        return x;
    end
end
"""
    subGradientMethod(problem,options)
performs a steepestDescent based on a GradientProblem containing all information
for the `problem <: GradientProblem` (Manifold, costFunction, Gradient)  and
Options for the solver (`x0 <: MPoint`, `lineSearch` and `lineSearchOptions`,
`retraction` and stoppingCriterion` functions); see the general Interface
for details on these parameters.
"""
function subGradientMethod(p::P, o::O) where {P <: SubGradientProblem, O <: Options}
    stop::Bool = false
    reason::String="";
    iter::Integer = 0
    x = getOptions(o).x0
    M = p.M
    while !stop
        ξ = getSubGradient(p,x)
        s = getStepsize(p,getOptions(o),x,ξ)
        xnew = getOptions(o).retraction(M,x,-s*ξ)
        iter=iter+1
        (stop, reason) = evaluateStoppingCriterion(getOptions(o),iter,ξ,x,xnew)
        subGradDescDebug(o,iter,x,xnew,ξ,s,reason)
        x=xnew
    end
    return x,reason
end
# fallback - do nothing just unpeel
function subGradDescDebug(o::O,iter::Int,x::MP,xnew::MP,ξ::MT,s::Float64,reason::String) where {O <: Options, MP <: MPoint, MT <: TVector}
    if getOptions(o) != o
        subGradDescDebug(getOptions(o),iter,x,xnew,ξ,s,reason)
    end
end
function subGradDescDebug(o::D,iter::Int,x::MP,xnew::MP,ξ::MT,s::Float64,reason::String) where {D <: DebugDecoOptions, MT <: TVector, MP <: MPoint}
    # decorate
    d = o.debugOptions;
    # Update values for debug
    if haskey(d,"x")
        d["x"] = xnew;
    end
    if haskey(d,"xnew")
        d["xnew"] = x;
    end
    if haskey(d,"subgradient")
        d["subgradient"] = ξ;
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
