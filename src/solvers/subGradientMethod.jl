#
# A simple steepest descent algorithm implementation
#
export subGradientMethod
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
function subGradientMethod(M::mT,
        F::Function, ∂F::Function, x::MP;
        retraction::Function = exp,
        stepSize::Function = (i,x,ξ) -> 0.01/i,
        stoppingCriterion::Function = (i,ξ,x,xnew) -> ((i>4999), (i>4999) ? "max Iter $(i) reached." : ""),
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
        o = DebugOptions(o,debug[1],debug[2],debug[3])
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
    xOpt = x
    M = p.M
    while !stop
        iter=iter+1
        ξ = getSubGradient(p,x)
        s = getStepsize(p,getOptions(o),iter,x,ξ)
        xnew = getOptions(o).retraction(M,x,-s*ξ)
        (stop, reason) = evaluateStoppingCriterion(o,iter,ξ,x,xnew)
        if getCost(p,xnew) < getCost(p,xOpt)
            xOpt=xnew
        end
        if optionsHasDebug(o)
            updateDebugValues!(o,Dict("Iteration"=>iter,"x"=>x,"xnew"=>xnew,"xopt"=>xOpt,"subgradient"=>ξ,"StepSize"=>s,"StopReason"=>reason))
            Debug(o)
        end
        x = xnew
    end
    return xOpt,reason
end
