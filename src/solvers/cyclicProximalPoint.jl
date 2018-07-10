export cyclicProximalPoint, cPPDebug
doc"""
    cyclicProximalPoint(M, F, proximalMaps, x)
perform a cyclic proximal point algorithm.
# Input
* `M` : a manifold $\mathcal M$
* `F` : a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `proximalMaps`: an Array of proximal maps (`Function`s) `(λ,x) -> y` for the summands of $F$
* `x` : an initial value $x\in\mathcal M$

# Optional
the default values are given in brackets
* `debug` : (off) a tuple `(f,p,v)` of a DebugFunction `f`
  that is called with its settings dictionary `p` and a verbosity `v`. Existing
  fields of `p` are updated during the iteration from (iter, x, xnew, stepSize).
* `evaluationOrder <: EvalOrder` ( `LinearEvalOrder()` ) whether to use a randomly
  permuted sequence (`FixedRandomEvalOrder()`), a per cycle permuted sequence (`RandomEvalOrder()`)
  or the default linear one.
* `λ`  : ( `iter -> 1/iter` ) a function returning the (square summable but not
  summable) sequence of λi
* `returnReason` : ( `false` ) whether or not to return the reason as second return
   value.
* `stoppingCriterion` : ( `(i,x,xnew,λ) -> ...` ) a function indicating when to stop.
  Default is to stop if the norm of the iterates change $d_{\mathcal M}(x,x_{\text{new}})$ is less
  than $10^{-4}$ or the iterations `i` exceed 500.

# Output
* `xOpt` : the resulting (approximately critical) point of gradientDescent
* `reason` : (if activated) a String containing the stopping criterion stopping
  reason.
"""
function cyclicProximalPoint{Mc <: Manifold, MP <: MPoint}(M::Mc,
        F::Function, proximalMaps::Array{Function,N} where N, x::MP;
        evaluationOrder::EvalOrder = LinearEvalOrder(),
        stoppingCriterion::Function = (i,x,xnew,λ) -> (distance(M,x,xnew) < 10.0^-4 || i > 499, (i>499) ? "max Iter $(i) reached.":"Minimal change small enough."),
        λ = iter -> 1/iter,
        returnReason=false,
        kwargs... #especially may contain debug
    )
    # TODO Test Input
    p = getProximalProblem(M,F,proximalMaps)
    o = CyclicProximalPointOptions(x,stoppingCriterion,λ,evaluationOrder,lineSearch[1],lineSearch[2])
    # create default here to check if the user provided a debug and still have the typecheck
    debug::Tuple{Function,Dict{String,Any},Int}= (x::Dict{String,Any}->print(""),Dict{String,Any}(),0);
    kwargs=Dict(kwargs)
    if haskey(kwargs, :debug) # if a key is given -> decorate Options.
        debug = kwargs[:debug]
        o = DebugDecoOptions(o,debug[1],debug[2],debug[3])
    end
    x,r = cyclicProximalPoint(p,o)
    if returnReason
        return x,r;
    else
        return x;
    end
end
doc"""
    cyclicProximalPoint(p,o)
compute a cyclic proximal point algorithm (CPPA) for the
`ProximalProblem p` and `CyclicProximalPointOptions o`.
"""
function cyclicProximalPoint{P<:ProximalProblem, O<:CyclicProximalPointOptions}(p::P,o::O)
    x = o.x0
    M = p.M
    stop = false;
    iter = 0;
    c = len(p.proximalMaps);
    order = updateOrder(c,0,collect(1:c),o.OrderType)
    xnew = x;
    while !stop
        iter += 1;
        order = updateOrder(c,iter,order,o.OrderType)
        λi = o.λ(iter)
        for k=order
            xnew = getProximalMap(p,λi,xnew,k)
        end
        stop, reason = evaluateStoppingCriterion(o,iter,x,xnew,λ)
        cPPDebug(o,iter,x,xnew,λ,reason);
        x = xnew;
    end
    return x,reason;
end
updateOrder(n,i,o,::LinearEvalOrder) = o;
updateOrder(n,i,o,::RandomEvalOrder) = collect(1:n)[randperm(length(X))];
updateOrder(n,i,o,::FixedRandomEvalOrder) = (i==0) ? collect(1:n)[randperm(length(X))] : o;

function cPPDebug{O <: Options, MP <: MPoint}(o::O,iter::Int,x::MP,xnew::MP,λ::Float64,reason::String)
    if getOptions(o) != o
        cPPDebug(getOptions(o),iter,x,xnew,λ,reason)
    end
end
function cPPDebugDebug{D <: DebugDecoOptions{O} where O<:Options, MP <: MPoint}(o::D,iter::Int,x::MP,xnew::MP,λ::Float64,reason::String)
    # decorate
    d = o.debugOptions;
    # Update values for debug
    if haskey(d,"x")
        d["x"] = x;
    end
    if haskey(d,"xnew")
        d["xnew"] = xnew;
    end
    if haskey(d,"λ")
        d["λ"] = λ;
    end
    if haskey(d,"Iteration")
        d["Iteration"] = iter;
    end
    o.debugFunction(d);
    if getVerbosity(o) > 2 && length(reason) > 0
        print(reason)
    end
end
