export cyclicProximalPoint
@doc doc"""
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
* `evaluationOrder <: [`EvalOrder`](@ref) ( [`LinearEvalOrder`](@ref) ) whether
  to use a randomly permuted sequence ([`FixedRandomEvalOrder`](@ref)), a per
  cycle permuted sequence ([`RandomEvalOrder`](@ref)) or the default linear one.
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
function cyclicProximalPoint(M::Mc,
        F::Function, proximalMaps::Array{Function,N} where N, x::MP;
        evaluationOrder::EvalOrder = LinearEvalOrder(),
        stoppingCriterion::Function = (i,x,xnew,λ) -> (distance(M,x,xnew) < 10.0^-8 || i > 4999,
            (i>4999) ? "Maximal number of Iterations ($(i)) reached." :
            ( (distance(M,x,xnew) < 10.0^-8) ? "#$(i) | Last change in the iterate ($(distance(M,x,xnew))) below minimal change." :
            "" )),
        λ = iter -> typicalDistance(M)/iter,
        returnReason=false,
        kwargs... #especially may contain debug
    ) where {Mc <: Manifold, MP <: MPoint}
    # TODO Test Input
    p = ProximalProblem(M,F,proximalMaps)
    o = CyclicProximalPointOptions(x,stoppingCriterion,λ,evaluationOrder)
    # create default here to check if the user provided a debug and still have the typecheck
    debug::Tuple{Function,Dict{String,Any},Int}= (x::Dict{String,Any}->print(""),Dict{String,Any}(),0);
    kwargs=Dict(kwargs)
    if haskey(kwargs, :debug) # if a key is given -> decorate Options.
        debug = kwargs[:debug]
        o = DebugOptions(o,debug[1],debug[2],debug[3])
    end
    x,r = cyclicProximalPoint(p,o)
    if returnReason
        return x,r
    else
        return x
    end
end
"""
    cyclicProximalPoint(p,o)
compute a cyclic proximal point algorithm (CPPA) for the
[`ProximalProblem`](@ref)` p` and [`CyclicProximalPointOptions`](@ref)` o`.
"""
function cyclicProximalPoint(p::P,o::O) where {P<:ProximalProblem, O<:Options}
    lO = getOptions(o);
    x = lO.x0
    M = p.M
    stop = false; iter = 0;
    c = length(p.proximalMaps);
    order = updateOrder(c,0,collect(1:c),lO.orderType)
    xnew = x;
    reason::String="";
    while !stop
        iter += 1
        order = updateOrder(c,iter,order,lO.orderType)
        λi = lO.λ(iter)
        for k=order
            xnew = getProximalMap(p,λi,xnew,k)
        end
        stop, reason = evaluateStoppingCriterion(lO,iter,x,xnew,λi)
        if optionsHasDebug(o)
            updateDebugValues!(o,Dict("x" => x, "xnew" => xnew, "λ" => λi, "Iteration" => iter, "StopReason" => reason));
            Debug(o)
        end
        x = xnew
    end
    return x, reason
end
updateOrder(n,i,o,::LinearEvalOrder) = o
updateOrder(n,i,o,::RandomEvalOrder) = collect(1:n)[randperm(length(X))]
updateOrder(n,i,o,::FixedRandomEvalOrder) = (i==0) ? collect(1:n)[randperm(length(X))] : o
