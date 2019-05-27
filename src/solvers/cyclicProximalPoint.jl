export cyclicProximalPoint
@doc doc"""
    cyclicProximalPoint(M, F, proximalMaps, x)
perform a cyclic proximal point algorithm.
# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `proximalMaps` – an Array of proximal maps (`Function`s) `(λ,x) -> y` for the summands of $F$
* `x` – an initial value $x\in\mathcal M$

# Optional
the default values are given in brackets
* `evaluationOrder` – ( [`LinearEvalOrder`](@ref) ) – whether
  to use a randomly permuted sequence ([`FixedRandomEvalOrder`](@ref)), a per
  cycle permuted sequence ([`RandomEvalOrder`](@ref)) or the default linear one.
* `λ` – ( `iter -> 1/iter` ) a function returning the (square summable but not
  summable) sequence of λi
* `stoppingCriterion` – ([`stopWhenAny`](@ref)`(`[`stopAfterIteration`](@ref)`(5000),`[`stopWhenChangeLess`](@ref)`(10.0^-8))`) a [`StoppingCriterion`](@ref).

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
* `record` – (if activated) a String containing the stopping criterion stopping
  reason.
"""
function cyclicProximalPoint(M::Mc,
  F::Function, proximalMaps::Array{Function,N} where N, x0::MP;
  evaluationOrder::EvalOrder = LinearEvalOrder(),
  stoppingCriterion::StoppingCriterion = stopWhenAny( stopAfterIteration(5000), stopWhenChangeLess(10.0^-12)),
  λ = i -> typicalDistance(M)/2/i,
  kwargs... #decorator options
  ) where {Mc <: Manifold, MP <: MPoint}
    p = ProximalProblem(M,F,proximalMaps)
    o = CyclicProximalPointOptions(x0,stoppingCriterion,λ,evaluationOrder)
    
    o = decorateOptions(o; kwargs...)
    resultO = solve(p,o)
    if hasRecord(resultO)
        return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
    end
    return getSolverResult(p,resultO)
end
function initializeSolver!(p::ProximalProblem, o::CyclicProximalPointOptions)
    c = length(p.proximalMaps)
    o.order = updateOrder(c,0,[1:c...],o.orderType)
end
function doSolverStep!(p::ProximalProblem, o::CyclicProximalPointOptions, iter)
    c = length(p.proximalMaps)
    λi = o.λ(iter)
    for k=o.order
        o.x = getProximalMap(p,λi,o.x,k)
    end
    o.order = updateOrder(c,iter,o.order,o.orderType)
end
getSolverResult(p::ProximalProblem, o::CyclicProximalPointOptions) = o.x
updateOrder(n,i,o,::LinearEvalOrder) = o
updateOrder(n,i,o,::RandomEvalOrder) = collect(1:n)[randperm(length(o))]
updateOrder(n,i,o,::FixedRandomEvalOrder) = (i==0) ? collect(1:n)[randperm(length(o))] : o
