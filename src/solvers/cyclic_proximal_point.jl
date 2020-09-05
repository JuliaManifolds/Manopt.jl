@doc raw"""
    cyclic_proximal_point(M, F, proxes, x)

perform a cyclic proximal point algorithm.

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `proxes` – an Array of proximal maps (`Function`s) `(λ,x) -> y` for the summands of $F$
* `x` – an initial value $x ∈ \mathcal M$

# Optional
the default values are given in brackets
* `evaluationOrder` – ( [`LinearEvalOrder`](@ref) ) – whether
  to use a randomly permuted sequence ([`FixedRandomEvalOrder`](@ref)), a per
  cycle permuted sequence ([`RandomEvalOrder`](@ref)) or the default linear one.
* `λ` – ( `iter -> 1/iter` ) a function returning the (square summable but not
  summable) sequence of λi
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(5000),`[`StopWhenChangeLess`](@ref)`(10.0^-8))`) a [`StoppingCriterion`](@ref).
* `return_options` – (`false`) – if actiavated, the extended result, i.e. the
  complete [`Options`](@ref) are returned. This can be used to access recorded values.
  If set to false (default) just the optimal value `xOpt` if returned
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function cyclic_proximal_point(M::MT,
  F::TF, proxes::Union{Tuple,AbstractVector}, x0;
  evaluationOrder::EvalOrder = LinearEvalOrder(),
  stopping_criterion::StoppingCriterion = StopWhenAny( StopAfterIteration(5000), StopWhenChangeLess(10.0^-12)),
  λ = i -> 1/i,
  return_options=false,
  kwargs... #decorator options
  ) where {MT <: Manifold,TF}
    p = ProximalProblem(M,F,proxes)
    o = CyclicProximalPointOptions(x0,stopping_criterion,λ,evaluationOrder)

    o = decorate_options(o; kwargs...)
    resultO = solve(p,o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end
function initialize_solver!(p::ProximalProblem, o::CyclicProximalPointOptions)
    c = length(p.proxes)
    o.order = collect(1:c)
    update_cpp_order!(o.order,c,0,o.orderType)
end
function step_solver!(p::ProximalProblem, o::CyclicProximalPointOptions, iter)
    c = length(p.proxes)
    λi = o.λ(iter)
    for k=o.order
        o.x = getProximalMap(p,λi,o.x,k)
    end
    update_cpp_order!(o.order,c,iter,o.orderType)
end
get_solver_result(o::CyclicProximalPointOptions) = o.x
update_cpp_order!(o,n,i,::LinearEvalOrder) = o
update_cpp_order!(o,n,i,::RandomEvalOrder) = shuffle!(o)
update_cpp_order!(o,n,i,::FixedRandomEvalOrder) = (i==0) ? shuffle!(o) : o
