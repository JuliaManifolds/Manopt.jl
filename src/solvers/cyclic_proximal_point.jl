@doc raw"""
    cyclic_proximal_point(M, F, proxes, x)

perform a cyclic proximal point algorithm.

# Input

* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `proxes` – an Array of proximal maps (`Function`s) `(λ,x) -> y` for the summands of ``F``
* `x` – an initial value ``x ∈ \mathcal M``

# Optional
the default values are given in brackets
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the proximal maps work by allocation (default) form `prox(M, λ, x)`
  or [`MutatingEvaluation`](@ref) in place, i.e. is of the form `prox!(M, y, λ, x)`.
* `evaluation_order` – (`:Linear`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`Random`) or the default linear one.
* `λ` – ( `iter -> 1/iter` ) a function returning the (square summable but not
  summable) sequence of λi
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(5000),`[`StopWhenChangeLess`](@ref)`(10.0^-8))`) a [`StoppingCriterion`](@ref).
* `return_options` – (`false`) – if activated, the extended result, i.e. the
  complete [`Options`](@ref) are returned. This can be used to access recorded values.
  If set to false (default) just the optimal value `x_opt` if returned
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `x_opt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function cyclic_proximal_point(
    M::Manifold, F::Function, proxes::Union{Tuple,AbstractVector}, x0; kwargs...
)
    x_res = allocate(x0)
    copyto!(M, x_res, x0)
    return cyclic_proximal_point!(M, F, proxes, x_res; kwargs...)
end

@doc raw"""
    cyclic_proximal_point!(M, F, proxes, x)

perform a cyclic proximal point algorithm in place of `x`.

# Input

* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `proxes` – an Array of proximal maps (`Function`s) `(λ,x) -> y` for the summands of ``F``
* `x` – an initial value ``x ∈ \mathcal M``

for all options, see [`cyclic_proximal_point`](@ref).
"""
function cyclic_proximal_point!(
    M::Manifold,
    F::Function,
    proxes::Union{Tuple,AbstractVector},
    x0;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    evaluation_order::Symbol=:Linear,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(5000), StopWhenChangeLess(10.0^-12)
    ),
    λ=i -> 1 / i,
    return_options=false,
    kwargs..., #decorator options
)
    p = ProximalProblem(M, F, proxes; evaluation=evaluation)
    o = CyclicProximalPointOptions(x0, stopping_criterion, λ, evaluation_order)

    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end
function initialize_solver!(p::ProximalProblem, o::CyclicProximalPointOptions)
    c = length(p.proximal_maps!!)
    o.order = collect(1:c)
    (o.order_type == :FixedRandom) && shuffle!(o.order)
    return o
end
function step_solver!(p::ProximalProblem, o::CyclicProximalPointOptions, iter)
    c = length(p.proximal_maps!!)
    λi = o.λ(iter)
    for k in o.order
        get_proximal_map!(p, o.x, λi, o.x, k)
    end
    (o.order_type == :Random) && shuffle(o.order)
    return o
end
get_solver_result(o::CyclicProximalPointOptions) = o.x
