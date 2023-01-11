@doc raw"""
    cyclic_proximal_point(M, f, proxes_f, x)

perform a cyclic proximal point algorithm.

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `proxes_f` – an Array of proximal maps (`Function`s) `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the summands of ``f`` (see `evaluation`)
* `p` – an initial value ``p ∈ \mathcal M``

# Optional
the default values are given in brackets
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the proximal maps work by allocation (default) form `prox(M, λ, x)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `prox!(M, y, λ, x)`.
* `evaluation_order` – (`:Linear`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Random`) or the default linear one.
* `λ` – ( `iter -> 1/iter` ) a function returning the (square summable but not
  summable) sequence of λi
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(5000),`[`StopWhenChangeLess`](@ref)`(10.0^-8))`) a [`StoppingCriterion`](@ref).

and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
function cyclic_proximal_point(
    M::AbstractManifold, f::TF, proxes_f::Union{Tuple,AbstractVector}, p; kwargs...
) where {TF}
    q = copy(M, p)
    return cyclic_proximal_point!(M, f, proxes_f, q; kwargs...)
end

@doc raw"""
    cyclic_proximal_point!(M, F, proxes, x)

perform a cyclic proximal point algorithm in place of `p`.

# Input

* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `proxes` – an Array of proximal maps (`Function`s) `(M, λ, p) -> q` or `(M, q, λ, p)` for the summands of ``F``
* `p` – an initial value ``p ∈ \mathcal M``

for all options, see [`cyclic_proximal_point`](@ref).
"""
function cyclic_proximal_point!(
    M::AbstractManifold,
    f::TF,
    proxes_f::Union{Tuple,AbstractVector},
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    evaluation_order::Symbol=:Linear,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(5000), StopWhenChangeLess(10.0^-12)
    ),
    λ=i -> 1 / i,
    kwargs..., #decorator options
) where {TF}
    mpo = ManifoldProximalMapObjective(f, proxes_f; evaluation=evaluation)
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    cpps = CyclicProximalPointState(
        M, p; stopping_criterion=stopping_criterion, λ=λ, evaluation_order=evaluation_order
    )
    cpps = decorate_state!(cpps; kwargs...)
    return get_solver_return(solve!(dmp, cpps))
end
function initialize_solver!(amp::AbstractManoptProblem, cpps::CyclicProximalPointState)
    c = length(get_objective(amp).proximal_maps!!)
    cpps.order = collect(1:c)
    (cpps.order_type == :FixedRandom) && shuffle!(cpps.order)
    return cpps
end
function step_solver!(amp::AbstractManoptProblem, cpps::CyclicProximalPointState, i)
    λi = cpps.λ(i)
    for k in cpps.order
        get_proximal_map!(amp, cpps.p, λi, cpps.p, k)
    end
    (cpps.order_type == :Random) && shuffle(cpps.order)
    return cpps
end
