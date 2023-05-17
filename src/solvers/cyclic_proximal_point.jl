
function show(io::IO, cpps::CyclicProximalPointState)
    i = get_count(cpps, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(cpps.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Cyclic Proximal Point Algorithm
    $Iter
    ## Parameters
    * evaluation order of the proximal maps: :$(cpps.order_type)

    ## Stopping Criterion
    $(status_summary(cpps.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
@doc raw"""
    cyclic_proximal_point(M, f, proxes_f, p)
    cyclic_proximal_point(M, mpo, p)

perform a cyclic proximal point algorithm.

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `proxes_f` – an Array of proximal maps (`Function`s) `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the summands of ``f`` (see `evaluation`)
* `p` – an initial value ``p ∈ \mathcal M``

where `f` and the proximal maps `proxes_f` can also be given directly as a [`ManifoldProximalMapObjective`](@ref) `mpo`

# Optional

* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the proximal maps work by allocation (default) form `prox(M, λ, x)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `prox!(M, y, λ, x)`.
* `evaluation_order` – (`:Linear`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Random`) or the default linear one.
* `λ` – ( `iter -> 1/iter` ) a function returning the (square summable but not
  summable) sequence of λi
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(5000),`[`StopWhenChangeLess`](@ref)`(10.0^-8))`) a [`StoppingCriterion`](@ref).

All other keyword arguments are passed to [`decorate_state!`](@ref) for decorators or
[`decorate_objective!`](@ref), respectively.
If you provide the [`ManifoldProximalMapObjective`](@ref) directly, these decorations can still be specified.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
cyclic_proximal_point(M::AbstractManifold, args...; kwargs...)
function cyclic_proximal_point(
    M::AbstractManifold,
    f,
    proxes_f::Union{Tuple,AbstractVector},
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    mpo = ManifoldProximalMapObjective(f, proxes_f; evaluation=evaluation)
    return cyclic_proximal_point(M, mpo, p; evaluation=evaluation, kwargs...)
end
function cyclic_proximal_point(
    M::AbstractManifold,
    f,
    proxes_f::Union{Tuple,AbstractVector},
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    # redefine our initial point and encapsulate functons
    q = [p]
    f_(M, p) = f(M, p[])
    if evaluation isa AllocatingEvaluation
        proxes_f_ = [(M, λ, p) -> [pf(M, λ, p[])] for pf in proxes_f]
    else
        proxes_f_ = [(M, q, λ, p) -> (q .= [pf(M, λ, p[])]) for pf in proxes_f]
    end
    rs = cyclic_proximal_point(M, f_, proxes_f_, q; evaluation=evaluation, kwargs...)
    #return just a number if  the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function cyclic_proximal_point(
    M::AbstractManifold, mpo::ManifoldProximalMapObjective, p; kwargs...
)
    q = copy(M, p)
    return cyclic_proximal_point!(M, mpo, q; kwargs...)
end

@doc raw"""
    cyclic_proximal_point!(M, F, proxes, p)
    cyclic_proximal_point!(M, mpo, p)

perform a cyclic proximal point algorithm in place of `p`.

# Input

* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `proxes` – an Array of proximal maps (`Function`s) `(M, λ, p) -> q` or `(M, q, λ, p)` for the summands of ``F``
* `p` – an initial value ``p ∈ \mathcal M``

where `f` and the proximal maps `proxes_f` can also be given directly as a [`ManifoldProximalMapObjective`](@ref) `mpo`

for all options, see [`cyclic_proximal_point`](@ref).
"""
cyclic_proximal_point!(M::AbstractManifold, args...; kwargs...)
function cyclic_proximal_point!(
    M::AbstractManifold,
    f,
    proxes_f::Union{Tuple,AbstractVector},
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    mpo = ManifoldProximalMapObjective(f, proxes_f; evaluation=evaluation)
    return cyclic_proximal_point(M, mpo, p; evaluation=evaluation, kwargs...)
end
function cyclic_proximal_point!(
    M::AbstractManifold,
    mpo::ManifoldProximalMapObjective,
    p;
    evaluation_order::Symbol=:Linear,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(5000), StopWhenChangeLess(10.0^-12)
    ),
    λ=i -> 1 / i,
    kwargs...,
)
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    cpps = CyclicProximalPointState(
        M, p; stopping_criterion=stopping_criterion, λ=λ, evaluation_order=evaluation_order
    )
    dcpps = decorate_state!(cpps; kwargs...)
    solve!(dmp, dcpps)
    return get_solver_return(get_objective(dmp), dcpps)
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
