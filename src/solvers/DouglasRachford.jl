export DouglasRachford
@doc raw"""
     DouglasRachford(M, f, proxes_f, p)
Computes the Douglas-Rachford algorithm on the manifold ``\mathcal M``, initial
data ``p`` and the (two) proximal maps `proxMaps`.

For ``k>2`` proximal
maps the problem is reformulated using the parallelDouglasRachford: a vectorial
proximal map on the power manifold ``\mathcal M^k`` and the proximal map of the
set that identifies all entries again, i.e. the Karcher mean. This henve also
boild down to two proximal maps, though each evauates proximal maps in parallel,
i.e. component wise in a vector.

# Input
* `M` – a Riemannian Manifold ``\mathcal M``
* `F` – a cost function consisting of a sum of cost functions
* `proxes` – functions of the form `(λ,x)->...` performing a proximal map,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `F`.
* `x0` – initial data ``x_0 ∈ \mathcal M``

# Optional values
the default parameter is given in brackets
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the proximal maps work by allocation (default) form `prox(M, λ, x)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `prox!(M, y, λ, x)`.
* `λ` – (`(iter) -> 1.0`) function to provide the value for the proximal parameter
  during the calls
* `α` – (`(iter) -> 0.9`) relaxation of the step from old to new iterate, i.e.
  ``t_{k+1} = g(α_k; t_k, s_k)``, where ``s_k`` is the result
  of the double reflection involved in the DR algorithm
* `R` – ([`reflect`](@ref)) method employed in the iteration
  to perform the reflection of `x` at the prox `p`.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200),`[`StopWhenChangeLess`](@ref)`(10.0^-5))`) a [`StoppingCriterion`](@ref).
* `parallel` – (`false`) clarify that we are doing a parallel DR, i.e. on a
  `PowerManifold` manifold with two proxes. This can be used to trigger
  parallel Douglas–Rachford if you enter with two proxes. Keep in mind, that a
  parallel Douglas–Rachford implicitly works on a `PowerManifold` manifold and
  its first argument is the result then (assuming all are equal after the second
  prox.

and the ones that are passed to [`decorate_state`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function DouglasRachford(
    M::AbstractManifold, f::TF, proxes_f::Vector{<:Any}, p; kwargs...
) where {TF}
    q = copy(M, p)
    return DouglasRachford!(M, f, proxes_f, q; kwargs...)
end
@doc raw"""
     DouglasRachford(M, F, proxMaps, x)
Computes the Douglas-Rachford algorithm on the manifold ``\mathcal M``, initial
data ``x_0`` and the (two) proximal maps `proxMaps` in place of `x`.

For ``k>2`` proximal
maps the problem is reformulated using the parallelDouglasRachford: a vectorial
proximal map on the power manifold ``\mathcal M^k`` and the proximal map of the
set that identifies all entries again, i.e. the Karcher mean. This hence also
boils down to two proximal maps, though each evaluates proximal maps in parallel,
i.e. component wise in a vector.

# Input
* `M` – a Riemannian Manifold ``\mathcal M``
* `f` – a cost function consisting of a sum of cost functions
* `proxes_f` – functions of the form `(M, λ, p)->q` or `(M, q, λ, p)->q` performing a proximal map,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `f`.
* `p` – initial point ``p ∈ \mathcal M``

For more options, see [`DouglasRachford`](@ref).
"""
function DouglasRachford!(
    M::AbstractManifold,
    f::TF,
    proxes_f::Vector{<:Any},
    p;
    λ::Tλ=(iter) -> 1.0,
    α::Tα=(iter) -> 0.9,
    R::TR=Manopt.reflect,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    parallel::Int=0,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(200), StopWhenChangeLess(10.0^-5)
    ),
    kwargs..., #especially may contain decorator options
) where {TF,Tλ,Tα,TR}
    if length(proxes_f) < 2
        throw(
            ErrorException(
                "Less than two proximal maps provided, the (parallel) Douglas Rachford requires (at least) two proximal maps.",
            ),
        )
    elseif length(proxes_f) == 2
        prox1 = proxes_f[1]
        prox2 = proxes_f[2]
    else # more than 2 -> parallelDouglasRachford
        parallel = length(proxes_f)
        prox1 = (M, λ, x) -> [proxes_f[i](M.manifold, λ, x[i]) for i in 1:parallel]
        prox2 = (M, λ, x) -> fill(mean(M.manifold, x), parallel)
    end
    if parallel > 0
        M = PowerManifold(M, NestedPowerRepresentation(), parallel)
        x = [copy(p) for i in 1:parallel]
        nF = (M, x) -> f(M.manifold, x[1])
    else
        nF = f
    end
    mpo = ManifoldProximalMapObjective(nF, (prox1, prox2); evaluation=evaluation)
    dmp = DefaultManoptProblem(M, mpo)
    drs = DouglasRachfordState(
        M, p; λ=λ, α=α, R=R, stopping_criterion=stopping_criterion, parallel=parallel > 0
    )
    drs = decorate_state(drs; kwargs...)
    return get_solver_return(solve!(dmp, drs))
end
function initialize_solver!(::AbstractManoptProblem, ::DouglasRachfordState) end
function step_solver!(amp::AbstractManoptProblem, drs::DouglasRachfordState, i)
    M = get_manifold(amp)
    get_proximal_map!(amp, drs.p_tmp, drs.λ(i), drs.s, 1)
    drs.s_tmp = drs.R(M, drs.p_tmp, drs.s)
    drs.p = get_proximal_map(amp, drs.λ(i), drs.s_tmp, 2)
    drs.s_tmp = drs.R(M, drs.p, drs.s_tmp)
    # relaxation
    drs.s = shortest_geodesic(M, drs.s, drs.s_tmp, drs.α(i))
    return drs
end
get_solver_result(drs::DouglasRachfordState) = drs.parallel ? drs.p[1] : drs.p
