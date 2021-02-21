export DouglasRachford
@doc raw"""
     DouglasRachford(M, F, proxMaps, x)
Computes the Douglas-Rachford algorithm on the manifold ``\mathcal M``, initial
data ``x_0`` and the (two) proximal maps `proxMaps`.

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
* `return_options` – (`false`) – if actiavated, the extended result, i.e. the
    complete [`Options`](@ref) re returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `x_opt` if returned
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `x_opt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function DouglasRachford(M::Manifold, F::TF, proxes::Vector{<:Any}, x; kwargs...) where {TF}
    x_res = allocate(x)
    copyto!(x_res, x)
    return DouglasRachford!(M, F, proxes, x; kwargs...)
end
@doc raw"""
     DouglasRachford(M, F, proxMaps, x)
Computes the Douglas-Rachford algorithm on the manifold ``\mathcal M``, initial
data ``x_0`` and the (two) proximal maps `proxMaps` in place of `x`.

For ``k>2`` proximal
maps the problem is reformulated using the parallelDouglasRachford: a vectorial
proximal map on the power manifold ``\mathcal M^k`` and the proximal map of the
set that identifies all entries again, i.e. the Karcher mean. This hence also
boils down to two proximal maps, though each evauates proximal maps in parallel,
i.e. component wise in a vector.

# Input
* `M` – a Riemannian Manifold ``\mathcal M``
* `F` – a cost function consisting of a sum of cost functions
* `proxes` – functions of the form `(λ,x)->...` performing a proximal map,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `F`.
* `x0` – initial data ``x_0 ∈ \mathcal M``

For more options, see [`DouglasRachford`](@ref).
"""
function DouglasRachford!(
    M::Manifold,
    F::TF,
    proxes::Vector{<:Any},
    x;
    λ::Tλ=(iter) -> 1.0,
    α::Tα=(iter) -> 0.9,
    R::TR=reflect,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    parallel::Int=0,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(200), StopWhenChangeLess(10.0^-5)
    ),
    return_options=false,
    kwargs..., #especially may contain decorator options
) where {TF,Tλ,Tα,TR}
    if length(proxes) < 2
        throw(
            ErrorException(
                "Less than two proximal maps provided, the (parallel) Douglas Rachford requires (at least) two proximal maps.",
            ),
        )
    elseif length(proxes) == 2
        prox1 = proxes[1]
        prox2 = proxes[2]
        parallel = 0
    else # more than 2 -> parallelDouglasRachford
        parallel = length(proxes)
        prox1 = (M, λ, x) -> [proxes[i](M, λ, x[i]) for i in 1:parallel]
        prox2 = (M, λ, x) -> fill(mean(M.manifold, x), parallel)
    end
    if parallel > 0
        M = PowerManifold(M, NestedPowerRepresentation(), parallel)
        x = [copy(x) for i in 1:parallel]
        nF = (M, x) -> F(M.manifold, x[1])
    else
        nF = F
    end
    p = ProximalProblem(M, nF, (prox1, prox2); evaluation=evaluation)
    o = DouglasRachfordOptions(x, λ, α, reflect, stopping_criterion, parallel > 0)

    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end
function initialize_solver!(::ProximalProblem, ::DouglasRachfordOptions) end
function step_solver!(p::ProximalProblem, o::DouglasRachfordOptions, iter)
    get_proximal_map!(p, o.xtmp, o.λ(iter), o.s, 1)
    o.stmp = o.R(p.M, o.xtmp, o.s)
    o.x = get_proximal_map(p, o.λ(iter), o.stmp, 2)
    o.stmp = o.R(p.M, o.x, o.stmp)
    # relaxation
    o.s = shortest_geodesic(p.M, o.s, o.stmp, o.α(iter))
    return o
end
get_solver_result(o::DouglasRachfordOptions) = o.parallel ? o.x[1] : o.x
