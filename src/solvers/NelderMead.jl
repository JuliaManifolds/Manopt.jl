@doc raw"""
    NelderMead(M, F [, p])
perform a nelder mead minimization problem for the cost funciton `F` on the
manifold `M`. If the initial population `p` is not given, a random set of
points is chosen.

This algorithm is adapted from the Euclidean Nelder-Mead method, see
[https://en.wikipedia.org/wiki/Nelder–Mead_method](https://en.wikipedia.org/wiki/Nelder–Mead_method)
and
[http://www.optimization-online.org/DB_FILE/2007/08/1742.pdf](http://www.optimization-online.org/DB_FILE/2007/08/1742.pdf).

# Input

* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `population` – (n+1 `random_point(M)`) an initial population of ``n+1`` points, where ``n``
  is the dimension of the manifold `M`.

# Optional

* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(2000)`) a [`StoppingCriterion`](@ref)
* `α` – (`1.`) reflection parameter (``α > 0``)
* `γ` – (`2.`) expansion parameter (``γ``)
* `ρ` – (`1/2`) contraction parameter, ``0 < ρ ≤ \frac{1}{2}``,
* `σ` – (`1/2`) shrink coefficient, ``0 < σ ≤ 1``
* `retraction_method` – (`ExponentialRetraction`) the rectraction to use
* `inverse_retraction_method` - (`LogarithmicInverseRetraction`) an inverse retraction to use.

and the ones that are passed to [`decorate_options`](@ref) for decorators.

!!! note The manifold `M` used here has to either provide a `mean(M, pts)` or you have to
    load `Manifolds.jl` to use its statistics part.

# Output
* either `x` the last iterate or the complete options depending on the optional
  keyword `return_options`, which is false by default (hence then only `x` is
  returned).
"""
function NelderMead(
    M::Manifold,
    F::TF,
    population=[random_point(M) for i in 1:(manifold_dimension(M) + 1)];
    kwargs...,
) where {TF}
    res_population = deepcopy(population)
    return NelderMead!(M, F, res_population; kwargs...)
end
@doc raw"""
    NelderMead(M, F [, p])
perform a nelder mead minimization problem for the cost funciton `F` on the
manifold `M`. If the initial population `p` is not given, a random set of
points is chosen. If it is given, the computation is done in place of `p`.

For more options see [`NelderMead`](@ref).
"""
function NelderMead!(
    M::Manifold,
    F::TF,
    population=[random_point(M) for i in 1:(manifold_dimension(M) + 1)];
    stopping_criterion::StoppingCriterion=StopAfterIteration(200000),
    α=1.0,
    γ=2.0,
    ρ=1 / 2,
    σ=1 / 2,
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    inverse_retraction_method::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
    return_options=false,
    kwargs..., #collect rest
) where {TF}
    p = CostProblem(M, F)
    o = NelderMeadOptions(
        population,
        stopping_criterion;
        α=α,
        γ=γ,
        ρ=ρ,
        σ=σ,
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end
#
# Solver functions
#
function initialize_solver!(p::CostProblem, o::NelderMeadOptions)
    # init cost and x
    o.costs = get_cost.(Ref(p), o.population)
    return o.x = o.population[argmin(o.costs)] # select min
end
function step_solver!(p::CostProblem, o::NelderMeadOptions, ::Any)
    m = mean(p.M, o.population)
    ind = sortperm(o.costs) # reordering for cost and p, i.e. minimizer is at ind[1]
    ξ = inverse_retract(p.M, m, o.population[last(ind)], o.inverse_retraction_method)
    # reflect last
    xr = retract(p.M, m, -o.α * ξ, o.retraction_method)
    Costr = get_cost(p, xr)
    # is it better than the worst but not better than the best?
    if Costr >= o.costs[first(ind)] && Costr < o.costs[last(ind)]
        # store as last
        o.population[last(ind)] = xr
        o.costs[last(ind)] = Costr
    end
    # --- Expansion ---
    if Costr < o.costs[first(ind)] # reflected is better than fist -> expand
        xe = retract(p.M, m, -o.γ * o.α * ξ, o.retraction_method)
        Coste = get_cost(p, xe)
        # successful? use the expanded, otherwise still use xr
        o.population[last(ind)] .= Coste < Costr ? xe : xr
        o.costs[last(ind)] = min(Coste, Costr)
    end
    # --- Contraction ---
    if Costr > o.costs[ind[end - 1]] # even worse than second worst
        s = (Costr < o.costs[last(ind)] ? -o.ρ : o.ρ)
        xc = retract(p.M, m, s * ξ, o.retraction_method)
        Costc = get_cost(p, xc)
        if Costc < o.costs[last(ind)] # better than last ? -> store
            o.population[last(ind)] = xc
            o.costs[last(ind)] = Costc
        end
    end
    # --- Shrink ---
    for i in 2:length(ind)
        o.population[ind[i]] = retract(
            p.M,
            o.population[ind[1]],
            inverse_retract(
                p.M, o.population[ind[1]], o.population[ind[i]], o.inverse_retraction_method
            ),
            o.σ,
            o.retraction_method,
        )
        # update cost
        o.costs[ind[i]] = get_cost(p, o.population[ind[i]])
    end
    # store best
    return o.x = o.population[argmin(o.costs)]
end
