
"""
    NelderMeadSimplex

A simplex for the Nelder-Mead algorithm.

# Constructors

    NelderMeadSimplex(M::AbstractManifold)

Construct a  simplex using ``n+1`` random points from manifold `M`, where ``n`` is the manifold dimension of `M`.

    NelderMeadSimplex(
        M::AbstractManifold,
        p,
        B::AbstractBasis=DefaultOrthonormalBasis();
        a::Real=0.025,
        retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    )

Construct a simplex from a basis `B` with one point being `p` and other points
constructed by moving by `a` in each principal direction defined by basis `B` of the tangent
space at point `p` using retraction `retraction_method`. This works similarly to how
the initial simplex is constructed in the Euclidean Nelder-Mead algorithm, just in
the tangent space at point `p`.
"""
struct NelderMeadSimplex{TP}
    pts::Vector{TP}
end

function NelderMeadSimplex(M::AbstractManifold)
    return NelderMeadSimplex([rand(M) for i in 1:(manifold_dimension(M) + 1)])
end
function NelderMeadSimplex(
    M::AbstractManifold,
    p,
    B::AbstractBasis=DefaultOrthonormalBasis();
    a::Real=0.025,
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
)
    M_dim = manifold_dimension(M)
    vecs = [
        get_vector(M, p, [ifelse(i == j, a, zero(a)) for i in 1:M_dim], B) for j in 0:M_dim
    ]
    pts = map(X -> retract(M, p, X, retraction_method), vecs)
    return NelderMeadSimplex(pts)
end

@doc raw"""
    NelderMeadState <: AbstractManoptSolverState

Describes all parameters and the state of a Nealer-Mead heuristic based
optimization algorithm.

# Fields

The naming of these parameters follows the [Wikipedia article](https://en.wikipedia.org/wiki/Nelder–Mead_method)
of the Euclidean case. The default is given in brackets, the required value range
after the description

* `population` – an `Array{`point`,1}` of ``n+1`` points ``x_i``, ``i=1,…,n+1``, where ``n`` is the
  dimension of the manifold.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(2000) | `[`StopWhenPopulationConcentrated`](@ref)`()`) a [`StoppingCriterion`](@ref)
* `α` – (`1.`) reflection parameter (``α > 0``)
* `γ` – (`2.`) expansion parameter (``γ > 0``)
* `ρ` – (`1/2`) contraction parameter, ``0 < ρ ≤ \frac{1}{2}``,
* `σ` – (`1/2`) shrink coefficient, ``0 < σ ≤ 1``
* `p` – (`copy(population.pts[1])`) - a field to collect the current best value (initialized to _some_ point here)
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) the rectraction to use.
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction to use.

# Constructors

    NelderMead(M[, population::NelderMeadSimplex]; kwargs...)

Construct a Nelder-Mead Option with a default popultion (if not provided) of set of
`dimension(M)+1` random points stored in [`NelderMeadSimplex`](@ref).

In the constructor all fields (besides the population) are keyword arguments.
"""
mutable struct NelderMeadState{
    T,
    S<:StoppingCriterion,
    Tα<:Real,
    Tγ<:Real,
    Tρ<:Real,
    Tσ<:Real,
    TR<:AbstractRetractionMethod,
    TI<:AbstractInverseRetractionMethod,
} <: AbstractManoptSolverState
    population::NelderMeadSimplex{T}
    stop::S
    α::Tα
    γ::Tγ
    ρ::Tρ
    σ::Tσ
    p::T
    costs::Vector{Float64}
    retraction_method::TR
    inverse_retraction_method::TI
    function NelderMeadState(
        M::AbstractManifold,
        population::NelderMeadSimplex{T}=NelderMeadSimplex(M);
        stopping_criterion::StoppingCriterion=StopAfterIteration(2000) |
                                              StopWhenPopulationConcentrated(),
        α=1.0,
        γ=2.0,
        ρ=1 / 2,
        σ=1 / 2,
        retraction_method::AbstractRetractionMethod=default_retraction_method(
            M, eltype(population.pts)
        ),
        inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
            M, eltype(population.pts)
        ),
        p::T=copy(M, population.pts[1]),
    ) where {T}
        return new{
            T,
            typeof(stopping_criterion),
            typeof(α),
            typeof(γ),
            typeof(ρ),
            typeof(σ),
            typeof(retraction_method),
            typeof(inverse_retraction_method),
        }(
            population,
            stopping_criterion,
            α,
            γ,
            ρ,
            σ,
            p,
            [],
            retraction_method,
            inverse_retraction_method,
        )
    end
end
function show(io::IO, nms::NelderMeadState)
    i = get_count(nms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(nms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Nelder Mead Algorithm
    $Iter
    ## Parameters
    * α: $(nms.α)
    * γ: $(nms.γ)
    * ρ: $(nms.ρ)
    * σ: $(nms.σ)
    * inverse retraction method: $(nms.inverse_retraction_method)
    * retraction method:         $(nms.retraction_method)

    ## Stopping Criterion
    $(status_summary(nms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
get_iterate(O::NelderMeadState) = O.p
function set_iterate!(O::NelderMeadState, ::AbstractManifold, p)
    O.p = p
    return O
end

@doc raw"""
    NelderMead(M::AbstractManifold, f [, population::NelderMeadSimplex])

Solve a Nelder-Mead minimization problem for the cost function ``f\colon \mathcal M`` on the
manifold `M`. If the initial population `p` is not given, a random set of
points is chosen.

This algorithm is adapted from the Euclidean Nelder-Mead method, see
[https://en.wikipedia.org/wiki/Nelder–Mead_method](https://en.wikipedia.org/wiki/Nelder–Mead_method)
and
[http://www.optimization-online.org/DB_FILE/2007/08/1742.pdf](http://www.optimization-online.org/DB_FILE/2007/08/1742.pdf).

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function to minimize
* `population` – (n+1 `rand(M)`s) an initial population of ``n+1`` points, where ``n``
  is the dimension of the manifold `M`.

# Optional

* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(2000) | `[`StopWhenPopulationConcentrated`](@ref)`()`) a [`StoppingCriterion`](@ref)
* `α` – (`1.`) reflection parameter (``α > 0``)
* `γ` – (`2.`) expansion parameter (``γ``)
* `ρ` – (`1/2`) contraction parameter, ``0 < ρ ≤ \frac{1}{2}``,
* `σ` – (`1/2`) shrink coefficient, ``0 < σ ≤ 1``
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) the rectraction to use
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction to use.

and the ones that are passed to [`decorate_state!`](@ref) for decorators.

!!! note
    The manifold `M` used here has to either provide a `mean(M, pts)` or you have to
    load `Manifolds.jl` to use its statistics part.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function NelderMead(
    M::AbstractManifold,
    f::TF,
    population::NelderMeadSimplex=NelderMeadSimplex(M);
    kwargs...,
) where {TF}
    res_population = NelderMeadSimplex(copy.(Ref(M), population.pts))
    return NelderMead!(M, f, res_population; kwargs...)
end
@doc raw"""
    NelderMead(M::AbstractManifold, f [, population::NelderMeadSimplex])

Solve a Nelder Mead minimization problem for the cost function `f` on the
manifold `M`. If the initial population `population` is not given, a random set of
points is chosen. If it is given, the computation is done in place of `population`.

For more options see [`NelderMead`](@ref).
"""
function NelderMead!(
    M::AbstractManifold,
    f::TF,
    population::NelderMeadSimplex=NelderMeadSimplex(M);
    stopping_criterion::StoppingCriterion=StopAfterIteration(2000) |
                                          StopWhenPopulationConcentrated(),
    α=1.0,
    γ=2.0,
    ρ=1 / 2,
    σ=1 / 2,
    retraction_method::AbstractRetractionMethod=default_retraction_method(
        M, eltype(population.pts)
    ),
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M, eltype(population.pts)
    ),
    kwargs..., #collect rest
) where {TF}
    dmco = decorate_objective!(M, ManifoldCostObjective(f); kwargs...)
    mp = DefaultManoptProblem(M, dmco)
    s = NelderMeadState(
        M,
        population;
        stopping_criterion=stopping_criterion,
        α=α,
        γ=γ,
        ρ=ρ,
        σ=σ,
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
    )
    s = decorate_state!(s; kwargs...)
    solve!(mp, s)
    return get_solver_return(s)
end
#
# Solver functions
#
function initialize_solver!(mp::AbstractManoptProblem, s::NelderMeadState)
    # init cost and p
    s.costs = get_cost.(Ref(mp), s.population.pts)
    return s.p = s.population.pts[argmin(s.costs)] # select min
end
function step_solver!(mp::AbstractManoptProblem, s::NelderMeadState, ::Any)
    M = get_manifold(mp)

    ind = sortperm(s.costs) # reordering for cost and p, i.e. minimizer is at ind[1]
    permute!(s.costs, ind)
    permute!(s.population.pts, ind)
    m = mean(M, s.population.pts[1:(end - 1)])
    ξ = inverse_retract(M, m, s.population.pts[end], s.inverse_retraction_method)

    # reflect last
    xr = retract(M, m, -s.α * ξ, s.retraction_method)
    Costr = get_cost(mp, xr)
    continue_steps = true
    # is it better than the worst but not better than the best?
    if Costr >= s.costs[1] && Costr < s.costs[end]
        # store as last
        s.population.pts[end] = xr
        s.costs[end] = Costr
        continue_steps = false
    end
    # --- Expansion ---
    if Costr < s.costs[1] # reflected is better than fist -> expand
        xe = retract(M, m, -s.γ * s.α * ξ, s.retraction_method)
        Coste = get_cost(mp, xe)
        # successful? use the expanded, otherwise still use xr
        s.population.pts[end] = Coste < Costr ? xe : xr
        s.costs[end] = min(Coste, Costr)
        continue_steps = false
    end
    # --- Contraction ---
    if continue_steps && Costr > s.costs[end - 1] # even worse than second worst
        step = (Costr < s.costs[end] ? -s.ρ : s.ρ)
        xc = retract(M, m, step * ξ, s.retraction_method)
        Costc = get_cost(mp, xc)
        if Costc < s.costs[end] # better than last ? -> store
            s.population.pts[end] = xc
            s.costs[end] = Costc
            continue_steps = false
        end
    end
    # --- Shrink ---
    if continue_steps
        for i in 2:length(ind)
            retract!(
                M,
                s.population.pts[i],
                s.population.pts[1],
                inverse_retract(
                    M, s.population.pts[1], s.population.pts[i], s.inverse_retraction_method
                ),
                s.σ,
                s.retraction_method,
            )
            # update cost
            s.costs[i] = get_cost(mp, s.population.pts[i])
        end
    end
    # store best
    s.p = s.population.pts[argmin(s.costs)]
    return s
end

"""
    StopWhenPopulationConcentrated <: StoppingCriterion

A stopping criterion for [`NelderMead`](@ref) to indicate to stop when
both

* the maximal distance of the first to the remaining the cost values and
* the maximal diistance of the first to the remaining the population points

drops below a ceertain tolerance `tol_f` and `tol_p`, respectively.

# Constructor

    StopWhenPopulationConcentrated(tol_f::Real=1e-8, tol_x::Real=1e-8)

"""
mutable struct StopWhenPopulationConcentrated{TF<:Real,TP<:Real} <: StoppingCriterion
    tol_f::TF
    tol_p::TP
    reason::String
    at_iteration::Int
    function StopWhenPopulationConcentrated(tol_f::Real=1e-8, tol_p::Real=1e-8)
        return new{typeof(tol_f),typeof(tol_p)}(tol_f, tol_p, "", 0)
    end
end
function (c::StopWhenPopulationConcentrated)(
    mp::AbstractManoptProblem, s::NelderMeadState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    M = get_manifold(mp)
    max_cdiff = maximum(cs -> abs(s.costs[1] - cs), s.costs[2:end])
    max_xdiff = maximum(
        p -> distance(M, s.population.pts[1], p, s.inverse_retraction_method),
        s.population.pts[2:end],
    )
    if max_cdiff < c.tol_f && max_xdiff < c.tol_p
        c.reason = "After $i iterations the simplex has shrunk below the assumed level (maximum cost difference is $max_cdiff, maximum point distance is $max_xdiff).\n"
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::StopWhenPopulationConcentrated)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Population concentration: in f < $(c.tol_f) and in p < $(c.tol_p):\t$s"
end
function show(io::IO, c::StopWhenPopulationConcentrated)
    return print(
        io,
        "StopWhenPopulationCincentrate($(c.f_tol), $(c.p_tol))\n    $(status_summary(c))",
    )
end
