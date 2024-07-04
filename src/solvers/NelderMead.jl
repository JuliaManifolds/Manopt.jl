
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
struct NelderMeadSimplex{TP,T<:AbstractVector{TP}}
    pts::T
end

function NelderMeadSimplex(M::AbstractManifold)
    return NelderMeadSimplex([rand(M) for i in 1:(manifold_dimension(M) + 1)])
end
function NelderMeadSimplex(M::AbstractManifold, p::Number, B::AbstractBasis; kwargs...)
    return NelderMeadSimplex(M, [p], B; kwargs...)
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

@doc """
    NelderMeadState <: AbstractManoptSolverState

Describes all parameters and the state of a Nelder-Mead heuristic based
optimization algorithm.

# Fields

The naming of these parameters follows the [Wikipedia article](https://en.wikipedia.org/wiki/Nelder–Mead_method)
of the Euclidean case. The default is given in brackets, the required value range
after the description

* `population::`[`NelderMeadSimplex`](@ref): a population (set) of ``d+1`` points ``x_i``, ``i=1,…,n+1``, where ``d``
  is the [`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`) of `M`.
* $_field_step
* `α`: the reflection parameter ``α > 0``:
* `γ` the expansion parameter ``γ > 0``:
* `ρ`: the contraction parameter, ``0 < ρ ≤ \\frac{1}{2}``,
* `σ`: the shrinkage coefficient, ``0 < σ ≤ 1``
* `p`: a field to store the current best value (initialized to _some_ point here)
* $_field_retr
* $_field_inv_retr

# Constructors

    NelderMeadState(M, population::NelderMeadSimplex=NelderMeadSimplex(M)); kwargs...)

Construct a Nelder-Mead Option with a default population (if not provided) of set of
`dimension(M)+1` random points stored in [`NelderMeadSimplex`](@ref).

# Keyword arguments

# Keyword arguments

* `stopping_criterion=`[`StopAfterIteration`](@ref)`(2000)`$_sc_any[`StopWhenPopulationConcentrated`](@ref)`()`):
  a [`StoppingCriterion`](@ref)
* `α=1.0`: reflection parameter ``α > 0``:
* `γ=2.0` expansion parameter ``γ``:
* `ρ=1/2`: contraction parameter, ``0 < ρ ≤ \\frac{1}{2}``,
* `σ=1/2`: shrink coefficient, ``0 < σ ≤ 1``
* $_kw_retraction_method_default: $_kw_retraction_method
* $_kw_inverse_retraction_method_default: $_kw_inverse_retraction_method`inverse_retraction_method=default_inverse_retraction_method(M, typeof(p))`: an inverse retraction to use.
* `p=copy(M, population.pts[1])`: initialise the storage for the best point (iterate)¨
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

    ## Stopping criterion

    $(status_summary(nms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
get_iterate(O::NelderMeadState) = O.p
function set_iterate!(O::NelderMeadState, ::AbstractManifold, p)
    O.p = p
    return O
end

_doc_NelderMead = """
    NelderMead(M::AbstractManifold, f, population=NelderMeadSimplex(M))
    NelderMead(M::AbstractManifold, mco::AbstractManifoldCostObjective, population=NelderMeadSimplex(M))
    NelderMead!(M::AbstractManifold, f, population)
    NelderMead!(M::AbstractManifold, mco::AbstractManifoldCostObjective, population)

Solve a Nelder-Mead minimization problem for the cost function ``f:  $_l_M`` on the
manifold `M`. If the initial [`NelderMeadSimplex`](@ref) is not provided, a random set of
points is chosen. The compuation can be performed in-place of the `population`.

This algorithm is adapted from the Euclidean Nelder-Mead method, see
[https://en.wikipedia.org/wiki/Nelder-Mead_method](https://en.wikipedia.org/wiki/Nelder-Mead_method)
and
[http://www.optimization-online.org/DB_FILE/2007/08/1742.pdf](http://www.optimization-online.org/DB_FILE/2007/08/1742.pdf).

# Input

$_arg_M
$_arg_f
* `population::`[`NelderMeadSimplex`](@ref)`=`[`NelderMeadSimplex`](@ref)`(M)`: an initial simplex of ``d+1`` points, where ``d``
  is the [`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`) of `M`.

# Keyword arguments

* `stopping_criterion=`[`StopAfterIteration`](@ref)`(2000)`$_sc_any[`StopWhenPopulationConcentrated`](@ref)`()`):
  a [`StoppingCriterion`](@ref)
* `α=1.0`: reflection parameter ``α > 0``:
* `γ=2.0` expansion parameter ``γ``:
* `ρ=1/2`: contraction parameter, ``0 < ρ ≤ \\frac{1}{2}``,
* `σ=1/2`: shrink coefficient, ``0 < σ ≤ 1``
* $_kw_retraction_method_default: $_kw_retraction_method
* $_kw_inverse_retraction_method_default: $_kw_inverse_retraction_method`inverse_retraction_method=default_inverse_retraction_method(M, typeof(p))`: an inverse retraction to use.

$_kw_others

!!! note
    The manifold `M` used here has to either provide a `mean(M, pts)` or you have to
    load `Manifolds.jl` to use its statistics part.

$_doc_sec_output
"""

@doc "$(_doc_NelderMead)"
NelderMead(M::AbstractManifold, args...; kwargs...)
function NelderMead(M::AbstractManifold, f; kwargs...)
    return NelderMead(M, f, NelderMeadSimplex(M); kwargs...)
end
function NelderMead(M::AbstractManifold, f, population::NelderMeadSimplex; kwargs...)
    mco = ManifoldCostObjective(f)
    return NelderMead(M, mco, population; kwargs...)
end
function NelderMead(
    M::AbstractManifold, mco::O, population::NelderMeadSimplex; kwargs...
) where {O<:Union{AbstractManifoldCostObjective,AbstractDecoratedManifoldObjective}}
    res_population = NelderMeadSimplex(copy.(Ref(M), population.pts))
    return NelderMead!(M, mco, res_population; kwargs...)
end
function NelderMead(
    M::AbstractManifold, f::F, population::NelderMeadSimplex{P,V}; kwargs...
) where {P<:Number,V<:AbstractVector{P},F<:Function}
    f_ = (M, p) -> f(M, p[])
    population_ = NelderMeadSimplex([[p] for p in population.pts])
    rs = NelderMead(M, f_, population_; kwargs...)
    return (P == eltype(rs)) ? rs[] : rs
end
@doc "$(_doc_NelderMead)"
NelderMead!(M::AbstractManifold, args...; kwargs...)
function NelderMead!(M::AbstractManifold, f, population::NelderMeadSimplex; kwargs...)
    mco = ManifoldCostObjective(f)
    return NelderMead!(M, mco, population; kwargs...)
end
function NelderMead!(
    M::AbstractManifold,
    mco::O,
    population::NelderMeadSimplex;
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
) where {O<:Union{AbstractManifoldCostObjective,AbstractDecoratedManifoldObjective}}
    dmco = decorate_objective!(M, mco; kwargs...)
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

    ind = sortperm(s.costs) # reordering for `s.cost` and `s.p`, that is the minimizer is at `ind[1]`
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
        # successful? use the expanded, otherwise still use `xr`
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
* the maximal distance of the first to the remaining the population points

drops below a certain tolerance `tol_f` and `tol_p`, respectively.

# Constructor

    StopWhenPopulationConcentrated(tol_f::Real=1e-8, tol_x::Real=1e-8)

"""
mutable struct StopWhenPopulationConcentrated{F<:Real} <: StoppingCriterion
    tol_f::F
    tol_p::F
    value_f::F
    value_p::F
    at_iteration::Int
    function StopWhenPopulationConcentrated(tol_f::F=1e-8, tol_p::F=1e-8) where {F<:Real}
        return new{F}(tol_f, tol_p, zero(tol_f), zero(tol_p), -1)
    end
end
function (c::StopWhenPopulationConcentrated)(
    mp::AbstractManoptProblem, s::NelderMeadState, i::Int
)
    if i == 0 # reset on init
        c.at_iteration = -1
    end
    M = get_manifold(mp)
    c.value_f = maximum(cs -> abs(s.costs[1] - cs), s.costs[2:end])
    c.value_p = maximum(
        p -> distance(M, s.population.pts[1], p, s.inverse_retraction_method),
        s.population.pts[2:end],
    )
    if c.value_f < c.tol_f && c.value_p < c.tol_p
        c.at_iteration = i
        return true
    end
    return false
end
function get_reason(c::StopWhenPopulationConcentrated)
    if (c.at_iteration >= 0)
        return "After $(c.at_iteration) iterations the simplex has shrunk below the assumed level. Maximum cost difference is $(c.value_f) < $(c.tol_f), maximum point distance is $(c.value_p) < $(c.tol_p).\n"
    end
    return ""
end
function status_summary(c::StopWhenPopulationConcentrated)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "Population concentration: in f < $(c.tol_f) and in p < $(c.tol_p):\t$s"
end
function show(io::IO, c::StopWhenPopulationConcentrated)
    return print(
        io,
        "StopWhenPopulationConcentrated($(c.tol_f), $(c.tol_p))\n    $(status_summary(c))",
    )
end
