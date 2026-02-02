"""
    NelderMeadSimplex

A simplex for the Nelder-Mead algorithm.

# Constructors

    NelderMeadSimplex(M::AbstractManifold)

Construct a  simplex using ``d+1`` random points from manifold `M`,
where ``d`` is the $(_link(:manifold_dimension; M = "")) of `M`.

    NelderMeadSimplex(
        M::AbstractManifold,
        p,
        B::AbstractBasis=default_basis(M, typeof(p));
        a::Real=0.025,
        retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    )

Construct a simplex from a basis `B` with one point being `p` and other points
constructed by moving by `a` in each principal direction defined by basis `B` of the tangent
space at point `p` using retraction `retraction_method`. This works similarly to how
the initial simplex is constructed in the Euclidean Nelder-Mead algorithm, just in
the tangent space at point `p`.
"""
struct NelderMeadSimplex{TP, T <: AbstractVector{TP}}
    pts::T
end

function NelderMeadSimplex(M::AbstractManifold)
    return NelderMeadSimplex([rand(M) for i in 1:(manifold_dimension(M) + 1)])
end
function NelderMeadSimplex(
        M::AbstractManifold,
        p,
        B::AbstractBasis = default_basis(M, typeof(p));
        a::Real = 0.025,
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    )
    p_ = _ensure_mutating_variable(p)
    M_dim = manifold_dimension(M)
    vecs = [
        get_vector(M, p_, [ifelse(i == j, a, zero(a)) for i in 1:M_dim], B) for j in 0:M_dim
    ]
    pts = map(X -> retract(M, p_, X, retraction_method), vecs)
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
  is the $(_link(:manifold_dimension; M = "")) of `M`.
$(_fields(:stepsize))
* `α`: the reflection parameter ``α > 0``:
* `γ` the expansion parameter ``γ > 0``:
* `ρ`: the contraction parameter, ``0 < ρ ≤ \\frac{1}{2}``,
* `σ`: the shrinkage coefficient, ``0 < σ ≤ 1``
$(_fields(:p))
  storing the current best point
$(_fields([:inverse_retraction_method, :retraction_method]))

# Constructors

    NelderMeadState(M::AbstractManifold; kwargs...)

Construct a Nelder-Mead Option with a default population (if not provided) of set of
`dimension(M)+1` random points stored in [`NelderMeadSimplex`](@ref).

# Keyword arguments

* `population=`[`NelderMeadSimplex`](@ref)`(M)`
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(2000)`$(_sc(:Any))[`StopWhenPopulationConcentrated`](@ref)`()"))
  a [`StoppingCriterion`](@ref)
* `α=1.0`: reflection parameter ``α > 0``:
* `γ=2.0` expansion parameter ``γ``:
* `ρ=1/2`: contraction parameter, ``0 < ρ ≤ \\frac{1}{2}``,
* `σ=1/2`: shrink coefficient, ``0 < σ ≤ 1``
$(_kwargs([:inverse_retraction_method, :retraction_method]))
* `p=copy(M, population.pts[1])`: initialise the storage for the best point (iterate)¨
"""
mutable struct NelderMeadState{
        T,
        S <: StoppingCriterion,
        Tα <: Real,
        Tγ <: Real,
        Tρ <: Real,
        Tσ <: Real,
        TR <: AbstractRetractionMethod,
        TI <: AbstractInverseRetractionMethod,
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
            M::AbstractManifold;
            population::NelderMeadSimplex{T} = NelderMeadSimplex(M),
            stopping_criterion::StoppingCriterion = StopAfterIteration(2000) |
                StopWhenPopulationConcentrated(),
            α = 1.0,
            γ = 2.0,
            ρ = 1 / 2,
            σ = 1 / 2,
            retraction_method::AbstractRetractionMethod = default_retraction_method(
                M, eltype(population.pts)
            ),
            inverse_retraction_method::AbstractInverseRetractionMethod = default_inverse_retraction_method(
                M, eltype(population.pts)
            ),
            p::T = copy(M, population.pts[1]),
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

Solve a Nelder-Mead minimization problem for the cost function ``f: $(_tex(:Cal, "M")) → ℝ`` on the
manifold `M`. If the initial [`NelderMeadSimplex`](@ref) is not provided, a random set of
points is chosen. The computation can be performed in-place of the `population`.

The algorithm consists of the following steps. Let ``d`` denote the dimension of the manifold ``$(_tex(:Cal, "M"))``.

1. Order the simplex vertices ``p_i, i=1,…,d+1`` by increasing cost, such that we have ``f(p_1) ≤ f(p_2) ≤ … ≤ f(p_{d+1})``.
2. Compute the Riemannian center of mass [Karcher:1977](@cite), cf. [`mean`](@extref Statistics.mean-Tuple{AbstractManifold, Vararg{Any}}), ``p_{$(_tex(:text, "m"))}``
    of the simplex vertices ``p_1,…,p_{d+1}``.
3. Reflect the point with the worst point at the mean ``p_{$(_tex(:text, "r"))} = $(_tex(:retr))_{p_{$(_tex(:text, "m"))}}\\bigl( - α$(_tex(:invretr))_{p_{$(_tex(:text, "m"))}} (p_{d+1}) \\bigr)``
    If ``f(p_1) ≤ f(p_{$(_tex(:text, "r"))}) ≤ f(p_{d})`` then set ``p_{d+1} = p_{$(_tex(:text, "r"))}`` and go to step 1.
4. Expand the simplex if ``f(p_{$(_tex(:text, "r"))}) < f(p_1)`` by computing the expansion point ``p_{$(_tex(:text, "e"))} = $(_tex(:retr))_{p_{$(_tex(:text, "m"))}}\\bigl( - γα$(_tex(:invretr))_{p_{$(_tex(:text, "m"))}} (p_{d+1}) \\bigr)``,
    which in this formulation allows to reuse the tangent vector from the inverse retraction from before.
    If ``f(p_{$(_tex(:text, "e"))}) < f(p_{$(_tex(:text, "r"))})`` then set ``p_{d+1} = p_{$(_tex(:text, "e"))}`` otherwise set set ``p_{d+1} = p_{$(_tex(:text, "r"))}``. Then go to Step 1.
5. Contract the simplex if ``f(p_{$(_tex(:text, "r"))}) ≥ f(p_d)``.
    1. If ``f(p_{$(_tex(:text, "r"))}) < f(p_{d+1})`` set the step ``s = -ρ``
    2. otherwise set ``s=ρ``.
    Compute the contraction point ``p_{$(_tex(:text, "c"))} = $(_tex(:retr))_{p_{$(_tex(:text, "m"))}}\\bigl(s$(_tex(:invretr))_{p_{$(_tex(:text, "m"))}} p_{d+1} \\bigr)``.
    1. in this case if ``f(p_{$(_tex(:text, "c"))}) < f(p_{$(_tex(:text, "r"))})`` set ``p_{d+1} = p_{$(_tex(:text, "c"))}`` and go to step 1
    2. in this case if ``f(p_{$(_tex(:text, "c"))}) < f(p_{d+1})`` set ``p_{d+1} = p_{$(_tex(:text, "c"))}`` and go to step 1
6. Shrink all points (closer to ``p_1``). For all ``i=2,...,d+1`` set
    ``p_{i} = $(_tex(:retr))_{p_{1}}\\bigl( σ$(_tex(:invretr))_{p_{1}} p_{i} \\bigr).``

For more details, see The Euclidean variant in the Wikipedia
[https://en.wikipedia.org/wiki/Nelder-Mead_method](https://en.wikipedia.org/wiki/Nelder-Mead_method)
or Algorithm 4.1 in [http://www.optimization-online.org/DB_FILE/2007/08/1742.pdf](http://www.optimization-online.org/DB_FILE/2007/08/1742.pdf).

# Input

$(_args([:M, :f]))
* `population::`[`NelderMeadSimplex`](@ref)`=`[`NelderMeadSimplex`](@ref)`(M)`: an initial simplex of ``d+1`` points, where ``d``
  is the $(_link(:manifold_dimension; M = "")) of `M`.

# Keyword arguments

$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(2000)`$(_sc(:Any))[`StopWhenPopulationConcentrated`](@ref)`()"))
  a [`StoppingCriterion`](@ref)
* `α=1.0`: reflection parameter ``α > 0``:
* `γ=2.0` expansion parameter ``γ``:
* `ρ=1/2`: contraction parameter, ``0 < ρ ≤ \\frac{1}{2}``,
* `σ=1/2`: shrink coefficient, ``0 < σ ≤ 1``
$(_kwargs([:inverse_retraction_method, :retraction_method]))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_NelderMead)"
NelderMead(M::AbstractManifold, args...; kwargs...)
function NelderMead(M::AbstractManifold, f; kwargs...)
    return NelderMead(M, f, NelderMeadSimplex(M); kwargs...)
end
function NelderMead(
        M::AbstractManifold, f::F, population::NelderMeadSimplex{P, V}; kwargs...
    ) where {P <: Number, V <: AbstractVector{P}, F <: Function}
    f_ = (M, p) -> f(M, p[])
    population_ = NelderMeadSimplex([[p] for p in population.pts])
    rs = NelderMead(M, f_, population_; kwargs...)
    return (P == eltype(rs)) ? rs[] : rs
end
function NelderMead(M::AbstractManifold, f, population::NelderMeadSimplex; kwargs...)
    mco = ManifoldCostObjective(f)
    return NelderMead(M, mco, population; kwargs...)
end
function NelderMead(
        M::AbstractManifold, mco::O, population::NelderMeadSimplex; kwargs...
    ) where {O <: Union{AbstractManifoldCostObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(NelderMead; kwargs...)
    res_population = NelderMeadSimplex(copy.(Ref(M), population.pts))
    return NelderMead!(M, mco, res_population; kwargs...)
end
calls_with_kwargs(::typeof(NelderMead)) = (NelderMead!,)

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
        stopping_criterion::StoppingCriterion = StopAfterIteration(2000) |
            StopWhenPopulationConcentrated(),
        α = 1.0,
        γ = 2.0,
        ρ = 1 / 2,
        σ = 1 / 2,
        retraction_method::AbstractRetractionMethod = default_retraction_method(
            M, eltype(population.pts)
        ),
        inverse_retraction_method::AbstractInverseRetractionMethod = default_inverse_retraction_method(
            M, eltype(population.pts)
        ),
        kwargs..., #collect rest
    ) where {O <: Union{AbstractManifoldCostObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(NelderMead; kwargs...)
    dmco = decorate_objective!(M, mco; kwargs...)
    mp = DefaultManoptProblem(M, dmco)
    s = NelderMeadState(
        M;
        population = population,
        stopping_criterion = stopping_criterion,
        α = α,
        γ = γ,
        ρ = ρ,
        σ = σ,
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
    )
    s = decorate_state!(s; kwargs...)
    solve!(mp, s)
    return get_solver_return(s)
end
calls_with_kwargs(::typeof(NelderMead!)) = (decorate_objective!, decorate_state!)

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
    if Costr >= s.costs[1] && Costr < s.costs[end - 1]
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
            ManifoldsBase.retract_fused!(
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
mutable struct StopWhenPopulationConcentrated{F <: Real} <: StoppingCriterion
    tol_f::F
    tol_p::F
    value_f::F
    value_p::F
    at_iteration::Int
    function StopWhenPopulationConcentrated(tol_f::F = 1.0e-8, tol_p::F = 1.0e-8) where {F <: Real}
        return new{F}(tol_f, tol_p, zero(tol_f), zero(tol_p), -1)
    end
end
function (c::StopWhenPopulationConcentrated)(
        mp::AbstractManoptProblem, s::NelderMeadState, k::Int
    )
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    M = get_manifold(mp)
    c.value_f = maximum(cs -> abs(s.costs[1] - cs), s.costs[2:end])
    c.value_p = maximum(
        p -> distance(M, s.population.pts[1], p, s.inverse_retraction_method),
        s.population.pts[2:end],
    )
    if c.value_f < c.tol_f && c.value_p < c.tol_p
        c.at_iteration = k
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
function status_summary(c::StopWhenPopulationConcentrated; inline = false)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    head = (!inline ? "Stop when the population of a swarm is concentrated in eher function values (tolerance: $(c.tol_f)) or points (tolerance: $(c.tol_p))\n\t" : "")
    return head * "Population concentration: in f < $(c.tol_f) and in p < $(c.tol_p):\t$s"
end
function show(io::IO, c::StopWhenPopulationConcentrated)
    return print(io, "StopWhenPopulationConcentrated($(c.tol_f), $(c.tol_p))")
end
