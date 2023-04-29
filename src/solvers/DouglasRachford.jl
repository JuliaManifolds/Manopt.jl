@doc raw"""
    DouglasRachfordState <: AbstractManoptSolverState

Store all options required for the DouglasRachford algorithm,

# Fields
* `p` - the current iterate (result) For the parallel Douglas-Rachford, this is
  not a value from the `PowerManifold` manifold but the mean.
* `s` – the last result of the double reflection at the proxes relaxed by `α`.
* `λ` – function to provide the value for the proximal parameter during the calls
* `α` – relaxation of the step from old to new iterate, i.e.
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result
  of the double reflection involved in the DR algorithm
* `R` – method employed in the iteration to perform the reflection of `x` at the prox `p`.
* `stop` – a [`StoppingCriterion`](@ref)
* `parallel` – indicate whether we are running a parallel Douglas-Rachford or not.

# Constructor

    DouglasRachfordState(M, p; kwargs...)

Generate the options for a Manifold `M` and an initial point `p`, where the following keyword arguments can be used

* `λ` – (`(iter)->1.0`) function to provide the value for the proximal parameter
  during the calls
* `α` – (`(iter)->0.9`) relaxation of the step from old to new iterate, i.e.
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result
  of the double reflection involved in the DR algorithm
* `R` – ([`reflect`](@ref)) method employed in the iteration to perform the reflection of `x` at
  the prox `p`.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(300)`) a [`StoppingCriterion`](@ref)
* `parallel` – (`false`) indicate whether we are running a parallel Douglas-Rachford
  or not.
"""
mutable struct DouglasRachfordState{P,Tλ,Tα,TR,S} <: AbstractManoptSolverState
    p::P
    p_tmp::P
    s::P
    s_tmp::P
    λ::Tλ
    α::Tα
    R::TR
    stop::S
    parallel::Bool
    function DouglasRachfordState(
        M::AbstractManifold,
        p::P;
        λ::Fλ=i -> 1.0,
        α::Fα=i -> 0.9,
        R::FR=Manopt.reflect,
        stopping_criterion::S=StopAfterIteration(300),
        parallel=false,
    ) where {P,Fλ,Fα,FR,S<:StoppingCriterion}
        return new{P,Fλ,Fα,FR,S}(
            p, copy(M, p), copy(M, p), copy(M, p), λ, α, R, stopping_criterion, parallel
        )
    end
end
function show(io::IO, drs::DouglasRachfordState)
    i = get_count(drs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(drs.stop) ? "Yes" : "No"
    P = drs.parallel ? "Parallel " : ""
    s = """
    # Solver state for `Manopt.jl`s $P Douglas Rachford Algorithm
    $Iter
    ## Stopping Criterion
    $(status_summary(drs.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
get_iterate(drs::DouglasRachfordState) = drs.p
function set_iterate!(drs::DouglasRachfordState, p)
    drs.p = p
    return drs
end

function (d::DebugProximalParameter)(
    ::AbstractManoptProblem, cpps::DouglasRachfordState, i::Int
)
    (i > 0) && Printf.format(d.io, Printf.Format(d.format), cpps.λ(i))
    return nothing
end
function (r::RecordProximalParameter)(
    ::AbstractManoptProblem, cpps::DouglasRachfordState, i::Int
)
    return record_or_reset!(r, cpps.λ(i), i)
end
@doc raw"""
    DouglasRachford(M, f, proxes_f, p)
    DouglasRachford(M, mpo, p)

Compute the Douglas-Rachford algorithm on the manifold ``\mathcal M``, initial
data ``p`` and the (two) proximal maps `proxMaps`.

For ``k>2`` proximal maps, the problem is reformulated using the parallel Douglas Rachford:
A vectorial proximal map on the power manifold ``\mathcal M^k`` is introduced as the first
proximal map and the second proximal map of the is set to the `mean` (Riemannian Center of mass).
This hence also boils down to two proximal maps, though each evaluates proximal maps in parallel,
i.e. component wise in a vector.

If you provide a [`ManifoldProximalMapObjective`](@ref) `mpo` instead, the proximal maps are kept unchanged.

# Input
* `M` – a Riemannian Manifold ``\mathcal M``
* `F` – a cost function consisting of a sum of cost functions
* `proxes_f` – functions of the form `(M, λ, p)->...` performing a proximal maps,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `F`.
  These can also be given in the [`InplaceEvaluation`](@ref) variants `(M, q, λ p) -> ...`
  computing in place of `q`.
* `p` – initial data ``p ∈ \mathcal M``

# Optional values

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

and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
DouglasRachford(::AbstractManifold, args...; kwargs...)
function DouglasRachford(
    M::AbstractManifold,
    f::TF,
    proxes_f::Vector{<:Any},
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    parallel=0,
    kwargs...,
) where {TF}
    N, f_, (prox1, prox2), parallel_, p0 = parallel_to_alternating_DR(
        M, f, proxes_f, p, parallel, evaluation
    )
    mpo = ManifoldProximalMapObjective(f_, (prox1, prox2); evaluation=evaluation)
    return DouglasRachford(N, mpo, p0; evaluation=evaluation, parallel=parallel_, kwargs...)
end
function DouglasRachford(
    M::AbstractManifold,
    f::TF,
    proxes_f::Vector{<:Any},
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF}
    q = [p]
    f_(M, p) = f(M, p[])
    if evaluation isa AllocatingEvaluation
        proxes_f_ = [(M, λ, p) -> [pf(M, λ, p[])] for pf in proxes_f]
    else
        proxes_f_ = [(M, q, λ, p) -> (q .= [pf(M, λ, p[])]) for pf in proxes_f]
    end
    rs = DouglasRachford(M, f_, proxes_f_, q; evaluation=evaluation, kwargs...)
    #return just a number if  the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function DouglasRachford(
    M::AbstractManifold, mpo::ManifoldProximalMapObjective, p; kwargs...
)
    q = copy(M, p)
    return DouglasRachford!(M, mpo, q; kwargs...)
end

@doc raw"""
     DouglasRachford!(M, f, proxes_f, p)
     DouglasRachford!(M, mpo, p)

Compute the Douglas-Rachford algorithm on the manifold ``\mathcal M``, initial
data ``p \in \mathcal M`` and the (two) proximal maps `proxes_f` in place of `p`.

For ``k>2`` proximal maps, the problem is reformulated using the parallel Douglas Rachford:
A vectorial proximal map on the power manifold ``\mathcal M^k`` is introduced as the first
proximal map and the second proximal map of the is set to the `mean` (Riemannian Center of mass).
This hence also boils down to two proximal maps, though each evaluates proximal maps in parallel,
i.e. component wise in a vector.

!!! note

    While creating the new staring point `p'` on the power manifold, a copy of `p`
    Is created, so that the (by k>2 implicitly generated) parallel Douglas Rachford does
    not work in-place for now.

If you provide a [`ManifoldProximalMapObjective`](@ref) `mpo` instead, the proximal maps are kept unchanged.

# Input
* `M` – a Riemannian Manifold ``\mathcal M``
* `f` – a cost function consisting of a sum of cost functions
* `proxes_f` – functions of the form `(M, λ, p)->q` or `(M, q, λ, p)->q` performing a proximal map,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `f`.
* `p` – initial point ``p ∈ \mathcal M``

For more options, see [`DouglasRachford`](@ref).
"""
DouglasRachford!(::AbstractManifold, args...; kwargs...)
function DouglasRachford!(
    M::AbstractManifold,
    f::TF,
    proxes_f::Vector{<:Any},
    p;
    evaluation=AllocatingEvaluation(),
    parallel::Int=0,
    kwargs...,
) where {TF}
    N, f_, (prox1, prox2), parallel_, p0 = parallel_to_alternating_DR(
        M, f, proxes_f, p, parallel, evaluation
    )
    mpo = ManifoldProximalMapObjective(f_, (prox1, prox2); evaluation=evaluation)
    return DouglasRachford!(
        N, mpo, p0; evaluation=evaluation, parallel=parallel_, kwargs...
    )
end
function DouglasRachford!(
    M::AbstractManifold,
    mpo::ManifoldProximalMapObjective,
    p;
    λ::Tλ=(iter) -> 1.0,
    α::Tα=(iter) -> 0.9,
    R::TR=Manopt.reflect,
    parallel::Int=0,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(200), StopWhenChangeLess(10.0^-5)
    ),
    kwargs..., #especially may contain decorator options
) where {Tλ,Tα,TR}
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    drs = DouglasRachfordState(
        M, p; λ=λ, α=α, R=R, stopping_criterion=stopping_criterion, parallel=parallel > 0
    )
    ddrs = decorate_state!(drs; kwargs...)
    return get_solver_return(solve!(dmp, ddrs))
end
#
# An internal function that turns more than 2 proxes into a parallel variant
# on the power manifold
function parallel_to_alternating_DR(
    M, f, proxes_f, p, parallel, evaluation::AbstractEvaluationType
)
    prox1, prox2, parallel_ = prepare_proxes(proxes_f, parallel, evaluation)
    if parallel_ > 0
        N = PowerManifold(M, NestedPowerRepresentation(), parallel_)
        p0 = [p]
        for _ in 2:parallel_
            push!(p0, copy(M, p))
        end
        f_ = (M, p) -> f(M.manifold, p[1])
    else
        N = M
        f_ = f
        p0 = p
    end
    return N, f_, (prox1, prox2), parallel_, p0
end#
# An internal function that turns more than 2 proxes into a parallel variant
function prepare_proxes(proxes_f, parallel, evaluation::AbstractEvaluationType)
    parallel_ = parallel
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
        parallel_ = length(proxes_f)
        if evaluation isa InplaceEvaluation
            prox1 = function (M, q, λ, p)
                [proxes_f[i](M.manifold, q[i], λ, p[i]) for i in 1:parallel_]
                return q
            end
            prox2 = (M, q, λ, p) -> fill!(q, mean(M.manifold, p))
        else
            prox1 = (M, λ, p) -> [proxes_f[i](M.manifold, λ, p[i]) for i in 1:parallel_]
            prox2 = (M, λ, p) -> fill(mean(M.manifold, p), parallel_)
        end
    end
    return prox1, prox2, parallel_
end
function initialize_solver!(::AbstractManoptProblem, ::DouglasRachfordState) end
function step_solver!(amp::AbstractManoptProblem, drs::DouglasRachfordState, i)
    M = get_manifold(amp)
    get_proximal_map!(amp, drs.p_tmp, drs.λ(i), drs.s, 1)
    drs.s_tmp = drs.R(M, drs.p_tmp, drs.s)
    get_proximal_map!(amp, drs.p, drs.λ(i), drs.s_tmp, 2)
    drs.s_tmp = drs.R(M, drs.p, drs.s_tmp)
    # relaxation
    drs.s = shortest_geodesic(M, drs.s, drs.s_tmp, drs.α(i))
    return drs
end
get_solver_result(drs::DouglasRachfordState) = drs.parallel ? drs.p[1] : drs.p
