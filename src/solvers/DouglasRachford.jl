@doc raw"""
    DouglasRachfordState <: AbstractManoptSolverState

Store all options required for the DouglasRachford algorithm,

# Fields
* `p`:                         the current iterate (result) For the parallel Douglas-Rachford,
  this is not a value from the `PowerManifold` manifold but the mean.
* `s`:                         the last result of the double reflection at the proximal maps relaxed by `α`.
* `λ`:                         function to provide the value for the proximal parameter during the calls
* `α`:                         relaxation of the step from old to new iterate, to be precise
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result of the double
  reflection involved in the DR algorithm
* `inverse_retraction_method`: an inverse retraction method
* `R`:                          method employed in the iteration to perform the reflection of `x` at the prox `p`.
* `reflection_evaluation`:     whether `R` works in-place or allocating
* `retraction_method`:         a retraction method
* `stop`:                      a [`StoppingCriterion`](@ref)
* `parallel`:                  indicate whether to use a parallel Douglas-Rachford or not.

# Constructor

    DouglasRachfordState(M, p; kwargs...)

Generate the options for a Manifold `M` and an initial point `p`, where the following keyword arguments can be used

* `λ=(iter)->1.0`: function to provide the value for the proximal parameter
  during the calls
* `α=(iter)->0.9`: relaxation of the step from old to new iterate, to be precise
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result of the double reflection involved in the DR algorithm
* `R`:                    ([`reflect`](@ref) or `reflect!`) method employed in the iteration to perform the reflection of `x` at
  the prox `p`, which function is used depends on `reflection_evaluation`.
* `reflection_evaluation`: ([`AllocatingEvaluation`](@ref)`()`) specify whether the reflection works in-place or allocating (default)
* `stopping_criterion`:   ([`StopAfterIteration`](@ref)`(300)`) a [`StoppingCriterion`](@ref)
* `parallel=false`: indicate whether to use a parallel Douglas-Rachford or not.
"""
mutable struct DouglasRachfordState{
    P,
    Tλ,
    Tα,
    TR,
    S,
    E<:AbstractEvaluationType,
    TM<:AbstractRetractionMethod,
    ITM<:AbstractInverseRetractionMethod,
} <: AbstractManoptSolverState
    p::P
    p_tmp::P
    s::P
    s_tmp::P
    λ::Tλ
    α::Tα
    R::TR
    reflection_evaluation::E
    retraction_method::TM
    inverse_retraction_method::ITM
    stop::S
    parallel::Bool
    function DouglasRachfordState(
        M::AbstractManifold,
        p::P;
        λ::Fλ=i -> 1.0,
        α::Fα=i -> 0.9,
        R::FR=Manopt.reflect,
        reflection_evaluation::E=AllocatingEvaluation(),
        stopping_criterion::S=StopAfterIteration(300),
        parallel=false,
        retraction_method::TM=default_retraction_method(M, typeof(p)),
        inverse_retraction_method::ITM=default_inverse_retraction_method(M, typeof(p)),
    ) where {
        P,
        Fλ,
        Fα,
        FR,
        S<:StoppingCriterion,
        E<:AbstractEvaluationType,
        TM<:AbstractRetractionMethod,
        ITM<:AbstractInverseRetractionMethod,
    }
        return new{P,Fλ,Fα,FR,S,E,TM,ITM}(
            p,
            copy(M, p),
            copy(M, p),
            copy(M, p),
            λ,
            α,
            R,
            reflection_evaluation,
            retraction_method,
            inverse_retraction_method,
            stopping_criterion,
            parallel,
        )
    end
end
function show(io::IO, drs::DouglasRachfordState)
    i = get_count(drs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    refl_e = drs.reflection_evaluation == AllocatingEvaluation() ? "allocating" : "in place"
    Conv = indicates_convergence(drs.stop) ? "Yes" : "No"
    P = drs.parallel ? "Parallel " : ""
    s = """
    # Solver state for `Manopt.jl`s $P Douglas Rachford Algorithm
    $Iter
    using an $(refl_e) reflection.

    ## Stopping criterion

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
data ``p`` and the (two) proximal maps `proxMaps`, see [ BergmannPerschSteidl:2016](@cite).

For ``k>2`` proximal maps, the problem is reformulated using the parallel Douglas Rachford:
a vectorial proximal map on the power manifold ``\mathcal M^k`` is introduced as the first
proximal map and the second proximal map of the is set to the [`mean`](@extref Statistics.mean-Tuple{AbstractManifold, Vararg{Any}}) (Riemannian center of mass).
This hence also boils down to two proximal maps, though each evaluates proximal maps in parallel,
that is, component wise in a vector.

If you provide a [`ManifoldProximalMapObjective`](@ref) `mpo` instead, the proximal maps are kept unchanged.

# Input
* `M`:        a Riemannian Manifold ``\mathcal M``
* `F`:        a cost function consisting of a sum of cost functions
* `proxes_f`: functions of the form `(M, λ, p)-> q` performing a proximal maps,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `F`.
  These can also be given in the [`InplaceEvaluation`](@ref) variants `(M, q, λ p) -> q`
  computing in place of `q`.
* `p`:        initial data ``p ∈ \mathcal M``

# Optional values

* `evaluation`:            ([`AllocatingEvaluation`](@ref)) specify whether the proximal maps work by allocation (default) form `prox(M, λ, x)`
  or [`InplaceEvaluation`](@ref) in-place
* `λ=(iter) -> 1.0`: function to provide the value for the proximal parameter during the calls
* `α=(iter) -> 0.9`: relaxation of the step from old to new iterate, to be precise
  ``t_{k+1} = g(α_k; t_k, s_k)``, where ``s_k`` is the result
  of the double reflection involved in the DR algorithm
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) the inverse retraction to use within
  - the reflection (ignored, if you set `R` directly)
  - the relaxation step
* `R`:                     method employed in the iteration to perform the reflection of `x` at the prox `p`.
  This uses by default [`reflect`](@ref) or `reflect!` depending on `reflection_evaluation` and
  the retraction and inverse retraction specified by `retraction_method` and `inverse_retraction_method`, respectively.
* `reflection_evaluation`: ([`AllocatingEvaluation`](@ref) whether `R` works in-place or allocating
* `retraction_method=default_retraction_method(M, typeof(p))`: the retraction to use in
  - the reflection (ignored, if you set `R` directly)
  - the relaxation step
* `stopping_criterion`:    ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenChangeLess`](@ref)`(1e-5)`)
  a [`StoppingCriterion`](@ref).
* `parallel=false`: indicate whether to use a parallel Douglas-Rachford or not.

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
    M::AbstractManifold, mpo::O, p; kwargs...
) where {O<:Union{ManifoldProximalMapObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return DouglasRachford!(M, mpo, q; kwargs...)
end

@doc raw"""
     DouglasRachford!(M, f, proxes_f, p)
     DouglasRachford!(M, mpo, p)

Compute the Douglas-Rachford algorithm on the manifold ``\mathcal M``, initial
data ``p ∈ \mathcal M`` and the (two) proximal maps `proxes_f` in place of `p`.

For ``k>2`` proximal maps, the problem is reformulated using the parallel Douglas Rachford:
a vectorial proximal map on the power manifold ``\mathcal M^k`` is introduced as the first
proximal map and the second proximal map of the is set to the [`mean`](@extref Statistics.mean-Tuple{AbstractManifold, Vararg{Any}}) (Riemannian center of mass).
This hence also boils down to two proximal maps, though each evaluates proximal maps in parallel,
that is component wise in a vector.

!!! note

    While creating the new staring point `p'` on the power manifold, a copy of `p`
    Is created, so that the (by k>2 implicitly generated) parallel Douglas Rachford does
    not work in-place for now.

If you provide a [`ManifoldProximalMapObjective`](@ref) `mpo` instead, the proximal maps are kept unchanged.

# Input
* `M`:        a Riemannian Manifold ``\mathcal M``
* `f`:        a cost function consisting of a sum of cost functions
* `proxes_f`: functions of the form `(M, λ, p)->q` or `(M, q, λ, p)->q` performing a proximal map,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `f`.
* `p`:        initial point ``p ∈ \mathcal M``

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
    mpo::O,
    p;
    λ::Tλ=(iter) -> 1.0,
    α::Tα=(iter) -> 0.9,
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M, typeof(p)
    ),
    reflection_evaluation::E=AllocatingEvaluation(),
    # Adapt to evaluation type
    R::TR=if reflection_evaluation == InplaceEvaluation()
        (M, r, p, q) -> Manopt.reflect!(
            M,
            r,
            p,
            q;
            retraction_method=retraction_method,
            inverse_retraction_method=inverse_retraction_method,
        )
    else
        (M, p, q) -> Manopt.reflect(
            M,
            p,
            q;
            retraction_method=retraction_method,
            inverse_retraction_method=inverse_retraction_method,
        )
    end,
    parallel::Int=0,
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenChangeLess(1e-5),
    kwargs..., #especially may contain decorator options
) where {
    Tλ,
    Tα,
    TR,
    O<:Union{ManifoldProximalMapObjective,AbstractDecoratedManifoldObjective},
    E<:AbstractEvaluationType,
}
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    drs = DouglasRachfordState(
        M,
        p;
        λ=λ,
        α=α,
        R=R,
        reflection_evaluation=reflection_evaluation,
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
        stopping_criterion=stopping_criterion,
        parallel=parallel > 0,
    )
    ddrs = decorate_state!(drs; kwargs...)
    solve!(dmp, ddrs)
    return get_solver_return(get_objective(dmp), ddrs)
end
#
# An internal function that turns more than 2 proximal maps into a parallel variant
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
# An internal function that turns more than 2 proximal maps into a parallel variant
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
    #dispatch on allocation type for the reflection, see below.
    _reflect!(M, drs.s_tmp, drs.p_tmp, drs.s, drs.R, drs.reflection_evaluation)
    get_proximal_map!(amp, drs.p, drs.λ(i), drs.s_tmp, 2)
    _reflect!(M, drs.s_tmp, drs.p, drs.s_tmp, drs.R, drs.reflection_evaluation)
    # relaxation
    drs.s = retract(
        M,
        drs.s,
        inverse_retract(M, drs.s, drs.s_tmp, drs.inverse_retraction_method),
        drs.α(i),
        drs.retraction_method,
    )
    return drs
end
get_solver_result(drs::DouglasRachfordState) = drs.parallel ? drs.p[1] : drs.p

function _reflect!(M, r, p, x, R, ::AllocatingEvaluation)
    copyto!(M, r, R(M, p, x))
    return r
end
_reflect!(M, r, p, x, R, ::InplaceEvaluation) = R(M, r, p, x)

@doc raw"""
    DouglasRachford(M, f, proxes_f, p; kwargs...)

a doc string with some math ``t_{k+1} = g(α_k; t_k, s_k)``
"""
DouglasRachford(M, f, proxes_f, p; kwargs...)
