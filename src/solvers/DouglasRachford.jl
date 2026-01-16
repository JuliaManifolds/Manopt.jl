@doc """
    DouglasRachfordState <: AbstractManoptSolverState

Store all options required for the DouglasRachford algorithm,

# Fields

* `α`:                         relaxation of the step from old to new iterate, to be precise
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result of the double
  reflection involved in the DR algorithm
$(_fields(:inverse_retraction_method))
* `λ`:                         function to provide the value for the proximal parameter during the calls
* `parallel`:                  indicate whether to use a parallel Douglas-Rachford or not.
* `R`:                          method employed in the iteration to perform the reflection of `x` at the prox `p`.
$(_fields(:p; add_properties = [:as_Iterate]))
  For the parallel Douglas-Rachford, this is not a value from the `PowerManifold` manifold but the mean.
* `reflection_evaluation`:     whether `R` works in-place or allocating
$(_fields(:retraction_method))
* `s`:                         the last result of the double reflection at the proximal maps relaxed by `α`.
$(_fields(:stopping_criterion; name = "stop"))

# Constructor

    DouglasRachfordState(M::AbstractManifold; kwargs...)

# Input

$(_args(:M))

# Keyword arguments

* `α= k -> 0.9`: relaxation of the step from old to new iterate, to be precise
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result of the double reflection involved in the DR algorithm
$(_kwargs(:inverse_retraction_method))
* `λ= k -> 1.0`: function to provide the value for the proximal parameter
  during the calls
$(_kwargs(:p; add_properties = [:as_Initial]))
* `R=`[`reflect`](@ref)`(!)`: method employed in the iteration to perform the reflection of `p` at
  the prox of `p`, which function is used depends on `reflection_evaluation`.
* `reflection_evaluation=`[`AllocatingEvaluation`](@ref)`()`) specify whether the reflection works in-place or allocating (default)
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(300)"))
* `parallel=false`: indicate whether to use a parallel Douglas-Rachford or not.
"""
mutable struct DouglasRachfordState{
        P,
        Tλ,
        Tα,
        TR,
        S,
        E <: AbstractEvaluationType,
        TM <: AbstractRetractionMethod,
        ITM <: AbstractInverseRetractionMethod,
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
            M::AbstractManifold;
            p::P = rand(M),
            λ::Fλ = i -> 1.0,
            α::Fα = i -> 0.9,
            reflection_evaluation::E = AllocatingEvaluation(),
            R::FR = (
                if reflection_evaluation isa AllocatingEvaluation
                    Manopt.reflect
                else
                    Manopt.reflect!
                end
            ),
            stopping_criterion::S = StopAfterIteration(300),
            parallel = false,
            retraction_method::TM = default_retraction_method(M, typeof(p)),
            inverse_retraction_method::ITM = default_inverse_retraction_method(M, typeof(p)),
        ) where {
            P,
            Fλ,
            Fα,
            FR,
            S <: StoppingCriterion,
            E <: AbstractEvaluationType,
            TM <: AbstractRetractionMethod,
            ITM <: AbstractInverseRetractionMethod,
        }
        return new{P, Fλ, Fα, FR, S, E, TM, ITM}(
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
        ::AbstractManoptProblem, cpps::DouglasRachfordState, k::Int
    )
    (k >= (d.at_init ? 0 : 1)) && Printf.format(d.io, Printf.Format(d.format), cpps.λ(k))
    return nothing
end
function (r::RecordProximalParameter)(
        ::AbstractManoptProblem, cpps::DouglasRachfordState, k::Int
    )
    return record_or_reset!(r, cpps.λ(k), k)
end
_doc_Douglas_Rachford = """
    DouglasRachford(M, f, proxes_f, p)
    DouglasRachford(M, mpo, p)
    DouglasRachford!(M, f, proxes_f, p)
    DouglasRachford!(M, mpo, p)

Compute the Douglas-Rachford algorithm on the manifold ``$(_math(:Manifold))``, starting from `p`
given the (two) proximal maps `proxes_f`, see [BergmannPerschSteidl:2016](@cite).

For ``k>2`` proximal maps, the problem is reformulated using the parallel Douglas Rachford:
a vectorial proximal map on the power manifold ``$(_math(:Manifold))^k`` is introduced as the first
proximal map and the second proximal map of the is set to the [`mean`](@extref Statistics.mean-Tuple{AbstractManifold, Vararg{Any}}) (Riemannian center of mass).
This hence also boils down to two proximal maps, though each evaluates proximal maps in parallel,
that is, component wise in a vector.

!!! note
    The parallel Douglas Rachford does not work in-place for now, since
    while creating the new staring point `p'` on the power manifold, a copy of `p`
    Is created

If you provide a [`ManifoldProximalMapObjective`](@ref) `mpo` instead, the proximal maps are kept unchanged.

# Input

$(_args([:M, :f]))
* `proxes_f`: functions of the form `(M, λ, p)-> q` performing a proximal maps,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `F`.
  These can also be given in the [`InplaceEvaluation`](@ref) variants `(M, q, λ p) -> q`
  computing in place of `q`.
$(_args(:p))

# Keyword arguments

* `α= k -> 0.9`: relaxation of the step from old to new iterate, to be precise
  ``p^{(k+1)} = g(α_k; p^{(k)}, q^{(k)})``, where ``q^{(k)}`` is the result of the double reflection
  involved in the DR algorithm and ``g`` is a curve induced by the retraction and its inverse.
$(_kwargs([:evaluation, :inverse_retraction_method]))
  This is used both in the relaxation step as well as in the reflection, unless you set `R` yourself.
* `λ= k -> 1.0`: function to provide the value for the proximal parameter ``λ_k``
* `R=reflect(!)`:           method employed in the iteration to perform the reflection of `p` at the prox of `p`.
  This uses by default [`reflect`](@ref) or `reflect!` depending on `reflection_evaluation` and
  the retraction and inverse retraction specified by `retraction_method` and `inverse_retraction_method`, respectively.
* `reflection_evaluation`: ([`AllocatingEvaluation`](@ref) whether `R` works in-place or allocating
$(_kwargs(:retraction_method))
  This is used both in the relaxation step as well as in the reflection, unless you set `R` yourself.
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-5)"))
* `parallel=false`: indicate whether to use a parallel Douglas-Rachford or not.

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_Douglas_Rachford)"
DouglasRachford(::AbstractManifold, args...; kwargs...)
function DouglasRachford(
        M::AbstractManifold,
        f::TF,
        proxes_f::Vector{<:Any},
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        parallel = 0,
        kwargs...,
    ) where {TF}
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    proxes_f_ = [_ensure_mutating_prox(prox_f, p, evaluation) for prox_f in proxes_f]
    N, f__, (prox1, prox2), parallel_, q = parallel_to_alternating_DR(
        M, f_, proxes_f_, p_, parallel, evaluation
    )
    mpo = ManifoldProximalMapObjective(f__, (prox1, prox2); evaluation = evaluation)
    rs = DouglasRachford(N, mpo, q; evaluation = evaluation, parallel = parallel_, kwargs...)
    return _ensure_matching_output(p, rs)
end
function DouglasRachford(
        M::AbstractManifold, mpo::O, p; kwargs...
    ) where {O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(DouglasRachford; kwargs...)
    q = copy(M, p)
    return DouglasRachford!(M, mpo, q; kwargs...)
end
calls_with_kwargs(::typeof(DouglasRachford)) = (DouglasRachford!,)

@doc "$(_doc_Douglas_Rachford)"
DouglasRachford!(::AbstractManifold, args...; kwargs...)
function DouglasRachford!(
        M::AbstractManifold,
        f::TF,
        proxes_f::Vector{<:Any},
        p;
        evaluation = AllocatingEvaluation(),
        parallel::Int = 0,
        kwargs...,
    ) where {TF}
    N, f_, (prox1, prox2), parallel_, p0 = parallel_to_alternating_DR(
        M, f, proxes_f, p, parallel, evaluation
    )
    mpo = ManifoldProximalMapObjective(f_, (prox1, prox2); evaluation = evaluation)
    return DouglasRachford!(
        N, mpo, p0; evaluation = evaluation, parallel = parallel_, kwargs...
    )
end
function DouglasRachford!(
        M::AbstractManifold,
        mpo::O,
        p;
        λ::Tλ = (iter) -> 1.0,
        α::Tα = (iter) -> 0.9,
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        inverse_retraction_method::AbstractInverseRetractionMethod = default_inverse_retraction_method(
            M, typeof(p)
        ),
        reflection_evaluation::E = AllocatingEvaluation(),
        # Adapt to evaluation type
        R::TR = if reflection_evaluation == InplaceEvaluation()
            (M, r, p, q) -> Manopt.reflect!(
                M,
                r,
                p,
                q;
                retraction_method = retraction_method,
                inverse_retraction_method = inverse_retraction_method,
            )
        else
            (M, p, q) -> Manopt.reflect(
                M,
                p,
                q;
                retraction_method = retraction_method,
                inverse_retraction_method = inverse_retraction_method,
            )
        end,
        parallel::Int = 0,
        stopping_criterion::StoppingCriterion = StopAfterIteration(200) |
            StopWhenChangeLess(M, 1.0e-5),
        kwargs..., #especially may contain decorator options
    ) where {
        Tλ,
        Tα,
        TR,
        O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective},
        E <: AbstractEvaluationType,
    }
    keywords_accepted(DouglasRachford!; kwargs...)
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    drs = DouglasRachfordState(
        M;
        p = p,
        λ = λ,
        α = α,
        R = R,
        reflection_evaluation = reflection_evaluation,
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
        stopping_criterion = stopping_criterion,
        parallel = parallel > 0,
    )
    ddrs = decorate_state!(drs; kwargs...)
    solve!(dmp, ddrs)
    return get_solver_return(get_objective(dmp), ddrs)
end
calls_with_kwargs(::typeof(DouglasRachford!)) = (decorate_objective!, decorate_state!)

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
end #
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
function step_solver!(amp::AbstractManoptProblem, drs::DouglasRachfordState, k)
    M = get_manifold(amp)
    get_proximal_map!(amp, drs.p_tmp, drs.λ(k), drs.s, 1)
    #dispatch on allocation type for the reflection, see below.
    _reflect!(M, drs.s_tmp, drs.p_tmp, drs.s, drs.R, drs.reflection_evaluation)
    get_proximal_map!(amp, drs.p, drs.λ(k), drs.s_tmp, 2)
    _reflect!(M, drs.s_tmp, drs.p, drs.s_tmp, drs.R, drs.reflection_evaluation)
    # relaxation
    drs.s = ManifoldsBase.retract_fused(
        M,
        drs.s,
        inverse_retract(M, drs.s, drs.s_tmp, drs.inverse_retraction_method),
        drs.α(k),
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

@doc """
    DouglasRachford(M, f, proxes_f, p; kwargs...)

a doc string with some math ``t_{k+1} = g(α_k; t_k, s_k)``
"""
DouglasRachford(M, f, proxes_f, p; kwargs...)
