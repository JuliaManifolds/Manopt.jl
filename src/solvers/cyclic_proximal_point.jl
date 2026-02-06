function show(io::IO, cpps::CyclicProximalPointState)
    i = get_count(cpps, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(cpps.stop) ? "Yes" : "No"
    inline && (return "$(repr(cpps)) – $(Iter) $(has_converged(cpps) ? "(converged)" : "")")
    s = """
    # Solver state for `Manopt.jl`s Cyclic Proximal Point Algorithm
    $Iter
    ## Parameters
    * evaluation order of the proximal maps: :$(cpps.order_type)

    ## Stopping criterion
    $(status_summary(cpps.stop; inline = false))
    This indicates convergence: $Conv"""
    return s
end
_doc_CPPA = """
    cyclic_proximal_point(M, f, proxes_f, p; kwargs...)
    cyclic_proximal_point(M, mpo, p; kwargs...)
    cyclic_proximal_point!(M, f, proxes_f; kwargs...)
    cyclic_proximal_point!(M, mpo; kwargs...)

perform a cyclic proximal point algorithm. This can be done in-place of `p`.

# Input

$(_args(:M))
* `f`:        a cost function ``f: $(_math(:Manifold))nifold)))→ℝ`` to minimize
* `proxes_f`: an Array of proximal maps (`Function`s) `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the summands of ``f`` (see `evaluation`)

where `f` and the proximal maps `proxes_f` can also be given directly as a [`ManifoldProximalMapObjective`](@ref) `mpo`

# Keyword arguments

$(_kwargs(:evaluation))
* `evaluation_order=:Linear`: whether to use a randomly permuted sequence (`:FixedRandom`:,
  a per cycle permuted sequence (`:Random`) or the default linear one.
* `λ=iter -> 1/iter`:         a function returning the (square summable but not summable) sequence of ``λ_i``
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(5000)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-12)"))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_CPPA)"
cyclic_proximal_point(M::AbstractManifold, args...; kwargs...)
function cyclic_proximal_point(
        M::AbstractManifold,
        f,
        proxes_f::Union{Tuple, AbstractVector},
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    proxes_f_ = [_ensure_mutating_prox(prox_f, p, evaluation) for prox_f in proxes_f]
    mpo = ManifoldProximalMapObjective(f_, proxes_f_; evaluation = evaluation)
    rs = cyclic_proximal_point(M, mpo, p_; evaluation = evaluation, kwargs...)
    return _ensure_matching_output(p, rs)
end
function cyclic_proximal_point(
        M::AbstractManifold, mpo::O, p; kwargs...
    ) where {O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(cyclic_proximal_point; kwargs...)
    q = copy(M, p)
    return cyclic_proximal_point!(M, mpo, q; kwargs...)
end
calls_with_kwargs(::typeof(cyclic_proximal_point)) = (cyclic_proximal_point!,)

@doc "$(_doc_CPPA)"
cyclic_proximal_point!(M::AbstractManifold, args...; kwargs...)
function cyclic_proximal_point!(
        M::AbstractManifold,
        f,
        proxes_f::Union{Tuple, AbstractVector},
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    mpo = ManifoldProximalMapObjective(f, proxes_f; evaluation = evaluation)
    return cyclic_proximal_point!(M, mpo, p; evaluation = evaluation, kwargs...)
end
function cyclic_proximal_point!(
        M::AbstractManifold,
        mpo::O,
        p;
        evaluation_order::Symbol = :Linear,
        stopping_criterion::StoppingCriterion = StopAfterIteration(5000) |
            StopWhenChangeLess(M, 1.0e-12),
        λ = i -> 1 / i,
        kwargs...,
    ) where {O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(cyclic_proximal_point!; kwargs...)
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    cpps = CyclicProximalPointState(
        M;
        p = p,
        stopping_criterion = stopping_criterion,
        λ = λ,
        evaluation_order = evaluation_order,
    )
    dcpps = decorate_state!(cpps; kwargs...)
    solve!(dmp, dcpps)
    return get_solver_return(get_objective(dmp), dcpps)
end
calls_with_kwargs(::typeof(cyclic_proximal_point!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(amp::AbstractManoptProblem, cpps::CyclicProximalPointState)
    c = length(get_objective(amp, true).proximal_maps!!)
    cpps.order = collect(1:c)
    (cpps.order_type == :FixedRandom) && shuffle!(cpps.order)
    return cpps
end
function step_solver!(amp::AbstractManoptProblem, cpps::CyclicProximalPointState, k)
    λi = cpps.λ(k)
    for k in cpps.order
        get_proximal_map!(amp, cpps.p, λi, cpps.p, k)
    end
    (cpps.order_type == :Random) && shuffle(cpps.order)
    return cpps
end
