#
#
# State
"""
    ProximalPointState{P} <: AbstractGradientSolverState

# Fields

$(_var(:Field, :p; add = [:as_Iterate]))
$(_var(:Field, :stopping_criterion, "stop"))
* `λ`:         a function for the values of ``λ_k`` per iteration(cycle ``k``

# Constructor

    ProximalPointState(M::AbstractManifold; kwargs...)

Initialize the proximal point method solver state, where

## Input

$(_var(:Argument, :M; type = true))

## Keyword arguments

* `λ=k -> 1.0` a function to compute the ``λ_k, k ∈ $(_tex(:Cal, "N"))``,
$(_var(:Keyword, :p; add = :as_Initial))
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(100)`"))

# See also

[`proximal_point`](@ref)
"""
mutable struct ProximalPointState{P, Tλ, TStop <: StoppingCriterion} <:
    AbstractGradientSolverState
    λ::Tλ
    p::P
    stop::TStop
end
function ProximalPointState(
        M::AbstractManifold;
        λ::F = k -> 1.0,
        p::P = rand(M),
        stopping_criterion::SC = StopAfterIteration(200),
    ) where {P, F, SC <: StoppingCriterion}
    return ProximalPointState{P, F, SC}(λ, p, stopping_criterion)
end
function show(io::IO, gds::ProximalPointState)
    i = get_count(gds, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(gds.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Proximal Point Method
    $Iter

    ## Stopping criterion

    $(status_summary(gds.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
#
#
# solver interface
_doc_PPA = """
    proximal_point(M, prox_f, p=rand(M); kwargs...)
    proximal_point(M, mpmo, p=rand(M); kwargs...)
    proximal_point!(M, prox_f, p; kwargs...)
    proximal_point!(M, mpmo, p; kwargs...)

Perform the proximal point algoritm from [FerreiraOliveira:2002](@cite) which reads

```math
p^{(k+1)} = $(_tex(:prox))_{λ_kf}(p^{(k)})
```

# Input

$(_var(:Argument, :M; type = true))
* `prox_f`: a proximal map `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the summands of ``f`` (see `evaluation`)

# Keyword arguments

$(_var(:Keyword, :evaluation))
* `f=nothing`: a cost function ``f: $(_math(:M))→ℝ`` to minimize. For running the algorithm, ``f`` is not required, but for example when recording the cost or using a stopping criterion that requires a cost function.
* `λ= k -> 1.0`: a function returning the (square summable but not summable) sequence of ``λ_i``
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-12)`)"))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_PPA)"
proximal_point(M::AbstractManifold, args...; kwargs...)
function proximal_point(
        M::AbstractManifold,
        prox_f,
        p = rand(M);
        f = nothing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    prox_f_ = _ensure_mutating_prox(prox_f, p, evaluation)
    mpo = ManifoldProximalMapObjective(f_, prox_f_; evaluation = evaluation)
    rs = proximal_point(M, mpo, p_; evaluation = evaluation, kwargs...)
    return _ensure_matching_output(p, rs)
end
function proximal_point(
        M::AbstractManifold, mpo::O, p; kwargs...
    ) where {O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(proximal_point; kwargs...)
    q = copy(M, p)
    return proximal_point!(M, mpo, q; kwargs...)
end
calls_with_kwargs(::typeof(proximal_point)) = (proximal_point!,)

@doc "$(_doc_PPA)"
proximal_point!(M::AbstractManifold, args...; kwargs...)
function proximal_point!(
        M::AbstractManifold,
        prox_f,
        p;
        f = nothing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    mpo = ManifoldProximalMapObjective(f, prox_f; evaluation = evaluation)
    return proximal_point!(M, mpo, p; evaluation = evaluation, kwargs...)
end
function proximal_point!(
        M::AbstractManifold,
        mpo::O,
        p;
        stopping_criterion::StoppingCriterion = StopAfterIteration(1000) |
            StopWhenChangeLess(M, 1.0e-12),
        λ = k -> 1,
        kwargs...,
    ) where {O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(proximal_point!; kwargs...)
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    pps = ProximalPointState(M; p = p, stopping_criterion = stopping_criterion, λ = λ)
    dpps = decorate_state!(pps; kwargs...)
    solve!(dmp, dpps)
    return get_solver_return(get_objective(dmp), dpps)
end
calls_with_kwargs(::typeof(proximal_point!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(::AbstractManoptProblem, pps::ProximalPointState)
    return pps
end
function step_solver!(amp::AbstractManoptProblem, pps::ProximalPointState, k)
    get_proximal_map!(amp, pps.p, pps.λ(k), pps.p)
    return pps
end
