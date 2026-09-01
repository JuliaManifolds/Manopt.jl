#
#
# State
"""
    ProximalPointState{P} <: AbstractGradientSolverState

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:stopping_criterion; name = "stop"))
* `λ`:         a function for the values of ``λ_k`` per iteration/cycle ``k``

# Constructor

    ProximalPointState(M::AbstractManifold; kwargs...)

Initialize the proximal point method solver state, where

## Input

$(_args(:M))

## Keyword arguments

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `λ=k -> 1.0`: a function to compute ``λ_k`` for ``k ∈ ℕ``,
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)"))

# See also

[`proximal_point`](@ref)
"""
mutable struct ProximalPointState{P, Tλ, C <: AbstractDict{Symbol}, TStop <: StoppingCriterion} <: AbstractGradientSolverState
    callbacks::C
    λ::Tλ
    p::P
    stop::TStop
    function ProximalPointState(
            M::AbstractManifold;
            callbacks::C = Dict{Symbol, Function}(), λ::F = k -> 1.0,
            p::P = rand(M), stopping_criterion::SC = StopAfterIteration(200),
        ) where {P, F, C <: AbstractDict{Symbol}, SC <: StoppingCriterion}
        return ProximalPointState(; callbacks = callbacks, λ = λ, p = p, stopping_criterion = stopping_criterion)
    end
    function ProximalPointState(; callbacks::C, λ::F, p::P, stopping_criterion::SC) where {P, F, C <: AbstractDict{Symbol}, SC <: StoppingCriterion}
        return new{P, F, C, SC}(callbacks, λ, p, stopping_criterion)
    end
end
get_callbacks(pps::ProximalPointState) = pps.callbacks
function Base.show(io::IO, pps::ProximalPointState)
    print(io, "ProximalPointState(; ")
    return print(io, "callbacks = ", pps.callbacks, ", λ = $(pps.λ), p = $(pps.p), stopping_criterion = $(pps.stop))")
end
function status_summary(pps::ProximalPointState; context::Symbol = :default)
    (context === :short) && return repr(pps)
    i = get_count(pps, :Iterations)
    conv_inl = (i > 0) ? (has_converged(pps.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the proximal point algorithm$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(pps.stop) ? "Yes" : "No"
    as = _callbacks_summary(pps)
    (length(as) > 0) && (as = "## Parameters\n$(as)\n\n")
    s = """
    # Solver state for `Manopt.jl`s Proximal Point Method
    $Iter$(as)
    ## Stopping criterion
    $(_in_str(status_summary(pps.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv"""
    return s
end
#
#
# solver interface
_doc_PPA = """
    proximal_point(M, prox_f, p=rand(M); kwargs...)
    proximal_point(M, mpmo, p=rand(M); kwargs...)
    proximal_point!(M, prox_f, p; kwargs...)
    proximal_point!(M, mpmo, p; kwargs...)

Perform the proximal point algorithm from [FerreiraOliveira:2002](@cite) which reads

```math
p^{(k+1)} = $(_tex(:prox))_{λ_kf}(p^{(k)})
```

# Input

$(_args(:M))
* `prox_f`: a proximal map `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the summands of ``f`` (see `evaluation`)

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
$(_kwargs(:evaluation))
* `f=nothing`: a cost function ``f: $(_math(:Manifold))→ℝ`` to minimize. For running the algorithm, ``f`` is not required, but for example when recording the cost or using a stopping criterion that requires a cost function.
* `λ= k -> 1.0`: a function returning the (square summable but not summable) sequence of ``λ_i``
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-12)"))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_PPA)"
proximal_point(M::AbstractManifold, args...; kwargs...)
function proximal_point(
        M::AbstractManifold, prox_f, p = rand(M);
        f = nothing, evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    )
    p_ = maybe_wrap_variable(p)
    mpo = ManifoldProximalMapObjective(f, prox_f; evaluation = evaluation, p = p)
    rs = proximal_point(M, mpo, p_; evaluation = evaluation, kwargs...)
    return maybe_unwrap_variable(p, rs)
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
        M::AbstractManifold, prox_f, p;
        f = nothing, evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    mpo = ManifoldProximalMapObjective(f, prox_f; evaluation = evaluation, p = p)
    return proximal_point!(M, mpo, p; evaluation = evaluation, kwargs...)
end
function proximal_point!(
        M::AbstractManifold, mpo::O, p;
        callbacks = Dict{Symbol, Function}(),
        stopping_criterion::StoppingCriterion = StopAfterIteration(1000) | StopWhenChangeLess(M, 1.0e-12),
        λ = k -> 1, kwargs...,
    ) where {O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(proximal_point!; kwargs...)
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    pps = ProximalPointState(
        M;
        callbacks = process_callbacks_arg(callbacks, ProximalPointState),
        p = p, stopping_criterion = stopping_criterion, λ = λ
    )
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
