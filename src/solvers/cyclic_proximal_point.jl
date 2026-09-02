@doc """
    CyclicProximalPointState <: AbstractManoptSolverState

stores options for the [`cyclic_proximal_point`](@ref) algorithm. These are the

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:stopping_criterion; name = "stop"))
* `λ`:         a function for the values of ``λ_k`` per iteration(cycle ``k``
* `order_type`: specify whether to use a fixed randomly permuted sequence (`:FixedRandom`),
  a per cycle newly permuted sequence (`:Random`), or the default `:Linear` order.

# Constructor

    CyclicProximalPointState(M::AbstractManifold; kwargs...)

Generate the options

## Input

$(_args(:M))

# Keyword arguments

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `evaluation_order=:Linear`: specify whether to use a fixed randomly permuted sequence (`:FixedRandom`),
  a per cycle newly permuted sequence (`:Random`), or the default `:Linear` order.
* `λ=i -> 1.0 / i` a function to compute the ``λ_k, k ∈ $(_math(:Manifold; M = "N"))``,
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(2000)"))

# See also

[`cyclic_proximal_point`](@ref)
"""
mutable struct CyclicProximalPointState{P, C <: AbstractDict{Symbol}, SC <: StoppingCriterion, Tλ, A <: AbstractVector{<:Int}} <: AbstractManoptSolverState
    callbacks::C
    order::A
    order_type::Symbol
    p::P
    stop::SC
    λ::Tλ
    function CyclicProximalPointState(;
            callbacks::C = Dict{Symbol, Function}(),
            order::A, order_type::Symbol, p::P, stopping_criterion::SC, λ::Tλ,
        ) where {P, C <: AbstractDict{Symbol}, SC <: StoppingCriterion, Tλ, A <: AbstractVector{<:Int}}
        return new{P, C, SC, Tλ, A}(callbacks, order, order_type, p, stopping_criterion, λ)
    end
end

function CyclicProximalPointState(
        M::AbstractManifold;
        callbacks::C = Dict{Symbol, Function}(),
        evaluation_order::Symbol = :Linear,
        p::P = rand(M),
        stopping_criterion::S = StopAfterIteration(2000),
        λ::F = (i) -> 1.0 / i,
    ) where {P, C <: AbstractDict{Symbol}, S, F}
    (evaluation_order in (:Linear, :FixedRandom, :Random)) || throw(
        DomainError(evaluation_order, "The evaluation order has to be one of :Linear, :FixedRandom, or :Random.")
    )
    return CyclicProximalPointState(; callbacks = callbacks, order = Int[], order_type = evaluation_order, p = p, stopping_criterion = stopping_criterion, λ = λ)
end
get_iterate(cpps::CyclicProximalPointState) = cpps.p
function set_iterate!(cpps::CyclicProximalPointState, p)
    cpps.p = p
    return p
end
get_callbacks(cpps::CyclicProximalPointState) = cpps.callbacks
function Base.show(io::IO, cpps::CyclicProximalPointState)
    print(io, "CyclicProximalPointState(; ")
    print(io, "callbacks = "); print(io, cpps.callbacks); print(io, ", ")
    print(io, "order = "); print(io, cpps.order); print(io, ", ")
    print(io, "order_type = "); print(io, cpps.order_type); print(io, ", ")
    print(io, "p = "); print(io, cpps.p); print(io, ", ")
    print(io, "stopping_criterion = "); print(io, cpps.stop); print(io, ", ")
    print(io, "λ = "); print(io, cpps.λ)
    return print(io, ")")
end
function status_summary(cpps::CyclicProximalPointState; context::Symbol = :default)
    (context === :short) && return repr(cpps)
    i = get_count(cpps, :Iterations)
    conv_inl = (i > 0) ? (has_converged(cpps.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the cyclic proximal point algorithm$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(cpps.stop) ? "Yes" : "No"
    as = _callbacks_summary(cpps)
    s = """
    # Solver state for `Manopt.jl`s Cyclic Proximal Point Algorithm
    $Iter
    ## Parameters$(as)
    * evaluation order of the proximal maps: :$(cpps.order_type)

    ## Stopping criterion
    $(_in_str(status_summary(cpps.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv"""
    return s
end

function (d::DebugProximalParameter)(
        ::AbstractManoptProblem, cpps::CyclicProximalPointState, k::Int
    )
    (k >= (d.at_init ? 0 : 1)) && Printf.format(d.io, Printf.Format(d.format), cpps.λ(k))
    return nothing
end

function (r::RecordProximalParameter)(
        ::AbstractManoptProblem, cpps::CyclicProximalPointState, k::Int
    )
    return record_or_reset!(r, cpps.λ(k), k)
end

_doc_CPPA = """
    cyclic_proximal_point(M, f, proxes_f, p; kwargs...)
    cyclic_proximal_point(M, mpo, p; kwargs...)
    cyclic_proximal_point!(M, f, proxes_f, p; kwargs...)
    cyclic_proximal_point!(M, mpo, p; kwargs...)

perform a cyclic proximal point algorithm. This can be done in-place of `p`.

# Input

$(_args(:M))
* `f`:        a cost function ``f: $(_math(:Manifold))→ℝ`` to minimize
* `proxes_f`: an Array of proximal maps (`Function`s) `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the summands of ``f`` (see `evaluation`)

where `f` and the proximal maps `proxes_f` can also be given directly as a [`ManifoldProximalMapObjective`](@ref) `mpo`

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
$(_kwargs(:evaluation))
* `evaluation_order=:Linear`: specify whether to use a fixed randomly permuted sequence (`:FixedRandom`),
  a per cycle newly permuted sequence (`:Random`), or the default `:Linear` order.
* `λ=iter -> 1/iter`:         a function returning the (square summable but not summable) sequence of ``λ_i``
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(5000)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-12)"))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_CPPA)"
cyclic_proximal_point(M::AbstractManifold, args...; kwargs...)
function cyclic_proximal_point(
        M::AbstractManifold, f, proxes_f::Union{Tuple, AbstractVector}, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    )
    p_ = maybe_wrap_variable(p)
    mpo = ManifoldProximalMapObjective(f, proxes_f; evaluation = evaluation, p = p)
    rs = cyclic_proximal_point(M, mpo, p_; evaluation = evaluation, kwargs...)
    return maybe_unwrap_variable(p, rs)
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
        M::AbstractManifold, f, proxes_f::Union{Tuple, AbstractVector}, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    )
    mpo = ManifoldProximalMapObjective(f, proxes_f; evaluation = evaluation)
    return cyclic_proximal_point!(M, mpo, p; evaluation = evaluation, kwargs...)
end
function cyclic_proximal_point!(
        M::AbstractManifold, mpo::O, p;
        callbacks = Dict{Symbol, Function}(),
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
        callbacks = process_callbacks_arg(callbacks, CyclicProximalPointState),
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
    c = length(get_objective(amp, true).proximal_maps!)
    cpps.order = collect(1:c)
    (cpps.order_type == :FixedRandom) && shuffle!(cpps.order)
    return cpps
end
function step_solver!(amp::AbstractManoptProblem, cpps::CyclicProximalPointState, k)
    λi = cpps.λ(k)
    for k in cpps.order
        get_proximal_map!(amp, cpps.p, λi, cpps.p, k)
    end
    (cpps.order_type == :Random) && shuffle!(cpps.order)
    return cpps
end
