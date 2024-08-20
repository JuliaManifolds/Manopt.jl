#
#
# Proximal Point Problem and State
#
#
@doc raw"""
    ManifoldProximalMapObjective{E<:AbstractEvaluationType, TC, TP, V <: Vector{<:Integer}} <: AbstractManifoldCostObjective{E, TC}

specify a problem for solvers based on the evaluation of proximal maps.

# Fields

* `cost`: a function ``F:\mathcal M→ℝ`` to
  minimize
* `proxes`: proximal maps ``\operatorname{prox}_{λ\varphi}:\mathcal M→\mathcal M``
  as functions `(M, λ, p) -> q`.
* `number_of_proxes`: number of proximal maps per function,
  to specify when one of the maps is a combined one such that the proximal maps
  functions return more than one entry per function, you have to adapt this value.
  if not specified, it is set to one prox per function.

# Constructor

    ManifoldProximalMapObjective(cost, proxes, numer_of_proxes=onex(length(proxes));
       evaluation=Allocating)


# See also

[`cyclic_proximal_point`](@ref), [`get_cost`](@ref), [`get_proximal_map`](@ref)
"""
mutable struct ManifoldProximalMapObjective{E<:AbstractEvaluationType,TC,TP,V} <:
               AbstractManifoldCostObjective{E,TC}
    cost::TC
    proximal_maps!!::TP
    number_of_proxes::V
    function ManifoldProximalMapObjective(
        f,
        proxes_f::Union{Tuple,AbstractVector};
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    )
        np = ones(length(proxes_f))
        return new{typeof(evaluation),typeof(f),typeof(proxes_f),typeof(np)}(
            f, proxes_f, np
        )
    end
    function ManifoldProximalMapObjective(
        f::F,
        proxes_f::Union{Tuple,AbstractVector},
        nOP::Vector{<:Integer};
        evaluation::E=AllocatingEvaluation(),
    ) where {E<:AbstractEvaluationType,F}
        return if length(nOP) != length(proxes_f)
            throw(
                ErrorException(
                    "The number_of_proxes ($(nOP)) has to be the same length as the number of proxes ($(length(proxes_f)).",
                ),
            )
        else
            new{E,F,typeof(proxes_f),typeof(nOP)}(f, proxes_f, nOP)
        end
    end
end
function check_prox_number(n, i)
    (i > n) && throw(ErrorException("the $(i)th entry does not exists, only $n available."))
    return true
end
@doc raw"""
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p)
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p, i)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p, i)

evaluate the (`i`th) proximal map of `ManifoldProximalMapObjective p` at the point `p` of `p.M` with parameter ``λ>0``.
"""
get_proximal_map(::AbstractManifold, ::ManifoldProximalMapObjective, ::Any...)

function get_proximal_map(amp::AbstractManoptProblem, λ, p, i)
    return get_proximal_map(get_manifold(amp), get_objective(amp), λ, p, i)
end
function get_proximal_map!(amp::AbstractManoptProblem, q, λ, p, i)
    return get_proximal_map!(get_manifold(amp), q, get_objective(amp), λ, p, i)
end
function get_proximal_map(amp::AbstractManoptProblem, λ, p)
    return get_proximal_map(get_manifold(amp), get_objective(amp), λ, p)
end
function get_proximal_map!(amp::AbstractManoptProblem, q, λ, p)
    return get_proximal_map!(get_manifold(amp), q, get_objective(amp), λ, p)
end

function get_proximal_map(
    M::AbstractManifold, mpo::ManifoldProximalMapObjective{AllocatingEvaluation}, λ, p, i
)
    check_prox_number(length(mpo.proximal_maps!!), i)
    return mpo.proximal_maps!![i](M, λ, p)
end
function get_proximal_map(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, λ, p, i
)
    return get_proximal_map(M, get_objective(admo, false), λ, p, i)
end

function get_proximal_map!(
    M::AbstractManifold, q, mpo::ManifoldProximalMapObjective{AllocatingEvaluation}, λ, p, i
)
    check_prox_number(length(mpo.proximal_maps!!), i)
    copyto!(M, q, mpo.proximal_maps!![i](M, λ, p))
    return q
end
function get_proximal_map!(
    M::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, λ, p, i
)
    return get_proximal_map!(M, q, get_objective(admo, false), λ, p, i)
end
function get_proximal_map(
    M::AbstractManifold, mpo::ManifoldProximalMapObjective{InplaceEvaluation}, λ, p, i
)
    check_prox_number(length(mpo.proximal_maps!!), i)
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps!![i](M, q, λ, p)
    return q
end
function get_proximal_map!(
    M::AbstractManifold, q, mpo::ManifoldProximalMapObjective{InplaceEvaluation}, λ, p, i
)
    check_prox_number(length(mpo.proximal_maps!!), i)
    mpo.proximal_maps!![i](M, q, λ, p)
    return q
end
#
#
# Proximal based State
#
#
@doc """
    CyclicProximalPointState <: AbstractManoptSolverState

stores options for the [`cyclic_proximal_point`](@ref) algorithm. These are the

# Fields

$(_var(:Field, :p))
* $_field_stop
* `λ`:         a function for the values of ``λ_k`` per iteration(cycle ``ì``
* `oder_type`: whether to use a randomly permuted sequence (`:FixedRandomOrder`),
  a per cycle permuted sequence (`:RandomOrder`) or the default linear one.

# Constructor

    CyclicProximalPointState(M; kwargs...)

Generate the options

# Keyword arguments

* `evaluation_order=:LinearOrder`: soecify the `order_type`
* `λ=i -> 1.0 / i` a function to compute the ``λ_k, k ∈ $(_l_Manifold("N"))``,
* $(_kw_p_default): $(_kw_p)
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(2000)`

# See also

[`cyclic_proximal_point`](@ref)
"""
mutable struct CyclicProximalPointState{P,TStop<:StoppingCriterion,Tλ} <:
               AbstractManoptSolverState
    p::P
    stop::TStop
    λ::Tλ
    order_type::Symbol
    order::AbstractVector{Int}
end

function CyclicProximalPointState(
    M::AbstractManifold;
    p::P=rand(M),
    stopping_criterion::S=StopAfterIteration(2000),
    λ::F=(i) -> 1.0 / i,
    evaluation_order::Symbol=:LinearOrder,
) where {P,S,F}
    return CyclicProximalPointState{P,S,F}(p, stopping_criterion, λ, evaluation_order, [])
end
get_iterate(cpps::CyclicProximalPointState) = cpps.p
function set_iterate!(cpps::CyclicProximalPointState, p)
    cpps.p = p
    return p
end

#
# Debug
#
# Debug the Cyclic Proximal point parameter
#
@doc raw"""
    DebugProximalParameter <: DebugAction

print the current iterates proximal point algorithm parameter given by
[`AbstractManoptSolverState`](@ref)s `o.λ`.
"""
mutable struct DebugProximalParameter <: DebugAction
    io::IO
    format::String
    function DebugProximalParameter(;
        long::Bool=false,
        prefix=long ? "Proximal Map Parameter λ(i):" : "λ:",
        format="$prefix%s",
        io::IO=stdout,
    )
        return new(io, format)
    end
end
function (d::DebugProximalParameter)(
    ::AbstractManoptProblem, cpps::CyclicProximalPointState, k::Int
)
    (k > 0) && Printf.format(d.io, Printf.Format(d.format), cpps.λ(k))
    return nothing
end

#
# Record
@doc raw"""
    RecordProximalParameter <: RecordAction

record the current iterates proximal point algorithm parameter given by in
[`AbstractManoptSolverState`](@ref)s `o.λ`.
"""
mutable struct RecordProximalParameter <: RecordAction
    recorded_values::Array{Float64,1}
    RecordProximalParameter() = new(Array{Float64,1}())
end
function (r::RecordProximalParameter)(
    ::AbstractManoptProblem, cpps::CyclicProximalPointState, k::Int
)
    return record_or_reset!(r, cpps.λ(k), k)
end
