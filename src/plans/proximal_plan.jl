#
#
# Proximal Point Problem and State
#
#
@doc """
    ManifoldProximalMapObjective{E<:AbstractEvaluationType, TC, TP, V <: Vector{<:Integer}} <: AbstractManifoldCostObjective{E, TC}

specify a problem for solvers based on the evaluation of proximal maps,
which represents proximal maps ``$(_tex(:prox))_{λf_i}`` for summands ``f = f_1 + f_2+ … + f_N`` of the cost function ``f``.

# Fields

* `cost`: a function ``f:$(_math(:Manifold))→ℝ`` to
  minimize
* `proxes`: proximal maps ``$(_tex(:prox))_{λf_i}:$(_math(:Manifold)) → $(_math(:Manifold))``
  as functions `(M, λ, p) -> q` or in-place `(M, q, λ, p)`.
* `number_of_proxes`: number of proximal maps per function,
  to specify when one of the maps is a combined one such that the proximal maps
  functions return more than one entry per function, you have to adapt this value.
  if not specified, it is set to one prox per function.

# Constructor

    ManifoldProximalMapObjective(f, proxes_f::Union{Tuple,AbstractVector}, number_of_proxes=onex(length(proxes));
       evaluation=Allocating)

Generate a proximal problem with a tuple or vector of functions, where by default every function computes a single prox
of one component of ``f``.

    ManifoldProximalMapObjective(f, prox_f); evaluation=Allocating)

Generate a proximal objective for ``f`` and its proxial map ``$(_tex(:prox))_{λf}``

# See also

[`cyclic_proximal_point`](@ref), [`get_cost`](@ref), [`get_proximal_map`](@ref)
"""
mutable struct ManifoldProximalMapObjective{E <: AbstractEvaluationType, TC, TP, V} <:
    AbstractManifoldCostObjective{E, TC}
    cost::TC
    proximal_maps!!::TP
    number_of_proxes::V
    function ManifoldProximalMapObjective(
            f,
            proxes_f::Union{Tuple, AbstractVector};
            evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        )
        np = ones(length(proxes_f))
        return new{typeof(evaluation), typeof(f), typeof(proxes_f), typeof(np)}(
            f, proxes_f, np
        )
    end
    function ManifoldProximalMapObjective(
            f::F,
            proxes_f::Union{Tuple, AbstractVector},
            nOP::Vector{<:Integer};
            evaluation::E = AllocatingEvaluation(),
        ) where {E <: AbstractEvaluationType, F}
        return if length(nOP) != length(proxes_f)
            throw(
                ErrorException(
                    "The number_of_proxes ($(nOP)) has to be the same length as the number of proxes ($(length(proxes_f)).",
                ),
            )
        else
            new{E, F, typeof(proxes_f), typeof(nOP)}(f, proxes_f, nOP)
        end
    end
    function ManifoldProximalMapObjective(
            f::F, prox_f::PF; evaluation::E = AllocatingEvaluation()
        ) where {E <: AbstractEvaluationType, F, PF}
        i = 1
        return new{E, F, PF, typeof(i)}(f, prox_f, i)
    end
end
@doc """
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p)
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p, i)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p, i)

evaluate the (`i`th) proximal map of the [`ManifoldProximalMapObjective`](@ref)` mpo` at
the point `p` of `M` with parameter ``λ>0``.
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
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, λ, p
    )
    return get_proximal_map(M, get_objective(admo, false), λ, p)
end

function check_prox_number(pf::Union{Tuple, Vector}, i)
    n = length(pf)
    (i > n) && throw(ErrorException("the $(i)th entry does not exists, only $n available."))
    return true
end

function get_proximal_map(
        M::AbstractManifold,
        mpo::ManifoldProximalMapObjective{AllocatingEvaluation, F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    check_prox_number(mpo.proximal_maps!!, i)
    return mpo.proximal_maps!![i](M, λ, p)
end
function get_proximal_map(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_proximal_map(M, get_objective(admo, false), args...)
end
function get_proximal_map!(
        M::AbstractManifold,
        q,
        mpo::ManifoldProximalMapObjective{AllocatingEvaluation, F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    check_prox_number(mpo.proximal_maps!!, i)
    copyto!(M, q, mpo.proximal_maps!![i](M, λ, p))
    return q
end
function get_proximal_map!(
        M::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_proximal_map!(M, q, get_objective(admo, false), args...)
end
function get_proximal_map(
        M::AbstractManifold,
        mpo::ManifoldProximalMapObjective{InplaceEvaluation, F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    check_prox_number(mpo.proximal_maps!!, i)
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps!![i](M, q, λ, p)
    return q
end
function get_proximal_map!(
        M::AbstractManifold, q,
        mpo::ManifoldProximalMapObjective{InplaceEvaluation, F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    check_prox_number(mpo.proximal_maps!!, i)
    mpo.proximal_maps!![i](M, q, λ, p)
    return q
end
#
#
# Single function accessors
function get_proximal_map(
        M::AbstractManifold, mpo::ManifoldProximalMapObjective{AllocatingEvaluation}, λ, p
    )
    return mpo.proximal_maps!!(M, λ, p)
end
function get_proximal_map!(
        M::AbstractManifold, q, mpo::ManifoldProximalMapObjective{AllocatingEvaluation}, λ, p
    )
    copyto!(M, q, mpo.proximal_maps!!(M, λ, p))
    return q
end
function get_proximal_map(
        M::AbstractManifold, mpo::ManifoldProximalMapObjective{InplaceEvaluation}, λ, p
    )
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps!!(M, q, λ, p)
    return q
end
function get_proximal_map!(
        M::AbstractManifold, q, mpo::ManifoldProximalMapObjective{InplaceEvaluation}, λ, p
    )
    mpo.proximal_maps!!(M, q, λ, p)
    return q
end
function status_summary(mpo::ManifoldProximalMapObjective; context::Symbol = :default)
    (context === :short) && (return repr(mpo))
    return "A proximal map objective for a cost with $(mpo.number_of_proxes) proximal maps"
end
function Base.show(io::IO, mpo::ManifoldProximalMapObjective{E}) where {E}
    print(io, "ManifoldProximalMapObjective(", mpo.cost, ", ", mpo.proximal_maps!!, ", ")
    print(io, mpo.number_of_proxes, "; ", _to_kw(E))
    return print(io, ")")
end

#
# Debug
#
# Debug the Cyclic Proximal point parameter
#
@doc """
    DebugProximalParameter <: DebugAction

print the current iterates proximal point algorithm parameter given by
[`AbstractManoptSolverState`](@ref)s `o.λ`.
"""
mutable struct DebugProximalParameter <: DebugAction
    io::IO
    format::String
    at_init::Bool
    function DebugProximalParameter(;
            long::Bool = false,
            prefix = long ? "Proximal Map Parameter λ(i):" : "λ:",
            format = "$prefix%s",
            io::IO = stdout,
            at_init::Bool = true,
        )
        return new(io, format, at_init)
    end
end
function Base.show(io::IO, d::DebugProximalParameter)
    return print(
        io, "DebugGradientChange(; io = ", d.io, ", format=\"$(escape_string(d.format))\", at_init = $(d.at_init))",
    )
end
function status_summary(d::DebugProximalParameter; context::Symbol = :Default)
    (context === :short) && (return "(:ProxParameter, \"$(escape_string(d.format))\")")
    # Inline and default
    return "A DebugAction printing the proximal parameter as “$(escape_string(d.format))”"
end
#
# Record
@doc """
    RecordProximalParameter{R <: Real} <: RecordAction

record the current iterates proximal point algorithm parameter given by in
[`AbstractManoptSolverState`](@ref)s `o.λ`.

## Constructor
    RecordProximalParameter(r::Type{<:Real}=Float64)
"""
mutable struct RecordProximalParameter{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordProximalParameter(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
show(io::IO, ::RecordProximalParameter{R}) where {R} = print(io, "RecordProximalParameter($R)")
function status_summary(rg::RecordProximalParameter{R}; context::Symbol = :default) where {R}
    (context === :short) && return ":ProximalParameter"
    return "A RecordAction to record the current proximal parameter (of type $R)"
end
