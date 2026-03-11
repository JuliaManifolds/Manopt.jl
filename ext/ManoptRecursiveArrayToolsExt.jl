module ManoptRecursiveArrayToolsExt
using Manopt
using ManifoldsBase
using ManifoldsBase: submanifold_components
import Manopt:
    max_stepsize,
    alternating_gradient_descent,
    alternating_gradient_descent!,
    get_gradient,
    get_gradient!,
    set_parameter!
using Manopt: _tex, ManifoldDefaultsFactory, _produce_type

using RecursiveArrayTools

@inline _unwrap_extruded_arg(x) = x
@inline _unwrap_extruded_arg(x::Base.Broadcast.Extruded) = x.x

@inline function _axpy_partition!(dest::AbstractArray{T}, src::AbstractArray{T}) where {T}
    axes(dest) == axes(src) ||
        throw(DimensionMismatch("ArrayPartition blocks must have matching axes for in-place addition."))
    @inbounds for i in eachindex(dest, src)
        dest[i] += src[i]
    end
    return dest
end

@inline function _axpy_partition!(
        dest::StridedArray{T},
        src::Base.ReshapedArray{
            T,
            N,
            <:SubArray{T, 1, <:Manopt.BlockNonzeroVector, Tuple{UnitRange{I}}, false},
            MI,
        },
    ) where {T, N, I <: Integer, MI}
    axes(dest) == axes(src) ||
        throw(DimensionMismatch("ArrayPartition blocks must have matching axes for in-place addition."))

    src_view = parent(src)
    src_parent = parent(src_view)
    src_range = parentindices(src_view)[1]
    src_first = first(src_range)
    src_last = last(src_range)
    dest_linear = vec(dest)

    for k in eachindex(src_parent.blocks)
        block = src_parent.blocks[k]
        block_start = src_parent.starts[k]
        block_end = block_start + length(block) - 1

        overlap_start = max(src_first, block_start)
        overlap_end = min(src_last, block_end)
        overlap_start > overlap_end && continue

        src_local =
            (overlap_start - src_first + 1):(overlap_end - src_first + 1)
        block_local =
            (overlap_start - block_start + 1):(overlap_end - block_start + 1)

        @views dest_linear[src_local] .+= block[block_local]
    end
    return dest
end

@inline function _add_arraypartition_blocks!(
        dest::ArrayPartition{T, <:Tuple{AbstractArray{T}, Vararg{AbstractArray{T}}}},
        src::ArrayPartition,
    ) where {T}
    length(dest.x) == length(src.x) ||
        throw(DimensionMismatch("ArrayPartition blocks must have matching lengths for in-place addition."))
    map(_axpy_partition!, dest.x, src.x)
    return dest
end

function Base.copyto!(
        dest::ArrayPartition{T, <:Tuple{Array{T, 3}, Matrix{T}, Matrix{T}}},
        bc::Base.Broadcast.Broadcasted{RecursiveArrayTools.ArrayPartitionStyle{Style}, Axes, typeof(+)},
    ) where {T, Style <: Union{Nothing, Base.Broadcast.BroadcastStyle}, Axes}
    bc = Base.Broadcast.instantiate(bc)
    if length(bc.args) == 2
        lhs = _unwrap_extruded_arg(bc.args[1])
        rhs = _unwrap_extruded_arg(bc.args[2])
        if lhs === dest && rhs isa ArrayPartition
            return _add_arraypartition_blocks!(dest, rhs)
        elseif rhs === dest && lhs isa ArrayPartition
            return _add_arraypartition_blocks!(dest, lhs)
        end
    end
    return invoke(Base.copyto!, Tuple{ArrayPartition, Base.Broadcast.Broadcasted}, dest, bc)
end

@doc """
    X = get_gradient(M::ProductManifold, ago::ManifoldAlternatingGradientObjective, p)
    get_gradient!(M::ProductManifold, P::ManifoldAlternatingGradientObjective, X, p)

Evaluate all summands gradients at a point `p` on the `ProductManifold M` (in place of `X`)
"""
get_gradient(M::ProductManifold, ::ManifoldAlternatingGradientObjective, ::Any...)

function get_gradient(
        M::AbstractManifold,
        mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    return ArrayPartition([gi(M, p) for gi in mago.gradient!!]...)
end

@doc """
    X = get_gradient(M::AbstractManifold, p::ManifoldAlternatingGradientObjective, p, i)
    get_gradient!(M::AbstractManifold, p::ManifoldAlternatingGradientObjective, X, p, i)

Evaluate one of the component gradients ``$(_tex(:grad)) f_i``, ``i∈ $(_tex(:set, "1,…,n"))``, at `x` (in place of `Y`).
"""
function get_gradient(
        M::ProductManifold,
        mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
        i,
    ) where {TC}
    return get_gradient(M, mago, p)[M, i]
end
function get_gradient!(
        M::AbstractManifold,
        X,
        mago::ManifoldAlternatingGradientObjective{InplaceEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    for (gi, Xi) in zip(mago.gradient!!, submanifold_components(M, X))
        gi(M, Xi, p)
    end
    return X
end

function get_gradient!(
        M::ProductManifold,
        X,
        mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
        k,
    ) where {TC}
    copyto!(M[k], X, mago.gradient!!(M, p)[M, k])
    return X
end

function alternating_gradient_descent(
        M::ProductManifold,
        f,
        grad_f::Union{TgF, AbstractVector{<:TgF}},
        p = rand(M);
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    ) where {TgF}
    ago = ManifoldAlternatingGradientObjective(f, grad_f; evaluation = evaluation)
    return alternating_gradient_descent(M, ago, p; evaluation = evaluation, kwargs...)
end
function alternating_gradient_descent(
        M::ProductManifold, ago::ManifoldAlternatingGradientObjective, p; kwargs...
    )
    Manopt.keywords_accepted(alternating_gradient_descent; kwargs...)
    q = copy(M, p)
    return alternating_gradient_descent!(M, ago, q; kwargs...)
end

function alternating_gradient_descent!(
        M::ProductManifold,
        f,
        grad_f::Union{TgF, AbstractVector{<:TgF}},
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    ) where {TgF}
    agmo = ManifoldAlternatingGradientObjective(f, grad_f; evaluation = evaluation)
    return alternating_gradient_descent!(M, agmo, p; evaluation = evaluation, kwargs...)
end
function alternating_gradient_descent!(
        M::ProductManifold,
        agmo::ManifoldAlternatingGradientObjective,
        p;
        inner_iterations::Int = 5,
        stopping_criterion::StoppingCriterion = StopAfterIteration(100) |
            StopWhenGradientNormLess(1.0e-9),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, AlternatingGradientDescentState
        ),
        order_type::Symbol = :Linear,
        order = collect(1:length(M.manifolds)),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        kwargs...,
    )
    Manopt.keywords_accepted(alternating_gradient_descent!; kwargs...)
    dagmo = Manopt.decorate_objective!(M, agmo; kwargs...)
    dmp = DefaultManoptProblem(M, dagmo)
    agds = AlternatingGradientDescentState(
        M;
        p = p,
        inner_iterations = inner_iterations,
        stopping_criterion = stopping_criterion,
        stepsize = _produce_type(stepsize, M),
        order_type = order_type,
        order = order,
        retraction_method = retraction_method,
    )
    agds = Manopt.decorate_state!(agds; kwargs...)
    Manopt.solve!(dmp, agds)
    return Manopt.get_solver_return(get_objective(dmp), agds)
end

## Prox TV on
end
