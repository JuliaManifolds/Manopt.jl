# Differential function - pass-through
function get_differential_function(
        admo::AbstractDecoratedManifoldObjective, recursive = false
    )
    return get_differential_function(get_objective(admo, recursive))
end

@doc """
     get_differential_function(admo::AbstractManifoldFirstOrderObjective, recursive::Bool=false)

Return the function to evaluate (just) the differential ``Df(p)[X]``.
For a decorated objective, the `recursive` positional parameter determines whether to
directly call this function on the next decorator or whether to get the “most inner” objective.
"""
get_differential_function(::AbstractManifoldFirstOrderObjective; recursive::Bool = false)

function get_differential_function(
        mfo::ManifoldFirstOrderObjective{<:AbstractEvaluationType}, recursive::Bool = false
    )
    if haskey(mfo.functions, :differential)
        return mfo.functions[:differential]
    else
        return (M, p, X; kwargs...) -> get_differential(M, mfo, p, X, kwargs...)
    end
end

# Decorator case
function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient(M, get_objective(admo, false), p)
end
function get_gradient!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient!(M, X, get_objective(admo, false), p)
end

function get_gradient!(
        M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{<:NamedTuple}, p,
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient](M, X, p))
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p)[2])
    return error("$mfo does not seem to provide a gradient")
end

@doc """
    get_gradient_function(amgo::AbstractManifoldFirstOrderObjective, recursive=false)

return the function to evaluate (just) the gradient ``$(_tex(:grad)) f(p)``,
where either the gradient function using the decorator or without the decorator is used.

By default `recursive` is set to `false`, since usually to just pass the gradient function
somewhere, one still wants for example the cached one or the one that still counts calls.

Depending on the [`AbstractEvaluationType`](@ref) `E` this is a function

* `(M, p) -> X` for the [`AllocatingEvaluation`](@ref) case
* `(M, X, p) -> X` for the [`InplaceEvaluation`](@ref) working in-place of `X`.
"""
get_gradient_function(::AbstractManifoldFirstOrderObjective; recursive = false)

function get_gradient_function(admo::AbstractDecoratedManifoldObjective, recursive = false)
    return get_gradient_function(get_objective(admo, recursive))
end
function get_gradient_function(
        mfo::ManifoldFirstOrderObjective, recursive = false
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient])
    return (M, X, p) -> get_gradient!(M, X, mfo, p)
end

#
#  Access cost and gradient – a bit of cases
# -----------------------------
function get_cost_and_gradient(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
    )
    return get_cost_and_gradient(M, get_objective(admo, false), p)
end
function get_cost_and_gradient!(
        M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
    )
    return get_cost_and_gradient!(M, X, get_objective(admo, false), p)
end

function get_cost_and_gradient(
        M::AbstractManifold, mfo::AbstractManifoldFirstOrderObjective, p
    )
    X = zero_vector(M, p)
    return get_cost_and_gradient!(M, X, mfo, p)
end
function get_cost_and_gradient!(
        M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective, p
    )
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p))
    if haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        return mfo.functions[:cost](M, p), mfo.functions[:gradient](M, X, p)
    end
    Y = zero_vector(M, p)
    if haskey(mfo.functions, :costdifferential) && haskey(mfo.functions, :gradient)
        return (
            mfo.functions[:costdifferential](M, p, Y)[1], mfo.functions[:gradient](M, X, p),
        )
    end
    return error("$mfo seems to either have no access to a cost or a gradient")
end
function status_summary(mfo::ManifoldFirstOrderObjective; context::Symbol = :default)
    _is_inline(context) && (return repr(mfo))
    return "A first order objective with $(length(mfo.functions)) provided functions.\n\n" * join([ "* $k:$(_MANOPT_INDENT) $(v)" for (k, v) in zip(keys(mfo.functions), mfo.functions) ], "\n")
end
function Base.show(io::IO, mfo::ManifoldFirstOrderObjective)
    print(io, "ManifoldFirstOrderObjective(; ")
    print(io, join([ "$k = $v" for (k, v) in zip(keys(mfo.functions), mfo.functions)], ", "))
    print(io, ", ")
    return print(io, ")")
end

#
#  Access gradient
# -----------------------------

@doc """
    get_gradient(amp::AbstractManoptProblem, p)
    get_gradient!(amp::AbstractManoptProblem, X, p)

evaluate the gradient of an [`AbstractManoptProblem`](@ref) `amp` at the point `p`.

The evaluation is done in place of `X` for the `!`-variant.
"""
function get_gradient(mp::AbstractManoptProblem, p)
    return get_gradient(get_manifold(mp), get_objective(mp), p)
end
function get_gradient!(mp::AbstractManoptProblem, X, p)
    return get_gradient!(get_manifold(mp), X, get_objective(mp), p)
end

"""
    X = get_subgradient(M::AbstractManifold, sgo::AbstractManifoldFirstOrderObjective, p)
    get_subgradient!(M::AbstractManifold, X, sgo::AbstractManifoldFirstOrderObjective, p)

Evaluate the subgradient, which for the case of a objective having a gradient, means evaluating the
gradient itself.

While in general, the result might not be deterministic, for this case it is.
"""
function get_subgradient(M::AbstractManifold, agmo::AbstractManifoldFirstOrderObjective, p)
    X = zero_vector!(M, p)
    return get_sub_gradient!(M, X, agmo, p)
end
function get_subgradient!(
        M::AbstractManifold, X, agmo::AbstractManifoldFirstOrderObjective, p
    )
    return get_gradient!(M, X, agmo, p)
end

#
# Records
#
@doc """
    RecordGradient <: RecordAction

record the gradient evaluated at the current iterate

# Constructors
    RecordGradient(ξ)

initialize the [`RecordAction`](@ref) to the corresponding type of the tangent vector.
"""
mutable struct RecordGradient{T} <: RecordAction
    recorded_values::Array{T, 1}
    RecordGradient{T}() where {T} = new(Array{T, 1}())
end
RecordGradient(::T) where {T} = RecordGradient{T}()
function (r::RecordGradient{T})(
        ::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    ) where {T}
    return record_or_reset!(r, get_gradient(s), k)
end
show(io::IO, ::RecordGradient{T}) where {T} = print(io, "RecordGradient($T)")
function status_summary(rg::RecordGradient; context::Symbol = :default)
    (context === :short) && return ":Gradient"
    return "A RecordAction to record the current gradient"
end
@doc """
    RecordGradientNorm{R<:Real} <: RecordAction

record the norm of the current gradient

## Constructor
    RecordGradientNorm(r::Type{<:Real}=Float64)
"""
mutable struct RecordGradientNorm{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordGradientNorm(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
function (r::RecordGradientNorm)(
        mp::AbstractManoptProblem, ast::AbstractManoptSolverState, k::Int
    )
    M = get_manifold(mp)
    return record_or_reset!(r, norm(M, get_iterate(ast), get_gradient(ast)), k)
end
show(io::IO, ::RecordGradientNorm) = print(io, "RecordGradientNorm()")
function status_summary(rg::RecordGradientNorm; context::Symbol = :default)
    (context === :short) && return ":GradientNorm"
    return "A RecordAction to record the current gradient norm"
end

@doc """
    RecordStepsize <: RecordAction

record the step size.

## Constructor
    RecordStepsise(r::Type{<:Real}=Float64)
"""
mutable struct RecordStepsize{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordStepsize(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
function (r::RecordStepsize)(p::AbstractManoptProblem, s::AbstractGradientSolverState, k)
    return record_or_reset!(r, get_last_stepsize(p, s, k), k)
end
show(io::IO, ::RecordStepsize{R}) where {R} = print(io, "RecordStepsize($R)")
function status_summary(rg::RecordStepsize{R}; context::Symbol = :default) where {R}
    (context === :short) && return ":Stepsize"
    return "A RecordAction to record the current stepsize (of type $R)"
end
