@doc """
    AbstractEvaluationType

An abstract type to specify the kind of evaluation a [`AbstractManifoldObjective`](@ref) supports.
"""
abstract type AbstractEvaluationType end

@doc """
    AbstractManifoldObjective{E<:AbstractEvaluationType}

Describe the collection of the optimization function ``f: $(_math(:Manifold))nifold))nifold))) → ℝ`` (or even a vectorial range)
and its corresponding elements, which might for example be a gradient or (one or more) proximal maps.

All these elements should usually be implemented as functions
`(M, p) -> ...`, or `(M, X, p) -> ...` that is

* the first argument of these functions should be the manifold `M` they are defined on
* the argument `X` is present, if the computation is performed in-place of `X` (see [`InplaceEvaluation`](@ref))
* the argument `p` is the place the function (``f`` or one of its elements) is evaluated __at__.

the type `T` indicates the global [`AbstractEvaluationType`](@ref).
"""
abstract type AbstractManifoldObjective{E <: AbstractEvaluationType} end

function Base.show(io::IO, ::MIME"text/plain", amo::AbstractManifoldObjective)
    multiline = get(io, :multiline, true)
    if multiline
        return status_summary(io, amo)
    else
        show(io, amo)
    end
end

@doc """
    AbstractDecoratedManifoldObjective{E<:AbstractEvaluationType,O<:AbstractManifoldObjective}

A common supertype for all decorators of [`AbstractManifoldObjective`](@ref)s to simplify dispatch.
    The second parameter should refer to the undecorated objective (the most inner one).
"""
abstract type AbstractDecoratedManifoldObjective{E, O <: AbstractManifoldObjective} <:
AbstractManifoldObjective{E} end

@doc """
    AllocatingEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) allocate memory for their result, they work out of place.
"""
struct AllocatingEvaluation <: AbstractEvaluationType end

@doc """
    InplaceEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) do not allocate memory but work on their input, in place.
"""
struct InplaceEvaluation <: AbstractEvaluationType end

@doc """
    ParentEvaluationType <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) do inherit their property from a parent
[`AbstractManoptProblem`](@ref) or function.
"""
struct ParentEvaluationType <: AbstractEvaluationType end

@doc """
    AllocatingInplaceEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) that provides both an allocating variant and one,
that does not allocate memory but work on their input, in place.
"""
struct AllocatingInplaceEvaluation <: AbstractEvaluationType end

@doc """
    ReturnManifoldObjective{E,O2,O1<:AbstractManifoldObjective{E}} <:
       AbstractDecoratedManifoldObjective{E,O2}

A wrapper to indicate that `get_solver_result` should return the inner objective.

The types are such that one can still dispatch on the undecorated type `O2` of the
original objective as well.
"""
struct ReturnManifoldObjective{E, O2, O1 <: AbstractManifoldObjective{E}} <:
    AbstractDecoratedManifoldObjective{E, O2}
    objective::O1
end
function ReturnManifoldObjective(
        o::O
    ) where {E <: AbstractEvaluationType, O <: AbstractManifoldObjective{E}}
    return ReturnManifoldObjective{E, O, O}(o)
end
function ReturnManifoldObjective(
        o::O1
    ) where {
        E <: AbstractEvaluationType,
        O2 <: AbstractManifoldObjective,
        O1 <: AbstractDecoratedManifoldObjective{E, O2},
    }
    return ReturnManifoldObjective{E, O2, O1}(o)
end

#
# Internal converters if the variable in the high-level interface is a number.
#

function _ensure_mutating_cost(cost, q::Number)
    return isnothing(cost) ? cost : (M, p) -> cost(M, p[])
end
function _ensure_mutating_cost(cost, p)
    return cost
end

function _ensure_mutating_gradient(grad_f, p, evaluation::AbstractEvaluationType)
    return grad_f
end
function _ensure_mutating_gradient(grad_f, q::Number, evaluation::AllocatingEvaluation)
    return isnothing(grad_f) ? grad_f : (M, p) -> [grad_f(M, p[])]
end
function _ensure_mutating_gradient(grad_f, q::Number, evaluation::InplaceEvaluation)
    return isnothing(grad_f) ? grad_f : (M, X, p) -> (X .= [grad_f(M, p[])])
end

function _ensure_mutating_hessian(hess_f, p, evaluation::AbstractEvaluationType)
    return hess_f
end
function _ensure_mutating_hessian(hess_f, q::Number, evaluation::AllocatingEvaluation)
    return isnothing(hess_f) ? hess_f : (M, p, X) -> [hess_f(M, p[], X[])]
end
function _ensure_mutating_hessian(hess_f, q::Number, evaluation::InplaceEvaluation)
    return isnothing(hess_f) ? hess_f : (M, Y, p, X) -> (Y .= [hess_f(M, p[], X[])])
end

function _ensure_mutating_prox(prox_f, p, evaluation::AbstractEvaluationType)
    return prox_f
end
function _ensure_mutating_prox(prox_f, q::Number, evaluation::AllocatingEvaluation)
    return isnothing(prox_f) ? prox_f : (M, λ, p) -> [prox_f(M, λ, p[])]
end
function _ensure_mutating_prox(prox_f, q::Number, evaluation::InplaceEvaluation)
    return isnothing(prox_f) ? prox_f : (M, q, λ, p) -> (q .= [prox_f(M, λ, p[])])
end

_ensure_mutating_variable(p) = p
_ensure_mutating_variable(q::Number) = [q]
_ensure_matching_output(::T, q::Vector{T}) where {T} = length(q) == 1 ? q[] : q
_ensure_matching_output(p, q) = q

"""
    dispatch_objective_decorator(o::AbstractManoptSolverState)

Indicate internally, whether an [`AbstractManifoldObjective`](@ref) `o` to be of decorating type,
it stores (encapsulates) an object in itself, by default in the field `o.objective`.

Decorators indicate this by returning `Val{true}` for further dispatch.

The default is `Val{false}`, so by default an state is not decorated.
"""
dispatch_objective_decorator(::AbstractManifoldObjective) = Val(false)
dispatch_objective_decorator(::AbstractDecoratedManifoldObjective) = Val(true)

"""
    is_object_decorator(s::AbstractManifoldObjective)

Indicate, whether [`AbstractManifoldObjective`](@ref) `s` are of decorator type.
"""
function is_objective_decorator(s::AbstractManifoldObjective)
    return _extract_val(dispatch_objective_decorator(s))
end

@doc """
    get_objective(o::AbstractManifoldObjective, recursive=true)

return the (one step) undecorated [`AbstractManifoldObjective`](@ref) of the (possibly) decorated `o`.
As long as your decorated objective stores the objective within `o.objective` and
the [`dispatch_objective_decorator`](@ref) is set to `Val{true}`,
the internal state are extracted automatically.

By default the objective that is stored within a decorated objective is assumed to be at
`o.objective`. Overwrite `_get_objective(o, ::Val{true}, recursive) to change this behaviour for your objective `o`
for both the recursive and the direct case.

If `recursive` is set to `false`, only the most outer decorator is taken away instead of all.
"""
function get_objective(o::AbstractManifoldObjective, recursive = true)
    return _get_objective(o, dispatch_objective_decorator(o), recursive)
end
_get_objective(o::AbstractManifoldObjective, ::Val{false}, rec = true) = o
function _get_objective(o::AbstractManifoldObjective, ::Val{true}, rec = true)
    return rec ? get_objective(o.objective) : o.objective
end

"""
    set_parameter!(amo::AbstractManifoldObjective, element::Symbol, args...)

Set a certain `args...` from the [`AbstractManifoldObjective`](@ref) `amo` to `value.
This function should dispatch on `Val(element)`.

Currently supported
* `:Cost` passes to the [`get_cost_function`](@ref)
* `:Gradient` passes to the [`get_gradient_function`](@ref)
* `:SubGradient` passes to the [`get_subgradient_function`](@ref)
"""
set_parameter!(amo::AbstractManifoldObjective, e::Symbol, args...)

function set_parameter!(amo::AbstractManifoldObjective, ::Val{:Cost}, args...)
    set_parameter!(get_cost_function(amo), args...)
    return amo
end
function set_parameter!(amo::AbstractManifoldObjective, ::Val{:Gradient}, args...)
    set_parameter!(get_gradient_function(amo), args...)
    return amo
end
function set_parameter!(amo::AbstractManifoldObjective, ::Val{:SubGradient}, args...)
    set_parameter!(get_subgradient_function(amo), args...)
    return amo
end

function show(io::IO, o::AbstractManifoldObjective{E}) where {E}
    return print(io, "$(nameof(typeof(o))){$E}")
end
# Default: remove decorator for show
function show(io::IO, co::AbstractDecoratedManifoldObjective)
    return show(io, get_objective(co, false))
end
function show(io::IO, t::Tuple{<:AbstractManifoldObjective, P}) where {P}
    s = "$(status_summary(t[1]))"
    length(s) > 0 && (s = "$(s)\n\n")
    return print(
        io, "$(s)To access the solver result, call `get_solver_result` on this variable."
    )
end

function status_summary(::AbstractManifoldObjective{E}) where {E}
    return ""
end
# Default: remove decorator for status summary
function status_summary(co::AbstractDecoratedManifoldObjective)
    return status_summary(get_objective(co, false))
end
