@doc """
    AbstractManifoldObjective

Describe the objective function ``f: $(_math(:Manifold)) → ℝ`` and all its necessary ingredients,
for example when it consists of several summands.

Subtypes might depend on the kind of objective in order to distinguish different available
access functionality, e.g. to a gradient, or a proximal map.

Such a default component of the objective whose result is stored in a mutable variable,
like the gradient, should be implemented in the form

```
(M, v, args...) -> [...]; v
```

where `M` is a $(_link(:AbstractManifold)), `v` is memory the result is computed in,
as well as further arguments, most prominently usually the current iterate `p`.
The cost itself is the exception: it returns a number and is evaluated as an allocating
function `(M, p) -> c`, see [`get_cost_function`](@ref).

For an allocating variant, internally the wrapper [`InplaceManifoldFunction`](@ref) should be used.
"""
abstract type AbstractManifoldObjective end

function Base.show(io::IO, ::MIME"text/plain", amo::AbstractManifoldObjective)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, amo) : show(io, amo)
end

@doc """
    AbstractDecoratedManifoldObjective{O<:AbstractManifoldObjective}

A common supertype for all decorators of [`AbstractManifoldObjective`](@ref)s to simplify dispatch.
The parameter `O` should refer to the undecorated objective, that is, even for multiple decorators
it provides insight into the innermost one.
"""
abstract type AbstractDecoratedManifoldObjective{O <: AbstractManifoldObjective} <: AbstractManifoldObjective end

@doc """
    ReturnManifoldObjective{O2,O1<:AbstractManifoldObjective} <: AbstractDecoratedManifoldObjective{O2}

A wrapper to indicate that [`get_solver_return`](@ref) should return the inner objective.

The types are such that one can still dispatch on the undecorated type `O2` of the
original objective as well.
"""
struct ReturnManifoldObjective{O2, O1 <: AbstractManifoldObjective} <:
    AbstractDecoratedManifoldObjective{O2}
    objective::O1
end
function ReturnManifoldObjective(o::O) where {O <: AbstractManifoldObjective}
    return ReturnManifoldObjective{O, O}(o)
end
function ReturnManifoldObjective(
        o::O1
    ) where {
        O2 <: AbstractManifoldObjective,
        O1 <: AbstractDecoratedManifoldObjective{O2},
    }
    return ReturnManifoldObjective{O2, O1}(o)
end
# The human readable version is “transparent” by default here
function status_summary(o::ReturnManifoldObjective; kwargs...)
    return status_summary(o.objective; kwargs...)
end
function Base.show(io::IO, ro::ReturnManifoldObjective)
    print(io, "ReturnManifoldObjective(")
    show(io, ro.objective)
    return print(io, ")")
end

"""
    dispatch_objective_decorator(o::AbstractManifoldObjective)

Indicate internally whether an [`AbstractManifoldObjective`](@ref) `o` is of decorating type,
that is, whether it stores (encapsulates) another objective in itself, by default in the field `o.objective`.

Decorators indicate this by returning `Val{true}` for further dispatch.

The default is `Val{false}`, so by default an objective is not decorated.
"""
dispatch_objective_decorator(::AbstractManifoldObjective) = Val(false)
dispatch_objective_decorator(::AbstractDecoratedManifoldObjective) = Val(true)

@inline _extract_val(::Val{T}) where {T} = T

"""
    is_objective_decorator(s::AbstractManifoldObjective)

Indicate whether the [`AbstractManifoldObjective`](@ref) `s` is of decorator type.
"""
function is_objective_decorator(s::AbstractManifoldObjective)
    return _extract_val(dispatch_objective_decorator(s))
end

@doc """
    get_objective(o::AbstractManifoldObjective, recursive=true)

Return the undecorated [`AbstractManifoldObjective`](@ref) of the (possibly) decorated `o`.
As long as your decorated objective stores the objective within `o.objective` and
[`dispatch_objective_decorator`](@ref) is set to `Val{true}`,
the internal objective is extracted automatically.

By default the objective that is stored within a decorated objective is assumed to be at
`o.objective`. Overwrite `_get_objective(o, ::Val{true}, recursive)` to change this behavior for your objective `o`
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

Set a certain `element` from the [`AbstractManifoldObjective`](@ref) `amo` to a value specified
by `args...`
This function should dispatch on `Val(element)`.
"""
set_parameter!(amo::AbstractManifoldObjective, e::Symbol, args...)


# For decorators the human readable version is “transparent” by default, i.e.
# if no special addition is done, it just prints the human readable string from the child
function status_summary(io::IO, co::AbstractDecoratedManifoldObjective; context::Symbol = :default)
    return status_summary(io, get_objective(co, false); context = context)
end
function status_summary(co::AbstractDecoratedManifoldObjective; context::Symbol = :default)
    return status_summary(get_objective(co, false); context = context)
end
