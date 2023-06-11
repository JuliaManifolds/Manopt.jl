@doc raw"""
    AbstractEvaluationType

An abstract type to specify the kind of evaluation a [`AbstractManifoldObjective`](@ref) supports.
"""
abstract type AbstractEvaluationType end

@doc raw"""
    AbstractManifoldObjective{E<:AbstractEvaluationType}

Describe the collection of the optimization function ``f\colon \mathcal M → \bbR` (or even a vectorial range)
and its corresponding elements, which might for example be a gradient or (one or more) prxomial maps.

All these elements should usually be implemented as functions
`(M, p) -> ...`, or `(M, X, p) -> ...` that is

* the first argument of these functions should be the manifold `M` they are defined on
* the argument `X` is present, if the computation is performed inplace of `X` (see [`InplaceEvaluation`](@ref))
* the argument `p` is the place the function (``f`` or one of its elements) is evaluated __at__.

the type `T` indicates the global [`AbstractEvaluationType`](@ref).
"""
abstract type AbstractManifoldObjective{E<:AbstractEvaluationType} end

@doc raw"""
    AbstractDecoratedManifoldObjective{E<:AbstractEvaluationType,O<:AbstractManifoldObjective}

A common supertype for all decorators of [`AbstractManifoldObjective`](@ref)s to simplify dispatch.
    The second parameter should refer to the undecorated objective (i.e. the most inner one).
"""
abstract type AbstractDecoratedManifoldObjective{E,O<:AbstractManifoldObjective} <:
              AbstractManifoldObjective{E} end

@doc raw"""
    AllocatingEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) indicating that the problem uses functions that
allocate memory for their result, i.e. they work out of place.
"""
struct AllocatingEvaluation <: AbstractEvaluationType end

@doc raw"""
    InplaceEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) indicating that the problem uses functions that
do not allocate memory but work on their input, i.e. in place.
"""
struct InplaceEvaluation <: AbstractEvaluationType end

struct ReturnManifoldObjective{E,P,O<:AbstractManifoldObjective{E}} <:
       AbstractDecoratedManifoldObjective{E,P}
    objective::O
end
function ReturnManifoldObjective(
    o::O
) where {E<:AbstractEvaluationType,O<:AbstractManifoldObjective{E}}
    return ReturnManifoldObjective{E,O,O}(o)
end
function ReturnManifoldObjective(
    o::O
) where {
    E<:AbstractEvaluationType,
    P<:AbstractManifoldObjective,
    O<:AbstractDecoratedManifoldObjective{E,P},
}
    return ReturnManifoldObjective{E,P,O}(o)
end

"""
    dispatch_objective_decorator(o::AbstractManoptSolverState)

Indicate internally, whether an [`AbstractManifoldObjective`](@ref) `o` to be of decorating type, i.e.
it stores (encapsulates) an object in itself, by default in the field `o.objective`.

Decorators indicate this by returning `Val{true}` for further dispatch.

The default is `Val{false}`, i.e. by default an state is not decorated.
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

@doc raw"""
    get_objective(o::AbstractManifoldObjective, recursive=true)

return the (one step) undecorated [`AbstractManifoldObjective`](@ref) of the (possibly) decorated `o`.
As long as your decorated objective stores the objetive within `o.objective` and
the [`dispatch_objective_decorator`](@ref) is set to `Val{true}`,
the internal state are extracted automatically.

By default the objective that is stored within a decorated objective is assumed to be at
`o.objective`. Overwrtie `_get_objective(o, ::Val{true}, recursive) to change this bevahiour for your objective `o`
for both the recursive and the nonrecursive case.

If `recursive` is set to `false`, only the most outer decorator is taken away instead of all.
"""
function get_objective(o::AbstractManifoldObjective, recursive=true)
    return _get_objective(o, dispatch_objective_decorator(o), recursive)
end
_get_objective(o::AbstractManifoldObjective, ::Val{false}, rec=true) = o
function _get_objective(o::AbstractManifoldObjective, ::Val{true}, rec=true)
    return rec ? get_objective(o.objective) : o.objective
end
function status_summary(o::AbstractManifoldObjective{E}) where {E}
    return "$(nameof(typeof(o))){$E}"
end
# Default undecorate for summary
function status_summary(co::AbstractDecoratedManifoldObjective)
    return status_summary(get_objective(co, false))
end

function show(io::IO, o::AbstractManifoldObjective{E}) where {E}
    return print(io, "$(nameof(typeof(o))){$E}")
end
# Default undecorate for show
function show(io::IO, co::AbstractDecoratedManifoldObjective)
    return show(io, get_objective(co, false))
end

function show(io::IO, t::Tuple{<:AbstractManifoldObjective,P}) where {P}
    return print(
        io,
        """
$(status_summary(t[1]))

To access the solver result, call `get_solver_result` on this variable.""",
    )
end
