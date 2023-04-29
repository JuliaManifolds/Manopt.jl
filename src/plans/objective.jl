@doc raw"""
    AbstractEvaluationType

An abstract type to specify the kind of evaluation a [`AbstractManifoldObjective`](@ref) supports.
"""
abstract type AbstractEvaluationType end

@doc raw"""
    AbstractManifoldObjective{T<:AbstractEvaluationType}

Describe the collection of the optimization function ``f\colon \mathcal M → \bbR` (or even a vectorial range)
and its corresponding elements, which might for example be a gradient or (one or more) prxomial maps.

All these elements should usually be implemented as functions
`(M, p) -> ...`, or `(M, X, p) -> ...` that is

* the first argument of these functions should be the manifold `M` they are defined on
* the argument `X` is present, if the computation is performed inplace of `X` (see [`InplaceEvaluation`](@ref))
* the argument `p` is the place the function (``f`` or one of its elements) is evaluated __at__.

the type `T` indicates the global [`AbstractEvaluationType`](@ref).
"""
abstract type AbstractManifoldObjective{T<:AbstractEvaluationType} end

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

"""
    dispatch_objective_decorator(o::AbstractManoptSolverState)

Indicate internally, whether an [`AbstractManifoldObjective`](@ref) `o` to be of decorating type, i.e.
it stores (encapsulates) an object in itself, by default in the field `o.objective`.

Decorators indicate this by returning `Val{true}` for further dispatch.

The default is `Val{false}`, i.e. by default an state is not decorated.
"""
dispatch_objective_decorator(::AbstractManifoldObjective) = Val(false)

"""
    is_object_decorator(s::AbstractManifoldObjective)

Indicate, whether [`AbstractManifoldObjective`](@ref) `s` are of decorator type.
"""
function is_objective_decorator(s::AbstractManifoldObjective)
    return _extract_val(dispatch_objective_decorator(s))
end

@doc raw"""
    get_objective(o::AbstractManifoldObjective)

return the undecorated [`AbstractManifoldObjective`](@ref) of the (possibly) decorated `o`.
As long as your decorated objective stores the objetive within `o.objective` and
the [`dispatch_objective_decorator`](@ref) is set to `Val{true}`,
the internal state are extracted automatically.
"""
function get_objective(o::AbstractManifoldObjective)
    return _get_objective(o, dispatch_objective_decorator(o))
end
_get_objective(o::AbstractManifoldObjective, ::Val{false}) = o
_get_objective(o::AbstractManifoldObjective, ::Val{true}) = get_objective(o.objective)
