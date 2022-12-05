
@doc raw"""
    AbstractManifoldObjective{T}

Describe the collection of the optimization function ``f\colon \mathcal M → \bbR` (or even a vectorial range)
and its corresponding elements, which might for example be a gradient or (one or more) prxomial maps.

All these elements should usually be implemented as functions
`(M, p) -> ...`, or `(M, X, p) -> ...` that is

* the first argument of these functions should be the manifold `M` they are defined on
* the argument `X` is present, if the computation is performed inplace of `X` (see [`InplaceEvaluation`](@ref))
* the argument `p` is the place the function (``f`` or one of its elements) is evaluated __at__.

the type `T` indicates the global [`AbstractEvaluationType`](@ref).
"""
abstract type AbstractManifoldObjective{T} end

@doc raw"""
    AbstractEvaluationType

An abstract type to specify the kind of evaluation a [`Problem`](@ref) supports.
"""
abstract type AbstractEvaluationType end

@doc raw"""
    AllocatingEvaluation <: AbstractEvaluationType

A parameter for a [`Problem`](@ref) indicating that the problem uses functions that
allocate memory for their result, i.e. they work out of place.
"""
struct AllocatingEvaluation <: AbstractEvaluationType end

@doc raw"""
    InplaceEvaluation

A parameter for a [`Problem`](@ref) indicating that the problem uses functions that
do not allocate memory but work on their input, i.e. in place.
"""
struct InplaceEvaluation <: AbstractEvaluationType end

"""
    FieldReference{fieldname,TObj}

Reference to field `fieldname` of a mutable struct of type `TObj`.
Supports `getindex` and `setindex!` operations similarly to `Ref`.
"""
struct FieldReference{fieldname,TObj}
    obj::TObj
end

@inline FieldReference(obj, fieldname::Symbol) = FieldReference{fieldname,typeof(obj)}(obj)

@inline function Base.getindex(fa::FieldReference{fieldname}) where {fieldname}
    return getfield(fa.obj, fieldname)
end
@inline function Base.setindex!(fa::FieldReference{fieldname}, v) where {fieldname}
    return setfield!(fa.obj, fieldname, v)
end

@inline function _access_field(obj, fieldname, ::AllocatingEvaluation)
    return FieldReference(obj, fieldname)
end
@inline _access_field(obj, fieldname, ::InplaceEvaluation) = getfield(obj, fieldname)

"""
    @access_field obj.fieldname

A unified access method for allocating and mutating evaluation.
Typical usage: in a function related to a optiomization problem with type variable `Teval`
representing [`AbstractEvaluationType`](@ref) write `@access_field obj.fieldname`
to access field `fieldname` of object `obj` according to problem's evaluation
pattern. Depending that pattern the result is either the object referenced by `obj.fieldname`
or a reference to that field represented by an instance of [`FieldReference`](@ref).

Futher, in a function that uses fields accessed this way, you need to dispatch on
`FieldReference` type to determine which access pattern needs to be used.
"""
macro access_field(ex)
    @assert ex.head === :.
    obj = ex.args[1]
    fieldname = ex.args[2]
    return esc(:(Manopt._access_field($obj, $fieldname, Teval())))
end
