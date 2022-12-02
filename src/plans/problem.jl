#
# Define a global problem and ist constructors
#
# ---

"""
    Problem{T}

Describe the problem that should be optimized by stating all properties, that do not change
during an optimization or that are dependent of a certain solver.

The parameter `T` can be used to distinguish problems with different representations
or implementations.
The default parameter [`AllocatingEvaluation`](@ref), which might be slower but easier to use.
The usually faster parameter value is [`MutatingEvaluation`](@ref)

See [`Options`](@ref) for the changing and solver dependent properties.
"""
abstract type Problem{T} end

"""
    AbstractEvaluationType

An abstract type to specify the kind of evaluation a [`Problem`](@ref) supports.
"""
abstract type AbstractEvaluationType end

"""
    AllocatingEvaluation <: AbstractEvaluationType

A parameter for a [`Problem`](@ref) indicating that the problem uses functions that
allocate memory for their result, i.e. they work out of place.
"""
struct AllocatingEvaluation <: AbstractEvaluationType end

"""
    MutatingEvaluation

A parameter for a [`Problem`](@ref) indicating that the problem uses functions that
do not allocate memory but work on their input, i.e. in place.
"""
struct MutatingEvaluation <: AbstractEvaluationType end

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
@inline _access_field(obj, fieldname, ::MutatingEvaluation) = getfield(obj, fieldname)

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
    return esc(:(_access_field($obj, $fieldname, Teval())))
end

"""
    get_cost(P::Problem, p)

evaluate the cost function `F` stored within a [`Problem`](@ref) `P` at the point `p`.
"""
function get_cost(P::Problem, p)
    return P.cost(P.M, p)
end
