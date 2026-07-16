@doc """
    AbstractEvaluationType

An abstract type to specify the kind of evaluation a [`AbstractManifoldObjective`](@ref) supports.
"""
abstract type AbstractEvaluationType end

@doc """
    AllocatingEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) allocate memory for their result, they work out of place.
"""
struct AllocatingEvaluation <: AbstractEvaluationType end
_to_kw(::Type{AllocatingEvaluation}) = "evaluation = AllocatingEvaluation()"

@doc """
    InplaceEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) do not allocate memory but work on their input, in place.
"""
struct InplaceEvaluation <: AbstractEvaluationType end
_to_kw(::Type{InplaceEvaluation}) = "evaluation = InplaceEvaluation()"

@doc """
    ParentEvaluationType <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) do inherit their property from a parent
[`AbstractManoptProblem`](@ref) or function.
"""
struct ParentEvaluationType <: AbstractEvaluationType end
_to_kw(::Type{ParentEvaluationType}) = "evaluation = ParentEvaluationType()"

@doc """
    AllocatingInplaceEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) that provides both an allocating variant and one,
that does not allocate memory but work on their input, in place.
"""
struct AllocatingInplaceEvaluation <: AbstractEvaluationType end
_to_kw(::Type{AllocatingInplaceEvaluation}) = "evaluation = AllocatingInplaceEvaluation()"
