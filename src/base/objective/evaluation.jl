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

@doc """
    InplaceEvaluation <: AbstractEvaluationType

A parameter for a [`AbstractManoptProblem`](@ref) or a `Function` indicating that
the problem contains or the function(s) do not allocate memory but work on their input, in place.
"""
struct InplaceEvaluation <: AbstractEvaluationType end
