@doc """
    AbstractEvaluationType

An abstract type to specify the kind of evaluation an [`AbstractManifoldObjective`](@ref) supports.
"""
abstract type AbstractEvaluationType end

@doc """
    AllocatingEvaluation <: AbstractEvaluationType

A parameter indicating that functions work out of place.

They allocate memory for their result. This can refer to an [`AbstractManoptProblem`](@ref),
meaning the functions it contains, as well as to a single `Function`.
"""
struct AllocatingEvaluation <: AbstractEvaluationType end

@doc """
    InplaceEvaluation <: AbstractEvaluationType

A parameter indicating that functions work in place.

They do not allocate memory but work on their input. This can refer to an
[`AbstractManoptProblem`](@ref), meaning the functions it contains, as well as to a single `Function`.
"""
struct InplaceEvaluation <: AbstractEvaluationType end
