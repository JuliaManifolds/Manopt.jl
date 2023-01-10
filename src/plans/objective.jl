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
