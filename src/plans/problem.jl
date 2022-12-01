#
# Define a global problem and ist constructors
#
# ---

@doc raw"""
    AbstractManoptProblem

Describe a Riemannian optimization problem with all static (not-changing) properties.

The most prominent features that should always be stated here are

* the `AbstractManifold` ``\mathcal M`` (cf. [ManifoldsBase.jl#AbstractManifold](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#The-AbstractManifold))
* the cost function ``f\colon \mathcal M → ℝ``

Usually the cost should be within an [`AbstractManifoldObjective`](@ref).
"""
abstract type AbstractManoptProblem end

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

@doc raw"""
    DefaultManoptProblem{TM <: AbstractManifold, Objective <: AbstractManifoldObjective}

Model a default manifold problem, that (just) consists of the domain of optimisatio,
that is an `AbstractManifold` and a [`AbstractManifoldObjective`](@ref)
"""
struct DefaultManoptProblem{TM<:AbstractManifold,Objective<:AbstractManifoldObjective}
    manifold::TM
    objetive::Objective
end

@doc raw"""
    get_manifold(mp::AbstractManoptProblem)

return the manifold stored within an [`AbstractManoptProblem`](@ref)
"""
get_manifold(::AbstractManoptProblem)

get_manifold(mp::DefaultManoptProblem) = mp.manifold

@doc raw"""
    get_objective(mp::AbstractManoptProblem)

return the objective [`AbstractManifoldObjective`](@ref) stored within an [`AbstractManoptProblem`](@ref).
"""
get_objective(::AbstractManoptProblem)

get_objective(mp::DefaultManoptProblem) = mp.objective

@doc raw"""
    get_cost(mp::AbstractManoptProblem, p)

evaluate the cost function `f` stored within the [`AbstractManifoldObjective`](@ref) of an
[`AbstractManoptProblem`](@ref) `mp` at the point `p`.
"""
function get_cost(mp::AbstractManoptProblem, p)
    return get_cost(get_manifold(mp), get_objective(mp), p)
end

@doc raw"""
    get_cost(M::AbstractManifold, obj::AbstractManifoldObjective, p)

evaluate the cost function `f` defined on `M` stored within the [`AbstractManifoldObjective`](@ref) at the point `p`.
"""
get_cost(::AbstractManifold, ::AbstractManifoldObjective, p)
