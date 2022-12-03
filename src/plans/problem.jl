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
    DefaultManoptProblem{TM <: AbstractManifold, Objective <: AbstractManifoldObjective}

Model a default manifold problem, that (just) consists of the domain of optimisatio,
that is an `AbstractManifold` and a [`AbstractManifoldObjective`](@ref)
"""
struct DefaultManoptProblem{TM<:AbstractManifold,Objective<:AbstractManifoldObjective} <:
       AbstractManoptProblem
    manifold::TM
    objective::Objective
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
