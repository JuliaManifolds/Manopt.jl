#
# Define a global problem and its constructors
#
# ---

@doc """
    AbstractManoptProblem{M<:AbstractManifold}

Describe a Riemannian optimization problem with all static (not-changing) properties.

The most prominent features that should always be stated here are

* the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) ``$(_math(:Manifold))nifold)))``
* the cost function ``f:  $(_math(:Manifold))) → ℝ``

Usually the cost should be within an [`AbstractManifoldObjective`](@ref).
"""
abstract type AbstractManoptProblem{M <: AbstractManifold} end

function Base.show(io::IO, ::MIME"text/plain", amp::AbstractManoptProblem)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, amp) : show(io, amp)
end


@doc """
    DefaultManoptProblem{TM <: AbstractManifold, Objective <: AbstractManifoldObjective}

Model a default manifold problem, that (just) consists of the domain of optimisation,
that is an `AbstractManifold` and an [`AbstractManifoldObjective`](@ref)
"""
struct DefaultManoptProblem{TM <: AbstractManifold, O <: AbstractManifoldObjective} <:
    AbstractManoptProblem{TM}
    manifold::TM
    objective::O
end

"""
    evaluation_type(mp::AbstractManoptProblem)

Get the [`AbstractEvaluationType`](@ref) of the objective in [`AbstractManoptProblem`](@ref)
`mp`.
"""
evaluation_type(amp::AbstractManoptProblem) = evaluation_type(get_objective(amp))
"""
    evaluation_type(::AbstractManifoldObjective{Teval})

Get the [`AbstractEvaluationType`](@ref) of the objective.
"""
evaluation_type(::AbstractManifoldObjective{Teval}) where {Teval} = Teval

@doc """
    get_manifold(amp::AbstractManoptProblem)

return the manifold stored within an [`AbstractManoptProblem`](@ref)
"""
get_manifold(::AbstractManoptProblem)

get_manifold(amp::DefaultManoptProblem) = amp.manifold

@doc """
    get_objective(mp::AbstractManoptProblem, recursive=false)

return the objective [`AbstractManifoldObjective`](@ref) stored within an [`AbstractManoptProblem`](@ref).
If `recursive is set to true, it additionally unwraps all decorators of the objective`
"""
get_objective(::AbstractManoptProblem)

function get_objective(amp::DefaultManoptProblem, recursive = false)
    return recursive ? get_objective(amp.objective, true) : amp.objective
end

@doc """
    get_cost(amp::AbstractManoptProblem, p)

evaluate the cost function `f` stored within the [`AbstractManifoldObjective`](@ref) of an
[`AbstractManoptProblem`](@ref) `amp` at the point `p`.
"""
function get_cost(amp::AbstractManoptProblem, p)
    return get_cost(get_manifold(amp), get_objective(amp), p)
end

@doc """
    get_cost(M::AbstractManifold, obj::AbstractManifoldObjective, p)

evaluate the cost function `f` defined on `M` stored within the [`AbstractManifoldObjective`](@ref) at the point `p`.
"""
get_cost(::AbstractManifold, ::AbstractManifoldObjective, p)

"""
    set_parameter!(ams::AbstractManoptProblem, element::Symbol, field::Symbol , value)

Set a certain field/element from the [`AbstractManoptProblem`](@ref) `ams` to `value`.
This function usually dispatches on `Val(element)`.
Instead of a single field, also a chain of elements can be provided, allowing to access
encapsulated parts of the problem.

Main values for `element` are `:Manifold` and `:Objective`.
"""
set_parameter!(amp::AbstractManoptProblem, e::Symbol, args...)

function set_parameter!(amp::AbstractManoptProblem, ::Val{:Manifold}, args...)
    set_parameter!(get_manifold(amp), args...)
    return amp
end
function set_parameter!(TpM::TangentSpace, ::Union{Val{:Basepoint}, Val{:p}}, p)
    copyto!(TpM.manifold, TpM.point, p)
    return TpM
end
function set_parameter!(amp::AbstractManoptProblem, ::Val{:Objective}, args...)
    set_parameter!(get_objective(amp), args...)
    return amp
end
