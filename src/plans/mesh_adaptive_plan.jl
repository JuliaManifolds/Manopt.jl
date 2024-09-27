"""
    AbstractMeshPollFunctiom

An abstract type for common “poll” strategies in the [`mesh_adaptive_direct_search`](@ref)
solver.
A subtype of this The functor has to fulfill

* be callable as `poll!(problem, state, mesh_size)` and modify the state

as well as

* provide a `get_poll_success(poll!)` function that indicates whether the last poll was successful in finding a new candidate.
"""
abstract type AbstractMeshPollFunctiom end

"""
    MeshAdaptiveState <: AbstractManoptSolverState

For a search step as a functor that is a subtype of this type,
the following is expected

* be callable as `search!(problem, state, mesh_size)` and modify the states iterate
* return a (maybe) new `meshsize` value.
* provide a `get_search_success(search!)` that returns whether the last search was sucressul.
"""
abstract type AbstractMeshSearchFunction end

"""
    MeshAdaptiveState <: AbstractManoptSolverState
"""
mutable struct MeshAdaptiveSearchState{P,PT,ST,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    p::P
    stop::TStop
    poll::PT
    search::ST
end
