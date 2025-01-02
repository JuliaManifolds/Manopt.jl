"""
    AbstractMeshPollFunction

An abstract type for common “poll” strategies in the [`mesh_adaptive_direct_search`](@ref)
solver.
A subtype of this The functor has to fullfil

* be callable as `poll!(problem, mesh_size)` and modify the state

as well as

* provide a `get_poll_success(poll!)` function that indicates whether the last poll was successful in finding a new candidate,
this returns the last sucessful mesh vector used.
"""
abstract type AbstractMeshPollFunctiom end

"""
    AbstractMeshSearchFunction

Should be callable as search!(problem, mesh_size)
"""
abstract type AbstractMeshSearchFunction end

"""
    MeshAdaptiveState <: AbstractManoptSolverState


"""
mutable struct MeshAdaptiveSearchState{P,F<:Real,M,PT,ST,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    p::P
    mesh_size::F
    poll_size::F
    stop::TStop
    poll::PT
    search::ST
end

# TODO: Stopping critertion based on poll_size
# TODO: Show for state