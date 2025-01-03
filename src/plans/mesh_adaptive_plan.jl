"""
    AbstractMeshPollFunction

An abstract type for common “poll” strategies in the [`mesh_adaptive_direct_search`](@ref)
solver.
A subtype of this The functor has to fulllfil

* be callable as `poll!(problem, mesh_size; kwargs...)` and modify the state

as well as

* provide a `get_poll_success(poll!)` function that indicates whether the last poll was successful in finding a new candidate,
this returns the last successful mesh vector used.

The `kwargs...` could include
* `scale_mesh=1.0`: to rescale the mesh globally
* `max_stepsize=Inf`: avoid exceeding a step size beyon this, e.g. injectivity radius.
  any vector longer than this should be shortened to the provided max stepsize.
"""
abstract type AbstractMeshPollFunction end

"""
    AbstractMeshSearchFunction

Should be callable as search!(problem, mesh_size, X; kwargs...)

where `X` is the last succesful poll direction, if that exists and the zero vector otherwise.
"""
abstract type AbstractMeshSearchFunction end

"""
    MeshAdaptiveDirectSearchState <: AbstractManoptSolverState

* `p`: current iterate
* `q`: temp (old) iterate

"""
mutable struct MeshAdaptiveDirectSearchState{P,F<:Real,M,PT,ST,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    p::P
    q::P
    mesh_size::F
    scale_mesh::F
    max_stepsize::F
    poll_size::F
    stop::TStop
    poll::PT
    search::ST
end

# TODO: Stopping critertion based on poll_size
# TODO: Show for state
