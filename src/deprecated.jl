Base.@deprecate_binding AbstractManifoldGradientObjective AbstractManifoldFirstOrderObjective
# the start point is a keyword since 0.6.8, as for every other state
Base.@deprecate ProjectedGradientMethodState(M::AbstractManifold, p; kwargs...) ProjectedGradientMethodState(M; p = p, kwargs...)
Base.@deprecate MeshAdaptiveDirectSearchState(M::AbstractManifold, p; kwargs...) MeshAdaptiveDirectSearchState(M; p = p, kwargs...)
Base.@deprecate CoordinatesNormalSystemState(M::AbstractManifold, p; kwargs...) CoordinatesNormalSystemState(M; p = p, kwargs...)
Base.@deprecate_binding AbstractSubProblemSolverState AbstractManoptSolverState
