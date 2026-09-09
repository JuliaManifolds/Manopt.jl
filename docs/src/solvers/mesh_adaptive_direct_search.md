# Mesh adaptive direct search (MADS)

```@meta
CurrentModule = Manopt
```

```@docs
    mesh_adaptive_direct_search
    mesh_adaptive_direct_search!
```

## State

```@docs
    MeshAdaptiveDirectSearchState
```

## Poll

```@docs
    AbstractMeshPollFunction
    LowerTriangularAdaptivePoll
```

as well as the internal functions

```@docs
Manopt.get_descent_direction(::LowerTriangularAdaptivePoll)
Manopt.is_successful(::LowerTriangularAdaptivePoll)
Manopt.get_candidate(::LowerTriangularAdaptivePoll)
Manopt.get_basepoint(::LowerTriangularAdaptivePoll)
Manopt.update_basepoint!(M, ltap::LowerTriangularAdaptivePoll{P}, p::P) where {P}
```

## Search

```@docs
    AbstractMeshSearchFunction
    DefaultMeshAdaptiveDirectSearch
```

as well as the internal functions

```@docs
Manopt.is_successful(::DefaultMeshAdaptiveDirectSearch)
Manopt.get_candidate(::DefaultMeshAdaptiveDirectSearch)
```

## Additional stopping criteria

```@docs
    StopWhenPollSizeLess
```

## Technical details

The [`mesh_adaptive_direct_search`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favorite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* Within the default initialization [`rand`](@extref Base.rand-Tuple{AbstractManifold})`(M)` is used to generate the initial point.
* A [`vector_transport_to!`](@extref ManifoldsBase :doc:`vector_transports`)`(M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) to a favorite vector transport. If this default is set, a `vector_transport_method=` does not have to be specified.
* A [`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`)`(M)` and [`get_vector!`](@extref `ManifoldsBase.get_vector`)`(M, X, p, c, b)` with respect to the [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `b` provided, which is [`default_basis`](@extref `ManifoldsBase.default_basis-Union{Tuple{T}, Tuple{AbstractManifold, Type{T}}} where T`)`(M, typeof(p))` by default from the `mesh_basis=` keyword, since the mesh is stored in coordinates.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points, and [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)` for the tangent vector buffer.

## Literature

```@bibliography
Pages = ["mesh_adaptive_direct_search.md"]
Canonical=false
```