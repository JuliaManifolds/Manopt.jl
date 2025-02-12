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

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* Within the default initialization [`rand`](@extref Base.rand-Tuple{AbstractManifold})`(M)` is used to generate the initial population
* A [`vector_transport_to!`](@extref ManifoldsBase :doc:`vector_transports`)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `vector_transport_method=` does not have to be specified.

## Literature

```@bibliography
Pages = ["mesh_adaptive_direct_search.md"]
Canonical=false
```