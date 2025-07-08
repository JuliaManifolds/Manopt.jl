# Covariance matrix adaptation evolutionary strategy

```@meta
CurrentModule = Manopt
```

The CMA-ES algorithm has been implemented based on [Hansen:2023](@cite) with basic Riemannian adaptations, related to transport of covariance matrix and its update vectors. Other attempts at adapting CMA-ES to Riemannian optimization include [ColuttoFruhaufFuchsScherzer:2010](@cite).
The algorithm is suitable for global optimization.

Covariance matrix transport between consecutive mean points is handled by `eigenvector_transport!` function which is based on the idea of transport of matrix eigenvectors.

```@docs
cma_es
```

## State

```@docs
CMAESState
```

## Stopping criteria

```@docs
StopWhenBestCostInGenerationConstant
StopWhenCovarianceIllConditioned
StopWhenEvolutionStagnates
StopWhenPopulationCostConcentrated
StopWhenPopulationDiverges
StopWhenPopulationStronglyConcentrated
```

## [Technical details](@id sec-cma-es-technical-details)

The [`cma_es`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* A [`vector_transport_to!`](@extref ManifoldsBase :doc:`vector_transports`)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `vector_transport_method=` does not have to be specified.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points and similarly `copy(M, p, X)` for tangent vectors.
* [`get_coordinates!`](@extref `ManifoldsBase.get_coordinates`)`(M, Y, p, X, b)` and [`get_vector!`](@extref `ManifoldsBase.get_vector`)`(M, X, p, c, b)` with respect to the [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `b` provided, which is [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) by default from the `basis=` keyword.
* An [`is_flat`](@extref `ManifoldsBase.is_flat-Tuple{AbstractManifold}`)`(M)`.

## Internal helpers

You may add new methods to `eigenvector_transport!` if you know a more optimized implementation
for your manifold.

```@docs
Manopt.eigenvector_transport!
```

## Literature

```@bibliography
Pages = ["cma_es.md"]
Canonical=false
```
