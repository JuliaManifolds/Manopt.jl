# Covariance matrix adaptation evolutionary strategy

```@meta
CurrentModule = Manopt
```

The CMA-ES algorithm has been implemented based on [Hansen:2023](@cite) paper with basic Riemannian adaptations, related to transport of covariance matrix and its update vectors. Other attempts at adapting CMA-ES to Riemannian optimzation include [ColuttoFruhaufFuchsScherzer:2010](@cite).

Covariance matrix transport between consecutive mean points is handled by `eigenvector_transport!` function which is based on the idea of transport of matrix eigenvectors.

```@docs
cma_es
Manopt.eigenvector_transport!
```

## State

```@docs
CMAESState
```

## Stopping Criteria

```@docs
CMAESConditionCov
StopWhenBestCostInGenerationConstant
StopWhenEvolutionStagnates
StopWhenPopulationCostConcentrated
StopWhenPopulationDiverges
StopWhenPopulationStuckConcentrated
```

## Literature

```@bibliography
Pages = ["cma_es.md"]
Canonical=false
```
