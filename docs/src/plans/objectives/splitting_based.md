# [Splitting based objectives](@id splitting_based_objectives)

```@meta
CurrentModule = Manopt
```

## Difference of convex objective

```@docs
ManifoldDifferenceOfConvexObjective
```

## Proximal gradient objective

```@docs
ManifoldProximalGradientObjective
```

## Primal-dual based objectives

```@docs
AbstractPrimalDualManifoldObjective
PrimalDualManifoldObjective
PrimalDualManifoldSemismoothNewtonObjective
```

### Access functions

```@docs
adjoint_linearized_operator
forward_operator
get_differential_dual_prox
get_differential_primal_prox
get_dual_prox
get_primal_prox
linearized_forward_operator
```