# [Difference of convex](@id DifferenceOfConvexSolvers)

```@meta
CurrentModule = Manopt
```

## [Difference of convex algorithm](@id DCASolver)

```@docs
difference_of_convex_algorithm
difference_of_convex_algorithm!
```

## [Difference of convex proximal point](@id DCPPASolver)

```@docs
difference_of_convex_proximal_point
difference_of_convex_proximal_point!
```

## Solver states

```@docs
DifferenceOfConvexState
DifferenceOfConvexProximalState
```

## The difference of convex objective

```@docs
ManifoldDifferenceOfConvexObjective
```

as well as for the corresponding sub problem

```@docs
LinearizedDCCost
LinearizedDCGrad
```

```@docs
ManifoldDifferenceOfConvexProximalObjective
```

as well as for the corresponding sub problems

```@docs
ProximalDCCost
ProximalDCGrad
```

## Helper functions

```@docs
get_subtrahend_gradient
```

## Literature

```@bibliography
Pages = ["difference_of_convex.md"]
Canonical=false
```