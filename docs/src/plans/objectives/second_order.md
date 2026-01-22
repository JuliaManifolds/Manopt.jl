# [Second order objectives](@id second_order_objectives)

```@meta
CurrentModule = Manopt
```

```@docs
AbstractManifoldHessianObjective
ManifoldHessianObjective
```

## Access functions

```@docs
get_hessian
get_preconditioner
```

and internally

```@docs
get_hessian_function
```

## Approximation of the Hessian

Several different methods to approximate the Hessian are available.

```@docs
ApproxHessianFiniteDifference
ApproxHessianSymmetricRankOne
ApproxHessianBFGS
```