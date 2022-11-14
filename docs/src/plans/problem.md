# [Problems](@id ProblemSection)

```@meta
CurrentModule = Manopt
```

A problem usually contains its cost function and provides an
implementation to access the cost

```@docs
Problem
get_cost
```

A problem can be of different type, more specifically, whether its containing functions,
for example to compute the gradient, work with allocation or without it. To be precise, an
allocation function `X = gradF(x)` allocates memory for its result `X`, while `gradF!(X,x)` does not.

```@docs
AbstractEvaluationType
AllocatingEvaluation
MutatingEvaluation
```

## Cost based problem

```@docs
CostProblem
```

## Gradient based problem

```@docs
AbstractGradientProblem
GradientProblem
StochasticGradientProblem
get_gradient
get_gradients
```

## Subgradient based problem

```@docs
SubGradientProblem
get_subgradient
```

## [Proximal Map(s) based problem](@id ProximalProblem)

```@docs
ProximalProblem
get_proximal_map
```

## [Hessian based problem](@id HessianProblem)

```@docs
HessianProblem
get_hessian
get_preconditioner
```

## [Primal dual based problem](@id PrimalDualProblem)

```@docs
AbstractPrimalDualProblem
PrimalDualProblem
PrimalDualSemismoothNewtonProblem
get_primal_prox
get_dual_prox
forward_operator
linearized_forward_operator
adjoint_linearized_operator
get_differential_primal_prox
get_differential_dual_prox
```
