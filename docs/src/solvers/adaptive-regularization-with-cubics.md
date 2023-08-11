# [Adaptive regularization with Cubics](@id ARSSection)



```@meta
CurrentModule = Manopt
```

```@docs
adaptive_regularization_with_cubics
adaptive_regularization_with_cubics!
```

## State

```@docs
AdaptiveRegularizationState
```

## Sub solvers

There are several ways to approach the subsolver. The default is the first one.

## Lanczos Iteration

```@docs
Manopt.LanczosState
```

## (Conjugate) Gradient Descent

There are two generic functors, that implement the sub problem

```@docs
AdaptiveRegularizationCubicCost
AdaptiveRegularizationCubicGrad
```

Since the sub problem is given on the tangent space, you have to provide

```
g = AdaptiveRegularizationCubicCost(M, mho, σ)
grad_g = AdaptiveRegularizationCubicGrad(M, mho, σ)
sub_problem = DefaultProblem(TangentSpaceAt(M,p), ManifoldGradienObjective(g, grad_g))
```

where `mho` is the hessian objective of `f` to solve.
Then use this for the `sub_problem` keyword
and use your favourite gradient based solver for the `sub_state` keyword, for example a
[`ConjugateGradientDescentState`](@ref)

## Additional Stopping Criteria

```@docs
StopWhenAllLanczosVectorsUsed
StopWhenFirstOrderProgress
```

## Literature

```@bibliography
Pages = ["solvers/adaptive-regularization-with-cubics.md"]
Canonical=false
```