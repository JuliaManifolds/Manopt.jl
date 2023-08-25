
# [Conjugate Gradient Descent](@id CGSolver)

```@meta
CurrentModule = Manopt
```

```@docs
conjugate_gradient_descent
conjugate_gradient_descent!
```

## State

```@docs
ConjugateGradientDescentState
```

## [Available Coefficients](@id cg-coeffs)

The update rules act as [`DirectionUpdateRule`](@ref), which internally always first evaluate the gradient itself.

```@docs
ConjugateGradientBealeRestart
ConjugateDescentCoefficient
DaiYuanCoefficient
FletcherReevesCoefficient
HagerZhangCoefficient
HestenesStiefelCoefficient
LiuStoreyCoefficient
PolakRibiereCoefficient
SteepestDirectionUpdateRule
```

# Literature

```@bibliography
Pages = ["solvers/conjugate_gradient_descent.md"]
Canonical=false
```