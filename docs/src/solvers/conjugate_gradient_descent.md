
# [Conjugate Gradient Descent](@id CGSolver)

```@meta
CurrentModule = Manopt
```

```@docs
conjugate_gradient_descent
conjugate_gradient_descent!
```

## Options

```@docs
ConjugateGradientDescentOptions
```

## [Available Coefficients](@id cg-coeffs)

The update rules act as [`DirectionUpdateRule`](@ref), which internally always first evaluate the gradient itself.

```@docs
ConjugateDescentCoefficient
DaiYuanCoefficient
FletcherReevesCoefficient
HagerZhangCoefficient
HeestenesStiefelCoefficient
LiuStoreyCoefficient
PolakRibiereCoefficient
SteepestDirectionUpdateRule
```

# Literature
