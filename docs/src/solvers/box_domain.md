# Optimization with box domains and products of manifolds and boxes

A [`Hyperrectangle`](@extref Manifolds.Hyperrectangle) is, in general, not a manifold but a manifold with corners, thus handling it as a domain in optimization requires special attention.
For simple methods like gradient descent using projected gradient and a stopping criterion involving [`StopWhenProjectedNegativeGradientNormLess`](@ref) may be sufficient, however methods that approximate the Hessian can benefit from a more advanced approach.
The core idea is considering a piecewise quadratic approximation of the objective along the descent direction, and selecting the generalized Cauchy point -- its minimizer.
The points at which the approximation might not be differentiable correspond to hitting new boundaries along the initially selected descent direction.
Then, we can perform standard line search between the initial iterate and the generalized Cauchy point.

## Public types and method

```@docs
QuasiNewtonLimitedMemoryBoxDirectionUpdate
```

## Internal types and method

```@docs
Manopt.init_updater!
Manopt.hess_val
Manopt.AbstractFPFPPUpdater
Manopt.GenericFPFPPUpdater
Manopt.get_bounds_index
Manopt.requires_gcp
Manopt.find_gcp_direction!
Manopt.hess_val_eb
Manopt.LimitedMemoryFPFPPUpdater
Manopt.get_bound_t
```
