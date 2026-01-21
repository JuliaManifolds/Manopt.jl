# Generalized Cauchy Direction subsolver

The Generalized Cauchy Direction (GCD) subsolver is a component in optimization algorithms that handle problems with bound constraints. It solves the following problem

```math
\begin{align*}
\operatorname*{arg\,min}_{Y  ∈  T_p D \times \mathcal{M}}&\ m_p(Y), \qquad m_p(Y) = ⟨X_g, Y⟩_p + \frac{1}{2} ⟨\mathcal{H}_p[Y], Y⟩_p\\
\text{such that}& \ \exp_p(Y) = \exp_p(\alpha X) \in D \times \mathcal{M} \text{ for some } \alpha \in \mathbb{R}
\end{align*}
```

where $X$ is a given direction, the exponential map handles projection of the tangent vector when reaching the boundary, $D$ is a box domain ([`Hyperrectangle`](@extref Manifolds.Hyperrectangle)), $\mathcal{M}$ is a Riemannian manifold, $X_g$ is the gradient of a scalar function $f$ at point $p$ and $\mathcal{H}_p$ is a linear operator that approximates the Hessian of $f$ at $p$.

The solver is currently primarily intended for internal use by optimization algorithms that require bound-constrained subproblem solutions.

## Internal types and method

### Symbols related to the GCP computation

These symbols are directly used by solvers to compute the descent direction corresponding to the Generalized Cauchy point.

```@docs
Manopt.requires_generalized_cauchy_direction_computation
Manopt.find_generalized_cauchy_point_direction!
Manopt.GeneralizedCauchyDirectionFinder
```

### Symbols related to the Hessian approximation

These symbols are used to evaluate the Hessian approximation at specific tangent vectors during the generalized Cauchy point computation.

```@docs
Manopt.hessian_value
Manopt.hessian_value_eb
```

### Symbols related to bound handling

These are internal symbols used to manage and manipulate bound constraints during the GCP computation.

```@docs
Manopt.init_updater!
Manopt.AbstractSegmentHessianUpdater
Manopt.GenericSegmentHessianUpdater
Manopt.get_bounds_index
Manopt.get_stepsize_bound
Manopt.get_at_bound_index
Manopt.set_stepsize_bound!
Manopt.set_zero_at_index!
```

### Symbols related to specific Hessian approximations

```@docs
Manopt.LimitedMemorySegmentHessianUpdater
Manopt.hessian_value_from_inner_products
Manopt.set_M_current_scale!
```
