# Generalized Cauchy direction subsolver

The generalized Cauchy direction (GCD) subsolver is a component in optimization algorithms that handle problems with bound constraints. It solves the following problem

```math
\begin{align*}
\operatorname*{arg\,min}_{Y  ∈  T_p D \times \mathcal{M}}&\ m_p(Y), \qquad m_p(Y) = ⟨X_g, Y⟩_p + \frac{1}{2} ⟨\mathcal{H}_p[Y], Y⟩_p\\
\text{such that}& \ \exp_p(Y) = \exp_p(\alpha X) \in D \times \mathcal{M} \text{ for some } \alpha \in [0, A]
\end{align*}
```

where $X=(X_{\mathrm{D}}, X_{\mathcal{M}})$ is a given direction, the exponential map handles projection of the tangent vector when reaching the boundary, $D$ is a box domain ([`Hyperrectangle`](@extref Manifolds.Hyperrectangle)), $\mathcal{M}$ is a Riemannian manifold, $X_g$ is the gradient of a scalar function $f$ at point $p=(p_{\mathrm{D}}, p_{\mathcal{M}})$, $A$ is the maximum allowed step size on $\mathcal{M}$ at point $p=(p_{\mathrm{D}}, p_{\mathcal{M}})$ in direction $X_{\mathcal{M}}$ (infinity is supported) and $\mathcal{H}_p$ is a linear operator that approximates the Hessian of $f$ at $p$.

Additionally, the subsolver indicates whether the selected direction $Y$ reaches the boundary of $D$ at some point, in which case the subsequent step size selection in direction $Y$ needs to be limited to the interval $[0, s_{\max}]$, where the number $1 ≤ s_{\max} ≤ ∞$ is also returned by the subsolver.
Note that the value $s_{\max}=1$ is obtained when the minimum lies at the boundary of $D$, while larger values indicate that we are further away from the boundary along the selected direction $X$.

The solver is currently primarily intended for internal use by optimization algorithms that require bound-constrained subproblem solutions.

## Simple stepsize limiting

In case there is no Hessian approximation available, a simple stepsize limiting procedure is can be used to limit the stepsize in direction $X$ to the maximum allowed by the boundary of $D$ and the maximum allowed stepsize on $\mathcal{M}$.
This procedure is available using the following:

```@docs
Manopt.MaxStepsizeInDirectionSubsolver
Manopt.find_max_stepsize_in_direction
```

## Internal types and method

### Symbols related to the GCD computation

These symbols are directly used by solvers to compute the descent direction corresponding to the Generalized Cauchy direction.

```@docs
Manopt.has_anisotropic_max_stepsize
Manopt.find_generalized_cauchy_direction!
Manopt.GeneralizedCauchyDirectionSubsolver
```

### Symbols related to the Hessian approximation

These symbols are used to evaluate the Hessian approximation at specific tangent vectors during the generalized Cauchy direction computation.

```@docs
Manopt.hessian_value
Manopt.hessian_value_diag
```

### Symbols related to bound handling

These are internal symbols used to manage and manipulate bound constraints during the GCP computation.

```@docs
Manopt.init_updater!
Manopt.UnitVector
Manopt.to_coordinate_index
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
