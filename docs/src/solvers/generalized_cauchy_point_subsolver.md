# Generalized Cauchy Point subsolver

The Generalized Cauchy Point (GCP) subsolver is a component in optimization algorithms that handle problems with bound constraints. It solves the following problem

```math
\begin{align*}
\operatorname*{arg\,min}_{Y  ∈  T_p D \times \mathcal{M}}&\ m_p(Y) = f(p) +
⟨\operatorname{grad}f(p), Y⟩_p + \frac{1}{2} ⟨\mathcal{H}_p[Y], Y⟩_p\\
\text{such that}& \ \exp_p(Y) \in D \times \mathcal{M}
\end{align*}
```

where $D$ is a box domain ([`Hyperrectangle`](@extref Manifolds.Hyperrectangle)), $\mathcal{M}$ is a Riemannian manifold, and $\mathcal{H}_p$ is a Hessian-like linear operator at point $p$.

The solver is currently primarily intended for internal use by optimization algorithms that require bound-constrained subproblem solutions.

## Internal types and method

### Symbols related to the GCP computation

These symbols are directly used by solvers to compute the descent direction corresponding to the Generalized Cauchy point.

```@docs
Manopt.requires_generalized_cauchy_point_computation
Manopt.find_generalized_cauchy_point_direction!
Manopt.GeneralizedCauchyPointFinder
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
Manopt.AbstractFPFPPUpdater
Manopt.GenericFPFPPUpdater
Manopt.get_bounds_index
Manopt.get_bound_t
Manopt.bound_direction_tweak!
Manopt.set_bound_t_at_index!
Manopt.get_at_bound_index
Manopt.set_bound_at_index!
```

### Symbols related to specific Hessian approximations

```@docs
Manopt.LimitedMemoryFPFPPUpdater
Manopt.hessian_value_from_wmwt_coords
Manopt.set_M_current_scale!
```
