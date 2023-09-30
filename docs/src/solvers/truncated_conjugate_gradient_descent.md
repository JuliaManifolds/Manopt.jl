# [Steihaug-Toint Truncated Conjugate-Gradient Method](@id tCG)

Solve the constraint optimization problem on the tangent space

```math
\begin{align*}
\operatorname*{arg\,min}_{Y  ∈  T_p\mathcal{M}}&\ m_p(Y) = f(p) +
⟨\operatorname{grad}f(p), Y⟩_p + \frac{1}{2} ⟨\mathcal{H}_p[Y], Y⟩_p\\
\text{such that}& \ \lVert Y \rVert_p ≤ Δ
\end{align*}
```

on the tangent space ``T_p\mathcal M`` of a Riemannian manifold ``\mathcal M`` by using
the Steihaug-Toint truncated conjugate-gradient (tCG) method,
see [AbsilBakerGallivan:2006](@cite), Algorithm 2, and [ConnGouldToint:2000](@cite).
Here ``\mathcal H_p`` is either the Hessian ``\operatorname{Hess} f(p)`` or a linear symmetric operator on the tangent space approximating the Hessian.

## Interface

```@docs
  truncated_conjugate_gradient_descent
  truncated_conjugate_gradient_descent!
```

## State

```@docs
TruncatedConjugateGradientState
```

## Stopping Criteria

```@docs
StopWhenResidualIsReducedByFactorOrPower
StopWhenTrustRegionIsExceeded
StopWhenCurvatureIsNegative
StopWhenModelIncreased
update_stopping_criterion!(::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualPower}, ::Any)
update_stopping_criterion!(::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualFactor}, ::Any)
```

## Literature

```@bibliography
Pages = ["solvers/truncated_conjugate_gradient_descent.md"]
Canonical=false
```