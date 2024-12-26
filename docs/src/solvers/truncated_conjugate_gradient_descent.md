# [Steihaug-Toint truncated conjugate gradient method](@id tCG)

Solve the constraint optimization problem on the tangent space

```math
\begin{align*}
\operatorname*{arg\,min}_{Y  ∈  T_p\mathcal{M}}&\ m_p(Y) = f(p) +
⟨\operatorname{grad}f(p), Y⟩_p + \frac{1}{2} ⟨\mathcal{H}_p[Y], Y⟩_p\\
\text{such that}& \ \lVert Y \rVert_p ≤ Δ
\end{align*}
```

on the tangent space ``T_p\mathcal M`` of a Riemannian manifold ``\mathcal M`` by using the Steihaug-Toint truncated conjugate-gradient (tCG) method,
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

## Stopping criteria

```@docs
StopWhenResidualIsReducedByFactorOrPower
StopWhenTrustRegionIsExceeded
StopWhenCurvatureIsNegative
StopWhenModelIncreased
Manopt.set_parameter!(::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualPower}, ::Any)
Manopt.set_parameter!(::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualFactor}, ::Any)
```

## Trust region model

```@docs
TrustRegionModelObjective
```

## [Technical details](@id sec-tr-technical-details)

The [`trust_regions`](@ref) solver requires the following functions of a manifold to be available

* if you do not provide a `trust_region_radius=`, then [`injectivity_radius`](@extref `ManifoldsBase.injectivity_radius-Tuple{AbstractManifold}`) on the manifold `M` is required.
* the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* A [`zero_vector!`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,X,p)`.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.

## Literature

```@bibliography
Pages = ["truncated_conjugate_gradient_descent.md"]
Canonical=false
```
