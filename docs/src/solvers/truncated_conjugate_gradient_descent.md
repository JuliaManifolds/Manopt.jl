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
update_stopping_criterion!(::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualPower}, ::Any)
update_stopping_criterion!(::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualFactor}, ::Any)
```

## Trust region model

```@docs
TrustRegionModelObjective
```

## [Technical details](@id sec-tr-technical-details)

The [`trust_regions`](@ref) solver requires the following functions of a manifold to be available

* if you do not provide a `trust_region_radius=`, then [`injectivity_radius`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.injectivity_radius-Tuple{AbstractManifold}) on the manifold `M` is required.
* the [`norm`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#LinearAlgebra.norm-Tuple{AbstractManifold,%20Any,%20Any}) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* A [`zero_vector!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.zero_vector-Tuple{AbstractManifold,%20Any})`(M,X,p)`.
* A [`copyto!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.copyto!-Tuple{AbstractManifold,%20Any,%20Any})`(M, q, p)` and [`copy`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.copy-Tuple{AbstractManifold,%20Any})`(M,p)` for points.

## Literature

```@bibliography
Pages = ["truncated_conjugate_gradient_descent.md"]
Canonical=false
```
