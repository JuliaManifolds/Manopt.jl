# The Riemannian trust regions solver

Minimize a function

```math
\operatorname*{\arg\,min}_{p ∈ \mathcal{M}}\ f(p)
```

by using the Riemannian trust-regions solver following [AbsilBakerGallivan:2006](@cite) a model is build by
lifting the objective at the ``k``th iterate ``p_k`` by locally mapping the
cost function ``f`` to the tangent space as ``f_k: T_{p_k}\mathcal M → ℝ`` as
``f_k(X) = f(\operatorname{retr}_{p_k}(X))``.
The trust region subproblem is then defined as

```math
\operatorname*{arg\,min}_{X ∈ T_{p_k}\mathcal M}\ m_k(X),
```

where

```math
\begin{align*}
m_k&: T_{p_K}\mathcal M → ℝ,\\
m_k(X) &= f(p_k) + ⟨\operatorname{grad} f(p_k), X⟩_{p_k} + \frac{1}{2}\langle \mathcal H_k(X),X⟩_{p_k}\\
\text{such that}&\ \lVert X \rVert_{p_k} ≤ Δ_k.
\end{align*}
```

Here ``Δ_k`` is a trust region radius, that is adapted every iteration, and ``\mathcal H_k`` is
some symmetric linear operator that approximates the Hessian ``\operatorname{Hess} f`` of ``f``.


## Interface

```@docs
trust_regions
trust_regions!
```

## State

```@docs
TrustRegionsState
```

## Approximation of the Hessian

Several different methods to approximate the Hessian are available.

```@docs
ApproxHessianFiniteDifference
ApproxHessianSymmetricRankOne
ApproxHessianBFGS
```

as well as their (non-exported) common supertype

```@docs
Manopt.AbstractApproxHessian
```

## [Technical details](@id sec-tr-technical-details)

The [`trust_regions`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* By default the stopping criterion uses the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* if you do not provide an initial `max_trust_region_radius`, a [`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`) is required.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.
* By default the tangent vectors are initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.


## Literature

```@bibliography
Pages = ["trust_regions.md"]
Canonical=false
```
