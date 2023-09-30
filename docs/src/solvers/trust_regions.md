# [The Riemannian Trust-Regions Solver](@id trust_regions)

Minimize a function

```math
\operatorname*{\arg\,min}_{p ∈ \mathcal{M}}\ f(p)
```

by using the Riemannian trust-regions solver following [AbsilBakerGallivan:2006](@cite),
i.e. by building a lifted model at the ``k``th iterate ``p_k`` by locally mapping the
cost function ``f`` to the tangent space as ``f_k: T_{p_k}\mathcal M → \mathbb R`` as
``f_k(X) = f(\operatorname{retr}_{p_k}(X))``.
We then define the trust region subproblem as

```math
\operatorname*{arg\,min}_{X ∈ T_{p_k}\mathcal M}\ m_k(X),
```

where

```math
\begin{align*}
m_k&: T_{p_K}\mathcal M → \mathbb R,\\
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

We currently provide a few different methods to approximate the Hessian.

```@docs
ApproxHessianFiniteDifference
ApproxHessianSymmetricRankOne
ApproxHessianBFGS
```

as well as their (non-exported) common supertype

```@docs
Manopt.AbstractApproxHessian
```

## Literature

```@bibliography
Pages = ["solvers/trust_regions.md"]
Canonical=false
```