# [The Primal-dual Riemannian semismooth Newton Algorithm](@id PDRSSNSolver)

The Primal-dual Riemannian semismooth Newton Algorithm is a second-order method derived from the [`ChambollePock`](@ref).

The aim is to solve an optimization problem on a manifold with a cost function of the form

```math
F(p) + G(Λ(p)),
```

where ``F:\mathcal M → \overline{ℝ}``, ``G:\mathcal N → \overline{ℝ}``, and
``Λ:\mathcal M →\mathcal N``.
If the manifolds ``\mathcal M`` or ``\mathcal N`` are not Hadamard, it has to be considered locally,
i.e. on geodesically convex sets ``\mathcal C \subset \mathcal M`` and ``\mathcal D \subset\mathcal N``
such that ``Λ(\mathcal C) \subset \mathcal D``.

The algorithm comes down to applying the Riemannian semismooth Newton method to the rewritten primal-dual optimality conditions, i.e., we define the vector field ``X: \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N} \rightarrow \mathcal{T} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}`` as

```math
X\left(p, \xi_{n}\right):=\left(\begin{array}{c}
-\log _{p} \operatorname{prox}_{\sigma F}\left(\exp _{p}\left(\mathcal{P}_{p \leftarrow m}\left(-\sigma\left(D_{m} \Lambda\right)^{*}\left[\mathcal{P}_{\Lambda(m) \leftarrow n} \xi_{n}\right]\right)^{\sharp}\right)\right) \\
\xi_{n}-\operatorname{prox}_{\tau G_{n}^{*}}\left(\xi_{n}+\tau\left(\mathcal{P}_{n \leftarrow \Lambda(m)} D_{m} \Lambda\left[\log _{m} p\right]\right)^{\flat}\right)
\end{array}\right)
```

and solve for ``X(p,ξ_{n})=0``.

Given base points ``m∈\mathcal C``, ``n=Λ(m)∈\mathcal D``,
initial primal and dual values ``p^{(0)} ∈\mathcal C``, ``ξ_{n}^{(0)} ∈ \mathcal T_{n}^{*}\mathcal N``,
and primal and dual step sizes ``\sigma``, ``\tau``.

The algorithms performs the steps ``k=1,…,`` (until a [`StoppingCriterion`](@ref) is reached)

1.  Choose any element
   ```math
   V^{(k)} ∈ ∂_C X(p^{(k)},ξ_n^{(k)})
   ```
   of the Clarke generalized covariant derivative
2. Solve
   ```math
   V^{(k)} [(d_p^{(k)}, d_n^{(k)})] = - X(p^{(k)},ξ_n^{(k)})
   ```
   in the vector space ``\mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}``
3. Update
   ```math
   p^{(k+1)} := \exp_{p^{(k)}}(d_p^{(k)})
   ```
   and
   ```math
   ξ_n^{(k+1)} := ξ_n^{(k)} + d_n^{(k)}
   ```

Furthermore you can exchange the exponential map, the logarithmic map, and the parallel transport
by a retraction, an inverse retraction and a vector transport.

Finally you can also update the base points ``m`` and ``n`` during the iterations.
This introduces a few additional vector transports. The same holds for the case that
``Λ(m^{(k)})\neq n^{(k)}`` at some point. All these cases are covered in the algorithm.

```@meta
CurrentModule = Manopt
```

```@docs
primal_dual_semismooth_Newton
primal_dual_semismooth_Newton!
```

## State

```@docs
PrimalDualSemismoothNewtonState
```

## Literature

```@bibliography
Pages = ["solvers/primal_dual_semismooth_Newton.md"]
Canonical=false
```