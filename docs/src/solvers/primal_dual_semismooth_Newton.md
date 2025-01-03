# [Primal-dual Riemannian semismooth Newton algorithm](@id solver-pdrssn)

The Primal-dual Riemannian semismooth Newton Algorithm is a second-order method derived from the [`ChambollePock`](@ref).

The aim is to solve an optimization problem on a manifold with a cost function of the form

```math
F(p) + G(Λ(p)),
```

where ``F:\mathcal M → \overline{ℝ}``, ``G:\mathcal N → \overline{ℝ}``, and
``Λ:\mathcal M →\mathcal N``.
If the manifolds ``\mathcal M`` or ``\mathcal N`` are not Hadamard, it has to be considered locally only, that is on geodesically convex sets ``\mathcal C \subset \mathcal M`` and ``\mathcal D \subset\mathcal N``
such that ``Λ(\mathcal C) \subset \mathcal D``.

The algorithm comes down to applying the Riemannian semismooth Newton method to the rewritten primal-dual optimality conditions. Define the vector field ``X: \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N} \rightarrow \mathcal{T} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}`` as

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

## [Technical details](@id sec-ssn-technical-details)

The [`primal_dual_semismooth_Newton`](@ref) solver requires the following functions of a manifold to be available for both the manifold ``\mathcal M``and ``\mathcal N``

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `inverse_retraction_method=` does not have to be specified.
* A [`vector_transport_to!`](@extref ManifoldsBase :doc:`vector_transports`)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `vector_transport_method=` does not have to be specified.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.
* A [`get_basis`](@extref `ManifoldsBase.get_basis-Tuple{AbstractManifold, Any, ManifoldsBase.AbstractBasis}`) for the [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) on ``\mathcal M``
* [`exp`](@extref `Base.exp-Tuple{AbstractManifold, Any, Any}`) and [`log`](@extref `Base.log-Tuple{AbstractManifold, Any, Any}`) (on ``\mathcal M``)
* A [`DiagonalizingOrthonormalBasis`](@extref `ManifoldsBase.DiagonalizingOrthonormalBasis`) to compute the differentials of the exponential and logarithmic map
* Tangent vectors storing the social and cognitive vectors are initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.

## Literature

```@bibliography
Pages = ["primal_dual_semismooth_Newton.md"]
Canonical=false
```
