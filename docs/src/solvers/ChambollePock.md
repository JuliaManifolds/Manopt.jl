# The Riemannian Chambolle-Pock algorithm

The Riemannian Chambolle—Pock is a generalization of the Chambolle—Pock algorithm [ChambollePock:2011](@citet*)
It is also known as primal-dual hybrid gradient (PDHG) or primal-dual proximal splitting (PDPS) algorithm.

In order to minimize over ``p∈\mathcal M`` the cost function consisting of
In order to minimize a cost function consisting of

```math
F(p) + G(Λ(p)),
```

 over ``p∈\mathcal M``

where ``F:\mathcal M → \overline{ℝ}``, ``G:\mathcal N → \overline{ℝ}``, and
``Λ:\mathcal M →\mathcal N``.
If the manifolds ``\mathcal M`` or ``\mathcal N`` are not Hadamard, it has to be considered locally only, that is on geodesically convex sets ``\mathcal C \subset \mathcal M`` and ``\mathcal D \subset\mathcal N``
such that ``Λ(\mathcal C) \subset \mathcal D``.

The algorithm is available in four variants: exact versus linearized (see `variant`)
as well as with primal versus dual relaxation (see `relax`). For more details, see
[BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021](@citet*).
In the following description is the case of the exact, primal relaxed Riemannian Chambolle—Pock algorithm.

Given base points ``m∈\mathcal C``, ``n=Λ(m)∈\mathcal D``,
initial primal and dual values ``p^{(0)} ∈\mathcal C``, ``ξ_n^{(0)} ∈T_n^*\mathcal N``,
and primal and dual step sizes ``\sigma_0``, ``\tau_0``, relaxation ``\theta_0``,
as well as acceleration ``\gamma``.

As an initialization, perform ``\bar p^{(0)} \gets p^{(0)}``.

The algorithms performs the steps ``k=1,…,`` (until a [`StoppingCriterion`](@ref) is fulfilled with)

1. ```math
   ξ^{(k+1)}_n = \operatorname{prox}_{\tau_k G_n^*}\Bigl(ξ_n^{(k)} + \tau_k \bigl(\log_n Λ (\bar p^{(k)})\bigr)^\flat\Bigr)
   ```
2. ```math
   p^{(k+1)} = \operatorname{prox}_{\sigma_k F}\biggl(\exp_{p^{(k)}}\Bigl( \operatorname{PT}_{p^{(k)}\gets m}\bigl(-\sigma_k DΛ(m)^*[ξ_n^{(k+1)}]\bigr)^\sharp\Bigr)\biggr)
   ```
3. Update
   * ``\theta_k = (1+2\gamma\sigma_k)^{-\frac{1}{2}}``
   * ``\sigma_{k+1} = \sigma_k\theta_k``
   * ``\tau_{k+1} =  \frac{\tau_k}{\theta_k}``
4. ```math
   \bar p^{(k+1)}  = \exp_{p^{(k+1)}}\bigl(-\theta_k \log_{p^{(k+1)}} p^{(k)}\bigr)
   ```

Furthermore you can exchange the exponential map, the logarithmic map, and the parallel transport
by a retraction, an inverse retraction, and a vector transport.

Finally you can also update the base points ``m`` and ``n`` during the iterations.
This introduces a few additional vector transports. The same holds for the case
``Λ(m^{(k)})\neq n^{(k)}`` at some point. All these cases are covered in the algorithm.

```@meta
CurrentModule = Manopt
```

```@docs
ChambollePock
ChambollePock!
```

## State

```@docs
ChambollePockState
```

## Useful terms

```@docs
primal_residual
dual_residual
```

## Debug

```@docs
DebugDualBaseIterate
DebugDualBaseChange
DebugPrimalBaseIterate
DebugPrimalBaseChange
DebugDualChange
DebugDualIterate
DebugDualResidual
DebugPrimalChange
DebugPrimalIterate
DebugPrimalResidual
DebugPrimalDualResidual
```

## Record

```@docs
RecordDualBaseIterate
RecordDualBaseChange
RecordDualChange
RecordDualIterate
RecordPrimalBaseIterate
RecordPrimalBaseChange
RecordPrimalChange
RecordPrimalIterate
```

## Internals

```@docs
Manopt.update_prox_parameters!
```

## [Technical details](@id sec-cp-technical-details)

The [`ChambollePock`](@ref) solver requires the following functions of a manifold to be available for both the manifold ``\mathcal M``and ``\mathcal N``

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` or `retraction_method_dual=` (for ``\mathcal N``) does not have to be specified.
* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `inverse_retraction_method=` or `inverse_retraction_method_dual=` (for ``\mathcal N``) does not have to be specified.
* A [`vector_transport_to!`](@extref ManifoldsBase :doc:`vector_transports`)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `vector_transport_method=` or `vector_transport_method_dual=` (for ``\mathcal N``) does not have to be specified.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.

## Literature



```@bibliography
Pages = ["ChambollePock.md"]
Canonical=false
```
