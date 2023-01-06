# [The Riemannian Chambolle-Pock Algorithm](@id ChambollePockSolver)

The Riemannian Chambolle–Pock is a generalization of the Chambolle–Pock algorithm[^ChambollePock2011].
It is also known as primal-dual hybrid gradient (PDHG) or primal-dual proximal splitting (PDPS) algorithm.

In order to minimize over $p∈\mathcal M§ the cost function consisting of

```math
F(p) + G(Λ(p)),
```

where $F:\mathcal M → \overline{ℝ}$, $G:\mathcal N → \overline{ℝ}$, and
$Λ:\mathcal M →\mathcal N$.
If the manifolds $\mathcal M$ or $\mathcal N$ are not Hadamard, it has to be considered locally,
i.e. on geodesically convex sets $\mathcal C \subset \mathcal M$ and $\mathcal D \subset\mathcal N$
such that $Λ(\mathcal C) \subset \mathcal D$.

The algorithm is available in four variants: exact versus linearized (see `variant`)
as well as with primal versus dual relaxation (see `relax`). For more details, see
[^BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez2020].
In the following we note the case of the exact, primal relaxed Riemannian Chambolle–Pock algorithm.

Given base points $m∈\mathcal C$, $n=Λ(m)∈\mathcal D$,
initial primal and dual values $p^{(0)} ∈\mathcal C$, $ξ_n^{(0)} ∈T_n^*\mathcal N$,
and primal and dual step sizes $\sigma_0$, $\tau_0$, relaxation $\theta_0$,
as well as acceleration $\gamma$.

As an initialization, perform $\bar p^{(0)} \gets p^{(0)}$.

The algorithms performs the steps $k=1,…,$ (until a [`StoppingCriterion`](@ref) is fulfilled with)

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

Finally you can also update the base points $m$ and $n$ during the iterations.
This introduces a few additional vector transports. The same holds for the case
$Λ(m^{(k)})\neq n^{(k)}$ at some point. All these cases are covered in the algorithm.

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

## Useful Terms

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

[^ChambollePock2011]:
    > A. Chambolle, T. Pock:
    > _A first-order primal-dual algorithm for convex problems with applications to imaging_,
    > Journal of Mathematical Imaging and Vision 40(1), 120–145, 2011.
    > doi: [10.1007/s10851-010-0251-1](https://dx.doi.org/10.1007/s10851-010-0251-1)