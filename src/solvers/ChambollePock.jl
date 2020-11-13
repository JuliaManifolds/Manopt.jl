@doc raw"""
    ChambollePock(M, N, cost, x0, ξ0, m, n, prox_F, prox_G_dual, Λ, adjoint_DΛ)

Perform the Riemannian Chambolle–Pock algorithm.

Given a `cost` function $\mathcal E\colon\mathcal M \to ℝ$ of the form
```math
\mathcal E(x) = F(x) + G( Λ(x) ),
```
where $F\colon\mathcal M \to ℝ$, $G\colon\mathcal N \to ℝ$,
and $\Lambda\colon\mathcal M \to \mathcal N$. The remaining input parameters are

* `x,ξ` primal and dual start points $x\in\mathcal M$ and $\xi\in T_n\mathcal N$
* `m,n` base points on $\mathcal M$ and $\mathcal N$, respectively.
* `forward_operator` the operator $Λ(⋅)$ or its linearization $DΛ(⋅)[⋅]$, depending  within $G$
* `adjDΛ` the adjoint $DΛ^*$ of the linearized operator $DΛ(m)\colon T_{m}\mathcal M \to T_{Λ(m)}\mathcal N$
* `prox_F, prox_G_Dual` the proximal maps of $F$ and $G^\ast_n$

By default, this performs the exact Riemannian Chambolle Pock algorithm, see the opional parameter
`DΛ` for ther linearized variant.

# Optional Parameters

*  `acceleration` – (`0.05`)
* `dual_stepsize` – (`1/sqrt(8)`)
* `Λ` (`missing`) the exact operator, that is required if the forward operator is linearized;
  `missing` indicates, that the forward operator is exact.
* `primal_stepsize` – (`1/sqrt(8)`)
* `relaxation` – (`1.`)
* `relax` – (`:primal`) whether to relax the primal or dual
* `stopping_criterion` – (`stopAtIteration(100)`) a [`StoppingCriterion`](@ref)
* `update_primal_base` – (`(p,o,i) -> o.m`) function to update `m` (identity by default)
* `update_dual_base` – (`(p,o,i) -> o.n`) function to update `n` (identity by default)
"""
