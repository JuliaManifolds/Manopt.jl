#
# Common files for bunlde-based solvers
#

function bundle_method_subsolver end
@doc raw"""
    bundle_method_subsolver(M, bms<:Union{ConvexBundleMethodState, ProximalBundleMethodState})

solver for the subproblem of both the convex and proximal bundle methods.

The subproblem for the convex bundle method is
```math
\begin{align*}
    \operatorname*{arg\,min}_{λ ∈ ℝ^{\lvert J_k\rvert}}&
    \frac{1}{2} \Bigl\lVert \sum_{j ∈ J_k} λ_j \mathrm{P}_{p_k←q_j} X_{q_j} \Bigr\rVert^2
    + \sum_{j ∈ J_k} λ_j \, c_j^k
    \\
    \text{s. t.}\quad &
    \sum_{j ∈ J_k} λ_j = 1,
    \quad λ_j ≥ 0
    \quad \text{for all }
    j ∈ J_k,
\end{align*}
```
where ``J_k = \{j ∈ J_{k-1} \ | \ λ_j > 0\} \cup \{k\}``.

The subproblem for the proximal bundle method is

```math
\begin{align*}
    \operatorname*{arg\,min}_{λ ∈ ℝ^{\lvert L_l\rvert}} &
    \frac{1}{2 \mu_l} \Bigl\lVert \sum_{j ∈ L_l} λ_j \mathrm{P}_{p_k←q_j} X_{q_j} \Bigr\rVert^2
    + \sum_{j ∈ L_l} λ_j \, c_j^k
    \\
    \text{s. t.} \quad &
    \sum_{j ∈ L_l} λ_j = 1,
    \quad λ_j ≥ 0
    \quad \text{for all } j ∈ L_l,
\end{align*}
```
where ``L_l = \{k\}`` if ``q_k`` is a serious iterate, and ``L_l = L_{l-1} \cup \{k\}`` otherwise.
"""
bundle_method_subsolver(M, s) #change to problem state?
