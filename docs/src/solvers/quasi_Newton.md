# [Riemannian quasi-Newton methods](@id quasiNewton)

```@meta
    CurrentModule = Manopt
```

The aim is to minimize a real-valued function on a Riemannian manifold, i.e.

```math
\min f(x), \quad x \in \mathcal{M}.
```

Riemannian quasi-Newtonian methods are as generalizations of their Euclidean counterparts Riemannian line search methods. These methods determine a search direction ``η_k ∈ T_{x_k} \mathcal{M}`` at the current iterate ``x_k`` and a suitable stepsize ``α_k`` along ``\gamma(α) = R_{x_k}(α η_k)``, where ``R \colon T \mathcal{M} \to \mathcal{M}`` is a retraction. The next iterate is obtained by

```math
x_{k+1} = R_{x_k}(α_k η_k).
```

In quasi-Newton methods, the search direction is given by

```math
η_k = -{\mathcal{H}_k}^{-1}[\operatorname{grad} f (x_k)] = -\mathcal{B}_k [\operatorname{grad} f (x_k)],
```

where ``\mathcal{H}_k \colon T_{x_k} \mathcal{M} \to T_{x_k} \mathcal{M}`` is a positive definite self-adjoint operator, which approximates the action of the Hessian ``\operatorname{Hess} f (x_k)[\cdot]`` and ``\mathcal{B}_k = {\mathcal{H}_k}^{-1}``. The idea of quasi-Newton methods is instead of creating a complete new approximation of the Hessian operator ``\operatorname{Hess} f(x_{k+1})`` or its inverse at every iteration, the previous operator ``\mathcal{H}_k`` or ``\mathcal{B}_k`` is updated by a convenient formula using the obtained information about the curvature of the objective function during the iteration. The resulting operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ acts on the tangent space $T_{x_{k+1}} \mathcal{M}$ of the freshly computed iterate $x_{k+1}$.
In order to get a well-defined method, the following requirements are placed on the new operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ that is created by an update. Since the Hessian $\operatorname{Hess} f(x_{k+1})$ is a self-adjoint operator on the tangent space $T_{x_{k+1}} \mathcal{M}$, and $\mathcal{H}_{k+1}$ approximates it, we require that $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ is also self-adjoint on $T_{x_{k+1}} \mathcal{M}$. In order to achieve a steady descent, we want $η_k$ to be a descent direction in each iteration. Therefore we require, that $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ is a positive definite operator on $T_{x_{k+1}} \mathcal{M}$. In order to get information about the cruvature of the objective function into the new operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$, we require that it satisfies a form of a Riemannian quasi-Newton equation:

$\mathcal{H}_{k+1} [T_{x_k \rightarrow x_{k+1}}({R_{x_k}}^{-1}(x_{k+1}))] = \operatorname{grad} f(x_{k+1}) - T_{x_k \rightarrow x_{k+1}}(\operatorname{grad} f(x_k))$

or

$\mathcal{B}_{k+1} [\operatorname{grad} f(x_{k+1}) - T_{x_k \rightarrow x_{k+1}}(\operatorname{grad} f(x_k))] = T_{x_k \rightarrow x_{k+1}}({R_{x_k}}^{-1}(x_{k+1}))$

where $T_{x_k \rightarrow x_{k+1}} \colon T_{x_k} \mathcal{M} \to T_{x_{k+1}} \mathcal{M}$ and the chosen retraction $R$ is the associated retraction of $T$.
In the following we denote the specific operators in matrix notation and hence use $H_k$ and $B_k$, respectively.

## Operator Updates

The following update rules for either ``H_{k+1}`` or `` B_{k+1}`` are available.

```@docs
AbstractQuasiNewtonUpdateRule
BFGS
DFP
Broyden
SR1
InverseBFGS
InverseDFP
InverseBroyden
InverseSR1
```

## Initialization

Initialize $x_0 \in \mathcal{M}$ and let $\mathcal{B}_0 \colon T_{x_0} \mathcal{M} \to T_{x_0} \mathcal{M}$ be a positive definite, self-adjoint operator.

## Iteration

Repeat until a convergence criterion is reached

1. Compute $η_k = -\mathcal{B}_k [\operatorname{grad} f (x_k)]$ or solve $\mathcal{H}_k [η_k] = -\operatorname{grad} f (x_k)]$.
2. Determine a suitable stepsize $α_k$ along the curve given by $\gamma(α) = R_{x_k}(α η_k)$ (e.g. by using the Riemannian Wolfe conditions).
3. Compute $x_{k+1} = R_{x_k}(α_k)$.
4. Define $s_k = T_{x_k, α_k η_k}(α_k η_k)$ and $y_k = \operatorname{grad} f(x_{k+1}) - T_{x_k, α_k η_k}(\operatorname{grad} f(x_k))$.
5. Update $\mathcal{B}_k \text{ or } \mathcal{H}_k \mapsto \mathcal{B}_{k+1} \text{ or } \mathcal{H}_{k+1} \colon T_{x_{k+1}} \mathcal{M} \to T_{x_{k+1}} \mathcal{M}$.

## Result

The result is given by the last computed $x_K$.

## Locking condition

Essential for the positive definiteness of the operators $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ for many methods is the fulfilment of

$g_{x_{k+1}}(s_k, y_k) > 0,$

which is the so-called Riemannian curvature condition. If this condition holds in each iteration, the corresponding update, which uses an isometric vector transport $T^S$, produces a sequence of positive definite self-adjoint operators $\{ \mathcal{H}_k \}_k$ or $\{ \mathcal{B}_k \}_k$. The same association exists, of course, in the Euclidean case, where the fulfilment of the curvature condition $s^{\mathrm{T}}_k y_k$ is ensured by choosing a stepsize $α_k > 0$ in each ietartion that fulfils the Wolfe conditions. The generalisation of the Wolfe conditions for Riemannian manifolds is that the stepsize $α_k > 0$ satisfies

$f( R_{x_k}(α_k η_k)) \leq f(x_k) + c_1 α_k g_{x_k} (\operatorname{grad} f(x_k), η_k)$

and

$\frac{\mathrm{d}}{\mathrm{d}t} f(R_{x_k}(t \; η_k)) \vert_{t=α_k} \geq c_2 \frac{\mathrm{d}}{\mathrm{d}t} f(R_{x_k}(t \; η_k)) \vert_{t=0}.$

Using the vector transport by differentiated retraction $T^{R}{x, η_x}(\xi_x) = \frac{\mathrm{d}}{\mathrm{d}t} R_{x}(η_x + t \; \xi_x) \; \vert_{t=0}$, where the retraction $R$ is the associated retraction of the vector transport used in the update formula, the second condition can be rewritten as

$g_{R_{x_k}(α_k η_k)} (\operatorname{grad} f(R_{x_k}(α_k η_k)), T^{R}{x_k, α_k η_k}(η_k)) \geq c_2 g_{x_k} (\operatorname{grad} f(x_k), η_k).$

Unfortunately, in general, a stepsize $α_k > 0$ that satisfies the Riemannian Wolfe conditions and an isometric vector transport $T^S$ in the update formula does not in general lead to a positive definite update of the operator $\mathcal{H}_k$ or $\mathcal{B}_k$. This is because the Wolfe conditions use vector transport by differentiated retraction $T^R$, which is generally not the same as isometric vetor transport $T^S$ used in the update. However, in order to create a positive definite operator in each iteration, the so-called locking condition was introduced, which requires that the isometric vector transport $T^S$ and its associate retraction $R$ fulfil the following condition

$T^{S}{x, \xi_x}(\xi_x) = \beta T^{R}{x, \xi_x}(\xi_x), \quad \beta = \frac{\lVert \xi_x \rVert_x}{\lVert T^{R}{x, \xi_x}(\xi_x) \rVert_{R_{x}(\xi_x)}}.$

where $T^R$ is again the vector transport by differentiated retraction. With the requirement that the isometric vector transport $T^S$ and its associared retraction $R$ satisfies the locking condition and using the tangent vector

$y_k = {\beta_k}^{-1} \operatorname{grad} f(x_{k+1}) - T^{S}{x_k, α_k η_k}(\operatorname{grad} f(x_k)),$

where

$\beta_k = \frac{\lVert α_k η_k \rVert_{x_k}}{\lVert \lVert T^{R}{x_k, α_k η_k}(α_k η_k) \rVert_{x_{k+1}}},$

in the update, it can be shown that choosing a stepsize $α_k > 0$ that satisfies the Riemannian wolfe conditions leads to the fulfilment of the Riemannian curvature condition, which in turn implies that the operator generated by the updates is positive definite.

## Cautious BFGS

As in the Euclidean case, the Riemannian BFGS method is not globally convergent for general objective functions. To mitigate this obstacle, the cautious update has been generalised to Riemannian manifolds. For that, in each iteration, based on a decision rule, either the usual update is used or the current operator is transported into the upcoming tangent, which in turn means that the update consists of the vector transport of B only. In summary, this can be expressed as follows:

$\mathcal{B}^{CRBFGS}_{k+1} = \begin{cases} \text{using \cref{RiemannianInverseBFGSFormula}}, & \; \frac{g_{x_{k+1}}(y_k,s_k)}{\lVert s_k \rVert^{2}_{x_{k+1}}} \geq \theta(\lVert \operatorname{grad} f(x_k) \rVert_{x_k}), \\ \widetilde{\mathcal{B}}^{CRBFGS}_k, & \; \text{otherwise}, \end{cases}$

## Limited-memory Riemannian BFGS

As in the Euclidean case, for manifolds of very large dimensions, both the memory required and the computationally cost can be extremely high for quasi-Newton methods, since in each iteration an operator on the tangent space with the same dimension of the manifold must be transported and stored. To overcome this obstacle, the limited-memory BFGS method was generalised for the Riemannian setup.
At the beginning of the k-th iteration, the search direction $η_k = \mathcal{B}^{LRBFGS}_k [\operatorname{grad}f(x_k)]$ is calculated using the two-loop recursion with $m$ ($<$ dimension of the manifold) stored tangent vectors $\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1} \subset T_{x_k} \mathcal{M}$ and a positive definite selfadjoint operator $\mathcal{B}^{(0)}_k \colon T_{x_k} \mathcal{M} \to T_{x_k} \mathcal{M}$ that varies from iteration to iteration:

\begin{algorithm}[H]
	\begin{algorithmic}[1]
        \State $q = \operatorname{grad} f(x_k)$

        \For{$i = k-1, k-2, \cdots, k-m$}
            \State $\rho_i = \frac{1}{g_{x_k}(\widetilde{s}_i, \widetilde{y}_i)}$
            \State $\xi_i = \rho_i g_{x_k}(\widetilde{s}_i, q)$
            \State $q = q - \xi_i \widetilde{y}_i$
        \EndFor

        \State $r = \mathcal{B}^{(0)}_k[q]$

        \For{$i = k-m, k-m+1, \cdots, k-1$}
            \State $\omega = \rho_i g_{x_k}(\widetilde{y}_i, r)$
            \State $r= r  + (\xi_i - \omega) \widetilde{s}_i$
		\EndFor

		\State \textbf{Stop with result} $\mathcal{B}^{LRBFGS}_k[\operatorname{grad} f(x_k)]$.
    \end{algorithmic}
\end{algorithm}

This algorithm can be understood that the usual RBFGS update is executed $m$ times in a row on $\mathcal{B}^{(0)}_k$ and is directly applied on $\operatorname{grad}f(x_k)$. Therefore, only inner products and linear combinations in the tangent space are needed. We note that the resulting operator $\mathcal{B}^{LRBFGS}_k$ is an approximation of the operator $\mathcal{B}^{RBFGS}_k$, since for k>m the information of all previous iterations is not included in the operator (unless of course one chooses m as large as the maximum number of iterations).

The operator $\mathcal{B}^{(0)}_k$ can be chosen to be fixed in each iteration, but the principle known to be successful from the Euclidean case can easily be generalised for the Riemannian setup by choosing $\mathcal{B}^{(0)}_k[\cdot] = c_k \id_{T_{x_k} \mathcal{M}}[\cdot],$ where

$
c_k = \frac{\widetilde{s}^{\flat}_{k-1} \widetilde{y}_{k-1}}{\widetilde{y}^{\flat}_{k-1} \widetilde{y}_{k-1}} = \frac{s^{\flat}_{k-1} y_{k-1}}{y^{\flat}_{k-1} y_{k-1}} = \frac{g_{x_k}(s_{k-1}, y_{k-1})}{g_{x_k}(y_{k-1}, y_{k-1})}.
$

For the first iteration usually the identity operator is used. In the $k$-th iteration, the next iterate is calculated as usual, i.e. $x_{k+1} = R_{x_k}(α_k η_k)$, where alpha is a step size that satisfies the wolfe conditions. Then comes the crux of the algortihmus, which has a memory advantage over the usual quasi-Newton methods. When updating, the oldest vector pair $\{ \widetilde{s}_{k−m}, \widetilde{y}_{k−m}\}$ of the set $\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}$ is exchanged with the newest vector pair $\{ \widetilde{s}_k, \widetilde{y}_k\}$. There are two cases: if there is still free memory, i.e. $k < m$, the previously stored vector pairs $\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}$ have to be transported into the new tangent space $T_{x_k} \mathcal{M}$; if there is no free memory, the oldest pair has to be discarded and then all the remaining vector pairs are transported into the new tangent space. This method ensures that all tangent vectors are in the correct tangent space, i.e. T, so that in the next iteration the search direction can be recursively calculated again. After that we calculate and store s and y . This ensures that new information about the target function is always included and the old, probably no longer relevant, information is discarded.

In summary, the algorithm takes the following form:


## Interface

```@docs
    quasi_Newton
```

## Problem & Options

The quasi Newton algorithm is based on a [`GradientProblem`](@ref).

```@docs
QuasiNewtonOptions
```

## Literature
