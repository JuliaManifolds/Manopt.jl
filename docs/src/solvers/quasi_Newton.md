# [Riemannian quasi-Newton methods](@id quasiNewton)

The aim is to minimize a real-valued function on a Riemannian manifold, i.e.

$\min f(x), \quad x \in \mathcal{M}$.

Riemannian quasi-Newtonian methods are as generalizations of their Euclidean counterparts Riemannian line search methods. These methods first determine a search direction $\eta_k$ in each iteration, which is a tangent vector in the tangent space $\tangent{x_k}$ of the current iterate $x_k$, then serach for a suitable stepsize $\alpha_k$ along the curve $\gamma(\alpha) = R_{x_k}(\alpha \eta_k)$ which is determined by a chosen retractio $R \colon \tangent{} \to \mathcal{M}$ and the search direction $\eta_k$. The next iterate is obtained by

$x_{k+1} = R_{x_k}(\alpha_k \eta_k)$.

The choice of a computationally efficient retraction is important, because it can influence the rate of convergence. In quasi-Newton methods, the search direction is given by

$\eta_k = -{\mathcal{H}_k}^{-1}[\operatorname{grad} f (x_k)] = -\mathcal{B}_k [\operatorname{grad} f (x_k)]$

where the operator $\mathcal{H}_k \colon \tangent{x_k} \to \tangent{x_k}$ approximates the action of the Hessian $\operatorname{Hess} f (x_k)[\cdot]$ and $\mathcal{B}_k = {\mathcal{H}_k}^{-1}$. The operators  are not generated anew in each iteration, but are updated at the end of each iteration with curvature information of the objective function to 


In order to get a well-defined method, the following requirements are placed on the operator that is created by an update. Since the hessian is a self-adjoint operator on the tangent space, and H-k tries to approximate it, this operator should also be self-adjoint. In order to achieve a steady descent, one wants eta to be a direction of descent in each iteration. This can only be achieved by ensuring through the update that h is a positive definite operator. In order to get information into the new operator H, we require that it satisfies the general Riemann quasi-Newton equation:

$\mathcal{H}_{k+1} [T_{x_k \rightarrow x_{k+1}}({R_{x_k}}^{-1}(x_{k+1}))] = \operatorname{grad} f(x_{k+1}) - T_{x_k \rightarrow x_{k+1}}(\operatorname{grad} f(x_k))$

where $R$ is the associated retraction of $T$. The idea of Riemannian quasi-Newton methods is to generate this operator H_k+1 by a convenient update formula.

## Operator Updates

## Initialization

Initialize $x_0 \in \mathcal{M}$ and let $\mathcal{B}_0 \colon \tangent{x_0} \to \tangent{x_0}$ be a positive definite, self-adjoint operator.

## Iteration

Repeat until a convergence criterion is reached

1. Compute $\eta_k = -\mathcal{B}_k [\operatorname{grad} f (x_k)]$ or solve $\mathcal{H}_k [\eta_k] = -\operatorname{grad} f (x_k)]$.
2. Determine a suitable stepsize $\alpha_k$ along the curve given by $\gamma(\alpha) = R_{x_k}(\alpha \eta_k)$ (e.g. by using the Riemannian Wolfe conditions).
3. Compute $x_{k+1} = R_{x_k}(\alpha_k)$.
4. Define $s_k = T_{x_k, \alpha_k \eta_k}(\alpha_k \eta_k)$ and $y_k = \operatorname{grad} f(x_{k+1}) - T_{x_k, \alpha_k \eta_k}(\operatorname{grad} f(x_k))$.
5. Update $\mathcal{B}_k \text{ or } \mathcal{H}_k \mapsto \mathcal{B}_{k+1} \text{ or } \mathcal{H}_{k+1} \colon \tangent{x_{k+1}} \to \tangent{x_{k+1}}$.

## Result

The result is given by the last computed $x_K$.

## Locking condition

## Cautious BFGS

## Limited-memory Riemannian BFGS


## Interface

```@meta
CurrentModule = Manopt
```

```@docs
quasi_Newton
```

## Problem & Options

```@docs
GradientProblem
QuasiNewtonOptions
```


## Literature

