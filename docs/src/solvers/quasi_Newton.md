# [Riemannian quasi-Newton methods](@id quasiNewton)

The aim is to minimize a real-valued function on a Riemannian manifold, i.e.

$\min f(x), \quad x \in \mathcal{M}$.

Riemannian quasi-Newtonian methods are as generalizations of their Euclidean counterparts Riemannian line search methods. These methods first determine a search direction $\eta_k$ in each iteration, which is a tangent vector in the tangent space $\tangent{x_k}$ at the current iterate $x_k$, then serach for a suitable stepsize $\alpha_k$ along the curve $\gamma(\alpha) = R_{x_k}(\alpha \eta_k)$ which is determined by a chosen retractio $R \colon \tangent{} \to \mathcal{M}$ and the search direction $\eta_k$. The next iterate is obtained by

$x_{k+1} = R_{x_k}(\alpha_k \eta_k)$.

The choice of a computationally efficient retraction is important, because it can influence the rate of convergence. 
In quasi-Newton methods, the search direction is given by

$\eta_k = -{\mathcal{H}_k}^{-1}[\operatorname{grad} f (x_k)] = -\mathcal{B}_k [\operatorname{grad} f (x_k)]$

where $\mathcal{H}_k \colon \tangent{x_k} \to \tangent{x_k}$ is a positive definite self-adjoint operator, which approximates the action of the Hessian $\operatorname{Hess} f (x_k)[\cdot]$ and $\mathcal{B}_k = {\mathcal{H}_k}^{-1}$. The idea of quasi-Newton methods is instead of creating a complete new approximation of the Hessian operator $\operatorname{Hess} f(x_{k+1})$ or its inverse at every iteration, the previous operator $\mathcal{H}_k$ or $\mathcal{B}_k$ is updated by a convenient formula using the obtained information about the curvature of the objective function during the iteration. The resulting operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ acts on the tangent space $\tangent{x_{k+1}}$ of the freshly computed iterate $x_{k+1}$.
In order to get a well-defined method, the following requirements are placed on the new operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ that is created by an update. Since the Hessian $\operatorname{Hess} f(x_{k+1})$ is a self-adjoint operator on the tangent space $\tangent{x_{k+1}}$, and $\mathcal{H}_{k+1}$ approximates it, we require that $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ is also self-adjoint on $\tangent{x_{k+1}}$. In order to achieve a steady descent, we want $\eta_k$ to be a descent direction in each iteration. Therefore we require, that $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ is a positive definite operator on $\tangent{x_{k+1}}$. In order to get information about the cruvature of the objective function into the new operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$, we require that it satisfies a form of a Riemannian quasi-Newton equation:

$\mathcal{H}_{k+1} [T_{x_k \rightarrow x_{k+1}}({R_{x_k}}^{-1}(x_{k+1}))] = \operatorname{grad} f(x_{k+1}) - T_{x_k \rightarrow x_{k+1}}(\operatorname{grad} f(x_k))$

or 

$\mathcal{B}_{k+1} [\operatorname{grad} f(x_{k+1}) - T_{x_k \rightarrow x_{k+1}}(\operatorname{grad} f(x_k))] = T_{x_k \rightarrow x_{k+1}}({R_{x_k}}^{-1}(x_{k+1}))$

where $T_{x_k \rightarrow x_{k+1}} \colon \tangent{x_k} \to \tangent{x_{k+1}}$ and the chosen retraction $R$ is the associated retraction of $T$. 
The idea of Riemannian quasi-Newton methods is to generate an operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ which meets all these requirements by a convenient update formula. Thereby, the quasi-Newton update formulas for matrices known from the Euclidean case were generalised for the Riemannian setup. 

## Operator Updates

There are many update formulas that pursue different goals and/or are based on different ideas. Before we can list them, we need to introduce some general definitions and terms that are used in the update formulas. 
Of course, in any update you want to take the information already stored in B, but this is an operator on T, which is generally not the same tangent space as T_ on which H operates. To overcome this obstacle, we introduce

$\widetilde{\mathcal{H}}_k = T^{S}_{x_k, \alpha_k \eta_k} \circ \mathcal{H}_k \circ {T^{S}_{x_k, \alpha_k \eta_k}}^{-1} \colon \tangent{x_{k+1}} \to \tangent{x_{k+1}},$

where $T^{S}_{x_k, \alpha_k \eta_k} \colon \tangent{x_k} \to \tangent{R_{x_k}(\alpha_k \eta_k)}$ is an isometric vector transport and $R_{x_k}(\cdot)$ is its associated retraction. Of course, one could take any vector transport $T$ to which $R$ is the associated retraction, i.e. $R_{x_k}(\alpha_k \eta_k) \in \mathcal{M}$, but since we want to take on the positive definiteness and self-adjointness of the operator $\mathcal{H}_k$, this is generally only ensured by an isometric vector transport $T^S$. The same method is used to define $\widetilde{\mathcal{B}}_k \colon \tangent{x_{k+1}} \to \tangent{x_{k+1}}$. 
To get the curvature information of the objective function into the update formula, the tangent vectors 

$s_k = T^{S}_{x_k, \alpha_k \eta_k}(\alpha_k \eta_k) \in \tangent{x_{k+1}}$

and 

$y_k = \operatorname{grad} f(x_{k+1}) - T^{S}_{x_k, \alpha_k \eta_k}(\operatorname{grad} f(x_k)) \in \tangent{x_{k+1}}$

are defined. Here, of course, we use the same vector transport as for $\widetilde{\mathcal{H}}_k$.

BFGS:

$\mathcal{H}^{RBFGS}_{k+1} [\cdot] = \widetilde{\mathcal{H}}^{RBFGS}_k [\cdot] + \frac{y_k y^{\flat}_k[\cdot]}{s^{\flat}_k [y_k]} - \frac{\widetilde{\mathcal{H}}^{RBFGS}_k [s_k] s^{\flat}_k (\widetilde{\mathcal{H}}^{RBFGS}_k [\cdot])}{s^{\flat}_k (\widetilde{\mathcal{H}}^{RBFGS}_k [s_k])}$

Inverse BFGS: 

$\mathcal{B}^{RBFGS}_{k+1} [\cdot] = \Big{(} \id_{\tangent{x_{k+1}}}[\cdot] - \frac{s_k y^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]} \Big{)} \widetilde{\mathcal{B}}^{RBFGS}_k [\cdot] \Big{(} \id_{\tangent{x_{k+1}}}[\cdot] - \frac{y_k s^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]} \Big{)} + \frac{s_k s^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]}$

DFP:

$\mathcal{H}^{RDFP}_{k+1} [\cdot] = \Big{(} \id_{\tangent{x_{k+1}}}[\cdot] - \frac{y_k s^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]} \Big{)} \widetilde{\mathcal{H}}^{RDFP}_k [\cdot] \Big{(} \id_{\tangent{x_{k+1}}}[\cdot] - \frac{s_k y^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]} \Big{)} + \frac{y_k y^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]}$

Inverse DFP:

$\mathcal{B}^{RDFP}_{k+1} [\cdot] = \widetilde{\mathcal{B}}^{RDFP}_k [\cdot] + \frac{s_k s^{\flat}_k[\cdot]}{s^{\flat}_k [y_k]} - \frac{\widetilde{\mathcal{B}}^{RDFP}_k [y_k]  y^{\flat}_k (\widetilde{\mathcal{B}}^{RDFP}_k [\cdot])}{y^{\flat}_k (\widetilde{\mathcal{B}}^{RDFP}_k [y_k])}$

Symmetric-Rank-1:

$\mathcal{H}^{RSR1}_{k+1} [\cdot] = \widetilde{\mathcal{H}}^{RSR1}_k [\cdot] + \frac{(y_k - \widetilde{\mathcal{H}}^{RSR1}_k [s_k])(y_k - \widetilde{\mathcal{H}}^{RSR1}_k [s_k])^{\flat}[\cdot]}{(y_k - \widetilde{\mathcal{H}}^{RSR1}_k [s_k])^{\flat}[s_k]}$

Inverse Symmetric-Rank-1:

$\mathcal{B}^{RSR1}_{k+1} [\cdot] = \widetilde{\mathcal{B}}^{RSR1}_k [\cdot] + \frac{(s_k - \widetilde{\mathcal{B}}^{RSR1}_k [y_k])(s_k - \widetilde{\mathcal{B}}^{RSR1}_k [y_k])^{\flat}[\cdot]}{(s_k - \widetilde{\mathcal{B}}^{RSR1}_k [y_k])^{\flat}[y_k]}$

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

