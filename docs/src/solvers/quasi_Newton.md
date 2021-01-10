# [Riemannian quasi-Newton methods](@id quasiNewton)

The aim is to minimize a real-valued function on a Riemannian manifold, i.e.

$\min f(x), \quad x \in \mathcal{M}.$

Riemannian quasi-Newtonian methods are as generalizations of their Euclidean counterparts Riemannian line search methods. These methods first determine a search direction $\eta_k$ in each iteration, which is a tangent vector in the tangent space $T_{x_k} \mathcal{M}$ at the current iterate $x_k$, then serach for a suitable stepsize $\alpha_k$ along the curve $\gamma(\alpha) = R_{x_k}(\alpha \eta_k)$ which is determined by a chosen retractio $R \colon T \mathcal{M} \to \mathcal{M}$ and the search direction $\eta_k$. The next iterate is obtained by

$x_{k+1} = R_{x_k}(\alpha_k \eta_k).$

The choice of a computationally efficient retraction is important, because it can influence the rate of convergence. 
In quasi-Newton methods, the search direction is given by

$\eta_k = -{\mathcal{H}_k}^{-1}[\operatorname{grad} f (x_k)] = -\mathcal{B}_k [\operatorname{grad} f (x_k)],$

where $\mathcal{H}_k \colon T_{x_k} \mathcal{M} \to T_{x_k} \mathcal{M}$ is a positive definite self-adjoint operator, which approximates the action of the Hessian $\operatorname{Hess} f (x_k)[\cdot]$ and $\mathcal{B}_k = {\mathcal{H}_k}^{-1}$. The idea of quasi-Newton methods is instead of creating a complete new approximation of the Hessian operator $\operatorname{Hess} f(x_{k+1})$ or its inverse at every iteration, the previous operator $\mathcal{H}_k$ or $\mathcal{B}_k$ is updated by a convenient formula using the obtained information about the curvature of the objective function during the iteration. The resulting operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ acts on the tangent space $T_{x_{k+1}} \mathcal{M}$ of the freshly computed iterate $x_{k+1}$.
In order to get a well-defined method, the following requirements are placed on the new operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ that is created by an update. Since the Hessian $\operatorname{Hess} f(x_{k+1})$ is a self-adjoint operator on the tangent space $T_{x_{k+1}} \mathcal{M}$, and $\mathcal{H}_{k+1}$ approximates it, we require that $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ is also self-adjoint on $T_{x_{k+1}} \mathcal{M}$. In order to achieve a steady descent, we want $\eta_k$ to be a descent direction in each iteration. Therefore we require, that $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ is a positive definite operator on $T_{x_{k+1}} \mathcal{M}$. In order to get information about the cruvature of the objective function into the new operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$, we require that it satisfies a form of a Riemannian quasi-Newton equation:

$\mathcal{H}_{k+1} [T_{x_k \rightarrow x_{k+1}}({R_{x_k}}^{-1}(x_{k+1}))] = \operatorname{grad} f(x_{k+1}) - T_{x_k \rightarrow x_{k+1}}(\operatorname{grad} f(x_k))$

or 

$\mathcal{B}_{k+1} [\operatorname{grad} f(x_{k+1}) - T_{x_k \rightarrow x_{k+1}}(\operatorname{grad} f(x_k))] = T_{x_k \rightarrow x_{k+1}}({R_{x_k}}^{-1}(x_{k+1}))$

where $T_{x_k \rightarrow x_{k+1}} \colon T_{x_k} \mathcal{M} \to T_{x_{k+1}} \mathcal{M}$ and the chosen retraction $R$ is the associated retraction of $T$. 
The idea of Riemannian quasi-Newton methods is to generate an operator $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ which meets all these requirements by a convenient update formula. Thereby, the quasi-Newton update formulas for matrices known from the Euclidean case were generalised for the Riemannian setup. 

## Operator Updates

There are many update formulas that pursue different goals and/or are based on different ideas. For this, the following general definitions and terms will be needed:

Of course, in any update one wants to take over the information already stored in $\mathcal{H}_k$, or $\mathcal{B}_k$, but this is an operator on $T_{x_k} \mathcal{M}$, which is in general not the same tangent space as $T_{x_{k+1}} \mathcal{M}$ on which $\mathcal{H}_{k+1}$, or $\mathcal{B}_{k+1}$, operates. To overcome this obstacle, we introduce

$\widetilde{\mathcal{H}}_k = T^{S}_{x_k, \alpha_k \eta_k} \circ \mathcal{H}_k \circ {T^{S}_{x_k, \alpha_k \eta_k}}^{-1} \colon T_{x_{k+1}} \mathcal{M} \to T_{x_{k+1}} \mathcal{M},$

where $T^{S}_{x_k, \alpha_k \eta_k} \colon T_{x_k} \mathcal{M} \to T_{R_{x_k}(\alpha_k \eta_k)} \mathcal{M}$ is an isometric vector transport and $R_{x_k}(\cdot)$ is its associated retraction. Of course, one could take any vector transport $T \colon T_{x_k} \mathcal{M} \to T_{x_{k+1}} \mathcal{M}$ to which $R$ is the associated retraction, i.e. $x_{k+1} = R_{x_k}(\alpha_k \eta_k) \in \mathcal{M}$, but since we want to take on the positive definiteness and self-adjointness of the operator $\mathcal{H}_k$, this is in general only ensured by an isometric vector transport $T^S$. The same method is used to define $\widetilde{\mathcal{B}}_k \colon T_{x_{k+1}} \mathcal{M} \to T_{x_{k+1}} \mathcal{M}$. 
To get the curvature information of the objective function into the update formula, the tangent vectors 

$s_k = T^{S}_{x_k, \alpha_k \eta_k}(\alpha_k \eta_k) \in T_{x_{k+1}} \mathcal{M}$

and 

$y_k = \operatorname{grad} f(x_{k+1}) - T^{S}_{x_k, \alpha_k \eta_k}(\operatorname{grad} f(x_k)) \in T_{x_{k+1}} \mathcal{M}$

are defined. Here, of course, the same vector transport as for $\widetilde{\mathcal{H}}_k$, or $\widetilde{\mathcal{B}}_k$ is used.
In Euclidean quasi-Newton methods, the property that $s_k s_k^{\mathrm{T}}$ is a positive definite symmetric rank one matrix is used to construct matrix updates. For the generalisation we need the musical isomorphism $\flat$, which turns a tangent vector into a cotangent vector, i.e. 

$\flat \colon \; T_{x} \mathcal{M} \ni \xi_x \mapsto \xi^{\flat}_x \in T^{*}_{x} \mathcal{M}$

and $\xi^{\flat}_x \colon T_{x} \mathcal{M} \to T_{x} \mathcal{M}$ satisfies

$\xi^{\flat}_x(\eta_x) = g_x(\xi_x,\eta_x),$

where $g_x \colon T_{x} \mathcal{M} \times T_{x} \mathcal{M} \to \mathbb{R}$ is the chosen Riemannian metric of the manifold $\mathcal{M}$. This allows one to define e.g. $y_k y^{\flat}_k$, which is a postive definite self-adjoint rank one operator on $T_{x_k} \mathcal{M}$. 
With these definitions and terms, the following update formulas can now be defined. We note that all the methods described here satisfy a form of the Riemannian quasi-Newton equation. 
The Riemannian BFGS update, also called RBFGS update, is a rank two operator update which turns a positive definite self-adjoint operator $\mathcal{H}^{RBFGS}_k$ on $T_{x_k} \mathcal{M}$ into a positive definite self-adjoint operator $\mathcal{H}^{RBFGS}_{k+1}$ on $T_{x_{k+1}} \mathcal{M}$, if $g_{x_{k+1}}(s_k, y_k) > 0$ holds:

$\mathcal{H}^{RBFGS}_{k+1} [\cdot] = \widetilde{\mathcal{H}}^{RBFGS}_k [\cdot] + \frac{y_k y^{\flat}_k[\cdot]}{s^{\flat}_k [y_k]} - \frac{\widetilde{\mathcal{H}}^{RBFGS}_k [s_k] s^{\flat}_k (\widetilde{\mathcal{H}}^{RBFGS}_k [\cdot])}{s^{\flat}_k (\widetilde{\mathcal{H}}^{RBFGS}_k [s_k])}.$

Using the Sherman-Morrison-Woodbury formula for operators, we obtain the inverse RBFGS update, which also produces a positive definite self-adjoint operator $\mathcal{B}^{RBFGS}_{k+1}$ if $\mathcal{B}^{RBFGS}_k$ is positive definite and self-adjoint on $T_{x_k} \mathcal{M}$ and $g_{x_{k+1}}(s_k, y_k) > 0$ holds:

$\mathcal{B}^{RBFGS}_{k+1} [\cdot] = \Big{(} \id_{T_{x_{k+1}} \mathcal{M}}[\cdot] - \frac{s_k y^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]} \Big{)} \widetilde{\mathcal{B}}^{RBFGS}_k [\cdot] \Big{(} \id_{T_{x_{k+1}} \mathcal{M}}[\cdot] - \frac{y_k s^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]} \Big{)} + \frac{s_k s^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]}.$

By replacing the triple $(\widetilde{\mathcal{B}}^{RBFGS}_k, s_k, y_k)$ by the triple $(\widetilde{\mathcal{H}}^{RDFP}_k, y_k, s_k)$ one obtains the Riemannian DFP update, short RDFP, which also generates a positive definite self-adjoint operator $\mathcal{H}^{RDFP}_{k+1}$ on $T_{x_{k+1}} \mathcal{M}$ if $\mathcal{H}^{RDFP}_k$ is positive definite and self-adjoint and $g_{x_{k+1}}(s_k, y_k) > 0$ holds:

$\mathcal{H}^{RDFP}_{k+1} [\cdot] = \Big{(} \id_{T_{x_{k+1}} \mathcal{M}}[\cdot] - \frac{y_k s^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]} \Big{)} \widetilde{\mathcal{H}}^{RDFP}_k [\cdot] \Big{(} \id_{T_{x_{k+1}} \mathcal{M}}[\cdot] - \frac{s_k y^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]} \Big{)} + \frac{y_k y^{\flat}_k [\cdot]}{s^{\flat}_k [y_k]}.$

The RDFP and RBFGS update can be referred to as dual updates, similar to the Euclidean case, since the other update formula is obtained by exchanging the variables in one update formula. Therefore, one can also very quickly set up the inverse RDFP update, but it can also be achieved again by the Sherman-Morrison-Woodbury formula for operators:

$\mathcal{B}^{RDFP}_{k+1} [\cdot] = \widetilde{\mathcal{B}}^{RDFP}_k [\cdot] + \frac{s_k s^{\flat}_k[\cdot]}{s^{\flat}_k [y_k]} - \frac{\widetilde{\mathcal{B}}^{RDFP}_k [y_k]  y^{\flat}_k (\widetilde{\mathcal{B}}^{RDFP}_k [\cdot])}{y^{\flat}_k (\widetilde{\mathcal{B}}^{RDFP}_k [y_k])}.$

By the convex combination of the RBFGS and the RDFP updates one obtains the so-called Riemannian Broyden class, which under the same requirements as before produces a positive definite self-adjoint operator $\mathcal{H}^{Broyden}_{k+1}$ on $T_{x_{k+1}} \mathcal{M}$. The factor $\phi_k$ can either evolve in each iteration as desribed below, or it can also be fixed, i.e. $\phi_k = \phi \in [0,1]$ for all $k$:

$\mathcal{H}^{Broyden}_{k+1} [\cdot] = (1-\phi_k) \, \mathcal{H}^{RBFGS}_{k+1} [\cdot] + \phi_k \, \mathcal{H}^{RDFP}_{k+1} [\cdot]$

where

$\phi_k \in (\phi^{\mathrm{c}}_k, \infty), \; \phi^{\mathrm{c}}_k = \frac{1}{1 - u_k}, \; u_k = \frac{g_{x_{k+1}}(y_k, {\widetilde{\mathcal{H}}^{Broyden}_k}^{-1} [y_k]) g_{x_{k+1}}(s_k, \widetilde{\mathcal{H}}^{Broyden}_k [s_k])}{g_{x_{k+1}}(s_k, y_k)^2}.$

Another Riemannian quasi-Newton update, which does not transfer the positive definiteness but the self-adjointness of the operator $\mathcal{H}^{SR1}_k$, is the generalised Symmetric Rank One update, or SR1 update for short. As the name suggests, this is a rank one operator update, which only ensures the preservation of symmetry, i.e. self-adjointness, of the operator $\mathcal{H}^{SR1}_k$:

$\mathcal{H}^{RSR1}_{k+1} [\cdot] = \widetilde{\mathcal{H}}^{RSR1}_k [\cdot] + \frac{(y_k - \widetilde{\mathcal{H}}^{RSR1}_k [s_k])(y_k - \widetilde{\mathcal{H}}^{RSR1}_k [s_k])^{\flat}[\cdot]}{(y_k - \widetilde{\mathcal{H}}^{RSR1}_k [s_k])^{\flat}[s_k]}.$

By replacing the triple $(\widetilde{\mathcal{H}}^{RSR1}_k, s_k, y_k)$ by the triple $(\widetilde{\mathcal{B}}^{SR1}_k, y_k, s_k)$ we get the inverse SR1 update:  

$\mathcal{B}^{RSR1}_{k+1} [\cdot] = \widetilde{\mathcal{B}}^{RSR1}_k [\cdot] + \frac{(s_k - \widetilde{\mathcal{B}}^{RSR1}_k [y_k])(s_k - \widetilde{\mathcal{B}}^{RSR1}_k [y_k])^{\flat}[\cdot]}{(s_k - \widetilde{\mathcal{B}}^{RSR1}_k [y_k])^{\flat}[y_k]}.$

## Initialization

Initialize $x_0 \in \mathcal{M}$ and let $\mathcal{B}_0 \colon T_{x_0} \mathcal{M} \to T_{x_0} \mathcal{M}$ be a positive definite, self-adjoint operator.

## Iteration

Repeat until a convergence criterion is reached

1. Compute $\eta_k = -\mathcal{B}_k [\operatorname{grad} f (x_k)]$ or solve $\mathcal{H}_k [\eta_k] = -\operatorname{grad} f (x_k)]$.
2. Determine a suitable stepsize $\alpha_k$ along the curve given by $\gamma(\alpha) = R_{x_k}(\alpha \eta_k)$ (e.g. by using the Riemannian Wolfe conditions).
3. Compute $x_{k+1} = R_{x_k}(\alpha_k)$.
4. Define $s_k = T_{x_k, \alpha_k \eta_k}(\alpha_k \eta_k)$ and $y_k = \operatorname{grad} f(x_{k+1}) - T_{x_k, \alpha_k \eta_k}(\operatorname{grad} f(x_k))$.
5. Update $\mathcal{B}_k \text{ or } \mathcal{H}_k \mapsto \mathcal{B}_{k+1} \text{ or } \mathcal{H}_{k+1} \colon T_{x_{k+1}} \mathcal{M} \to T_{x_{k+1}} \mathcal{M}$.

## Result

The result is given by the last computed $x_K$.

## Locking condition

Essential for the positive definiteness of the operators $\mathcal{H}_{k+1}$ or $\mathcal{B}_{k+1}$ for many methods is the fulfilment of 

$g_{x_{k+1}}(s_k, y_k) > 0,$

which is the so-called Riemannian curvature condition. If this condition holds in each iteration, the corresponding update, which uses an isometric vector transport $T^S$, produces a sequence of positive definite self-adjoint operators $\{ \mathcal{H}_k \}_k$ or $\{ \mathcal{B}_k \}_k$. The same association exists, of course, in the Euclidean case, where the fulfilment of the curvature condition $s^{\mathrm{T}}_k y_k$ is ensured by choosing a stepsize $\alpha_k > 0$ in each ietartion that fulfils the Wolfe conditions. The generalisation of the Wolfe conditions for Riemannian manifolds is that the stepsize $\alpha_k > 0$ satisfies 

$f( R_{x_k}(\alpha_k \eta_k)) \leq f(x_k) + c_1 \alpha_k g_{x_k} (\operatorname{grad} f(x_k), \eta_k)$

and 

$\frac{\mathrm{d}}{\mathrm{d}t} f(R_{x_k}(t \; \eta_k)) \vert_{t=\alpha_k} \geq c_2 \frac{\mathrm{d}}{\mathrm{d}t} f(R_{x_k}(t \; \eta_k)) \vert_{t=0}.$

Using the vector transport by differentiated retraction $T^{R}{x, \eta_x}(\xi_x) = \frac{\mathrm{d}}{\mathrm{d}t} R_{x}(\eta_x + t \; \xi_x) \; \vert_{t=0}$, where the retraction $R$ is the associated retraction of the vector transport used in the update formula, the second condition can be rewritten as

$g_{R_{x_k}(\alpha_k \eta_k)} (\operatorname{grad} f(R_{x_k}(\alpha_k \eta_k)), T^{R}{x_k, \alpha_k \eta_k}(\eta_k)) \geq c_2 g_{x_k} (\operatorname{grad} f(x_k), \eta_k).$

Unfortunately, in general, a stepsize $\alpha_k > 0$ that satisfies the Riemannian Wolfe conditions and an isometric vector transport $T^S$ in the update formula does not in general lead to a positive definite update of the operator $\mathcal{H}_k$ or $\mathcal{B}_k$. This is because the Wolfe conditions use vector transport by differentiated retraction $T^R$, which is generally not the same as isometric vetor transport $T^S$ used in the update. However, in order to create a positive definite operator in each iteration, the so-called locking condition was introduced, which requires that the isometric vector transport $T^S$ and its associate retraction $R$ fulfil the following condition

$T^{S}{x, \xi_x}(\xi_x) = \beta T^{R}{x, \xi_x}(\xi_x), \quad \beta = \frac{\lVert \xi_x \rVert_x}{\lVert T^{R}{x, \xi_x}(\xi_x) \rVert_{R_{x}(\xi_x)}}.$

where $T^R$ is again the vector transport by differentiated retraction. With the requirement that the isometric vector transport $T^S$ and its associared retraction $R$ satisfies the locking condition and using the tangent vector 

$y_k = {\beta_k}^{-1} \operatorname{grad} f(x_{k+1}) - T^{S}{x_k, \alpha_k \eta_k}(\operatorname{grad} f(x_k)),$

where 

$\beta_k = \frac{\lVert \alpha_k \eta_k \rVert_{x_k}}{\lVert \lVert T^{R}{x_k, \alpha_k \eta_k}(\alpha_k \eta_k) \rVert_{x_{k+1}}},$

in the update, it can be shown that choosing a stepsize $\alpha_k > 0$ that satisfies the Riemannian wolfe conditions leads to the fulfilment of the Riemannian curvature condition, which in turn implies that the operator generated by the updates is positive definite. 

## Cautious BFGS

As in the Euclidean case, the Riemannian BFGS method is not globally convergent for general objective functions. To mitigate this obstacle, the cautious update has been generalised to Riemannian manifolds. For that, in each iteration, based on a decision rule, either the usual update is used or the current operator is transported into the upcoming tangential, which in turn means that the update consists of the vector transport of B only. In summary, this can be expressed as follows:

$\mathcal{B}^{CRBFGS}_{k+1} = \begin{cases} \text{using \cref{RiemannianInverseBFGSFormula}}, & \; \frac{g_{x_{k+1}}(y_k,s_k)}{\lVert s_k \rVert^{2}_{x_{k+1}}} \geq \theta(\lVert \operatorname{grad} f(x_k) \rVert_{x_k}), \\ \widetilde{\mathcal{B}}^{CRBFGS}_k, & \; \text{otherwise}, \end{cases}$

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

