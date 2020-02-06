# [Steihaug-Toint Truncated Conjugate-Gradient Method](@id tCG)

The aim is to solve the trust-region subproblem

```math
\operatorname*{arg\,min}_{\eta  ∈  T_{x}\mathcal{M}} m_{x}(\eta) = F(x) +
\langle \nabla F(x), \eta \rangle_{x} + \frac{1}{2} \langle
\operatorname{Hess}[F](\eta)_ {x}, \eta \rangle_{x}
```
```math
\text{s.t.} \; \langle \eta, \eta \rangle_{x} \leq {\Delta}^2
```

on a manifold by using the Steihaug-Toint truncated conjugate-gradient method.
All terms involving the trust-region radius use an inner product w.r.t. the
preconditioner; this is because the iterates grow in length w.r.t. the
preconditioner, guaranteeing that we do not re-enter the trust-region.

## Initialization

Initialize $\eta_0 = \eta$ if using randomized approach and
$\eta$ the zero tangent vector otherwise, $r_0 = \nabla F(x)$,
$z_0 = \operatorname{P}(r_0)$, $\delta_0 = z_0$ and $k=0$

## Iteration

Repeat until a convergence criterion is reached

1. Set $\kappa = \langle \delta_k, \operatorname{Hess}[F] (\delta_k)_ {x} \rangle_{x}$,
    $\alpha =\frac{\langle r_k, z_k \rangle_{x}}{\kappa}$ and
    $\langle \eta_k, \eta_k \rangle_{x}^{* } = \langle \eta_k, \operatorname{P}(\eta_k) \rangle_{x} +
    2\alpha \langle \eta_k, \operatorname{P}(\delta_k) \rangle_{x} +  {\alpha}^2
    \langle \delta_k, \operatorname{P}(\delta_k) \rangle_{x}$.
2. If $\kappa \leqq 0$ or $\langle \eta_k, \eta_k \rangle_{x}^{* } \geqq {\Delta}^2$
    return $\eta_{k+1} = \eta_k + \tau \delta_k$ and stop.
3. Set $\eta_{k}^{* }= \eta_k + \alpha \delta_k$, if
    $\langle \eta_k, \eta_k \rangle_{x} + \frac{1}{2} \langle \eta_k,
    \operatorname{Hess}[F] (\eta_k)_ {x} \rangle_{x} \leqq \langle \eta_k^{* },
    \eta_k^{* } \rangle_{x} + \frac{1}{2} \langle \eta_k^{* },
    \operatorname{Hess}[F] (\eta_k)_ {x} \rangle_{x}$
    set $\eta_{k+1} = \eta_k$ else set $\eta_{k+1} = \eta_{k}^{* }$.
4. Set $r_{k+1} = r_k + \alpha \operatorname{Hess}[F] (\delta_k)_ {x}$,
     $z_{k+1} = \operatorname{P}(r_{k+1})$,
    $\beta = \frac{\langle r_{k+1},  z_{k+1} \rangle_{x}}{\langle r_k, z_k
    \rangle_{x}}$ and $\delta_{k+1} = -z_{k+1} + \beta \delta_k$.
5. Set $k=k+1$.

## Result

The result is given by the last computed $η_k$.

## Remarks
The $\operatorname{P}(\cdot)$ denotes the symmetric, positive deﬁnite
preconditioner. It is required if a randomized approach is used i.e. using
a random tangent vector $\eta$ as initial
vector. The idea behind it is to avoid saddle points. Preconditioning is
simply a rescaling of the variables and thus a redeﬁnition of the shape of
the trust region. Ideally $\operatorname{P}(\cdot)$ is a cheap, positive
approximation of the inverse of the Hessian of $F$ at $x$. On
default, the preconditioner is just the identity.

To step number 2: Obtain $\tau$ from the positive root of
$\left\lVert \eta_k + \tau \delta_k \right\rVert_{\operatorname{P}, x} = \Delta$
what becomes after the conversion of the equation to
````math
 \tau = \frac{-\langle \eta_k, \operatorname{P}(\delta_k) \rangle_{x} +
 \sqrt{\langle \eta_k, \operatorname{P}(\delta_k) \rangle_{x}^{2} +
 \langle \delta_k, \operatorname{P}(\delta_k) \rangle_{x} ( \Delta^2 -
 \langle \eta_k, \operatorname{P}(\eta_k) \rangle_{x})}}
 {\langle \delta_k, \operatorname{P}(\delta_k) \rangle_{x}}.
````
It can occur that $\langle \delta_k, \operatorname{Hess}[F] (\delta_k)_ {x} \rangle_{x}
= \kappa \leqq 0$ at iteration $k$. In this case, the model is not strictly
convex, and the stepsize $\alpha =\frac{\langle r_k, z_k \rangle_{x}}
{\kappa}$ computed in step 1. does not give a reduction in the modelfunction
$m_{x}(\cdot)$. Indeed, $m_{x}(\cdot)$ is unbounded from below along the
line $\eta_k + \alpha \delta_k$. If our aim is to minimize the model within
the trust-region, it makes far more sense to reduce $m_{x}(\cdot)$ along
$\eta_k + \alpha \delta_k$ as much as we can while staying within the
trust-region, and this means moving to the trust-region boundary along this
line. Thus when $\kappa \leqq 0$ at iteration k, we replace $\alpha =
\frac{\langle r_k, z_k \rangle_{x}}{\kappa}$ with $\tau$ described as above.
The other possibility is that $\eta_{k+1}$ would lie outside the trust-region at
iteration k (i.e. $\langle \eta_k, \eta_k \rangle_{x}^{* }
\geqq {\Delta}^2$ what can be identified with the norm of $\eta_{k+1}$). In
particular, when $\operatorname{Hess}[F] (\cdot)_ {x}$ is positive deﬁnite
and $\eta_{k+1}$ lies outside the trust region, the solution to the
trust-region problem must lie on the trust-region boundary. Thus, there
is no reason to continue with the conjugate gradient iteration, as it
stands, as subsequent iterates will move further outside the trust-region
boundary. A sensible strategy, just as in the case considered above, is to
move to the trust-region boundary by ﬁnding $\tau$.

## Interface

```@docs
  truncatedConjugateGradient
```

## Options

```@docs
TruncatedConjugateGradientOptions
```

## Additional Stopping Criteria

```@docs
stopIfResidualIsReducedByPower
```
```@docs
stopIfResidualIsReducedByFactor
```
```@docs
stopWhenTrustRegionIsExceeded
```
```@docs
stopWhenCurvatureIsNegative
```
