# [Steihaug-Toint Truncated Conjugate-Gradient Method](@id tCG)

The aim is to solve the trust-region subproblem

```math
\operatorname*{arg\,min}_{\eta \in T_{x}M} m_{x}(\eta) = \langle \nabla F(x), \eta \rangle_{x} + \frac{1}{2} \langle \operatorname{Hess}[F](\eta)_ {x}, \eta \rangle_{x}
```
```math
\text{s.t.} \; \langle \eta, \eta \rangle_{x} \leqq {\Delta}^2
```

on a manifold by using the Steihaug-Toint truncated conjugate-gradient method.
All terms involving the trust-region radius use an inner product w.r.t. the
preconditioner; this is because the iterates grow in length w.r.t. the
preconditioner, guaranteeing that we do not re-enter the trust-region.

## Initialization

Initialize $\eta_0 = \eta$ if using randomized approach else
$\eta_0$`=`[`zeroTVector`](@ref)`(M,x)`, $r_0 = \nabla F(x)$,
$z_0 = \operatorname{P}(r_0)$, $\delta_0 = z_0$ and $k=0$

## Iteration

Repeat until a convergence criterion is reached

1. Set $\kappa = \langle \delta_k, \operatorname{Hess}[F] (\delta_k)_ {x} \rangle_{x}$,
    $\alpha =\frac{\langle r_k, z_k \rangle_{x}}{\kappa}$ and
    $\langle \eta_k, \eta_k \rangle_{x}^{* } = \langle \eta_k, \operatorname{P}(\eta_k) \rangle_{x} -
    2\alpha \langle \eta_k, \operatorname{P}(\delta_k) \rangle_{x} +  {\alpha}^2
    \langle \delta_k, \operatorname{P}(\delta_k) \rangle_{x}$.
2. If $\kappa \leqq 0$ or $\langle \eta_k, \eta_k \rangle_{x}^{* } \geqq {\Delta}^2$
    return $\eta_{k+1} = \eta_k + \tau_k \delta_k$ and stop.
3. Set $\eta_{k}^{* }= \eta_k + \alpha_k \delta_k$,
    if $\langle \eta_k, \eta_k \rangle_{x} + \frac{1}{2} \langle \eta_k,
    \operatorname{Hess}[F] (\eta_k)_ {x} \rangle_{x} \leqq \langle \eta_k^{* },
    \eta_k^{* } \rangle_{x} + \frac{1}{2} \langle \eta_k^{* },
    \operatorname{Hess}[F] (\eta_k)_ {x} \rangle_{x}$ set $\eta_{k+1} = \eta_k$
    else set $\eta_{k+1} = \eta_{k}^{* }$.
4. Set $r_{k+1} = r_k + \alpha_k \operatorname{Hess}[F] (\delta_k)_ {x}$,
    $z_{k+1} = \operatorname{P}(r_{k+1})$,
    $\beta = \frac{\langle r_{k+1}, z_{k+1} \rangle_{x}}{\langle r_k, z_k
    \rangle_{x}}$ and $\delta_{k+1} = -z_{k+1} + \beta \delta_k$.
5. Set $k=k+1$.

## Result

The result is given by the last computed $η_K$.

## Remarks
1. $\operatorname{P}(\cdot)$ denotes the preconditioner. It is required if a
    randomized approach is used i.e. using a random tangent vector
    $\eta$`=`[`randomTVector`](@ref)`(M,x)` as initial vector. The idea behind
    it is to avoid saddle points.
2. To step number 2: Obtain $\tau_k$ the positive root of
    $\left\lVert \operatorname{P}(\eta_k + \tau \delta_k) \right\rVert_x = \Delta$
    what becomes after the conversion of the equation to
    $\tau = \frac{-\langle \eta_k, \operatorname{P}(\delta_k) \rangle_{x} +
    \sqrt{\langle \eta_k, \operatorname{P}(\delta_k) \rangle_{x}^{2} +
    \langle \delta_k, \operatorname{P}(\delta_k) \rangle_{x} ( \Delta^2 -
    \langle \eta_k, \operatorname{P}(\eta_k) \rangle_{x})}}
    {\langle \delta_k, \operatorname{P}(\delta_k) \rangle_{x}}$.

## Interface

```@docs
  truncatedConjugateGradient
```

## Options

```@docs
TruncatedConjugateGradientOptions
```
