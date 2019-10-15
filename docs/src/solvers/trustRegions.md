# [The Riemannian Trust-Regions Solver](@id trustRegions)

The aim is to solve an optimization problem on a manifold

```math
min_{x \in \mathcal{M}} F(x)
```

by using the Riemannian trust-regions solver. It is number one choice for smooth
optimization. This trust-region method uses the Steihaug-Toint truncated
conjugate-gradient method to solve the inner minimization problems called the
trust-regions subproblem. This inner solve can be preconditioned: simply provide
a preconditioner.

## Initialization

Initialize $x_0 = x$ which is an initial point $x$ on the manifold. It can be
given by the caller or set $x_0 = \operatorname{randomMPoint}(\mathcal{M})$.
Set the initial trust-region radius $\Delta =\frac{1}{8} \bar{\Delta}$ where
$\bar{\Delta}$ is the maximum radius the trust-region can have. Usually one uses
the root of the manifold dimension $\operatorname{dim}(\mathcal{M})$.
For accepting the next iterate and evaluating the new trust-region radius we
need a accept/reject threshold $rho_{prime} \in [0,\frac{1}{4})$, which is  
$\rho_{prime} = 0.1$ on default. Set $k=0$.

## Iteration

Repeat until a convergence criterion is reached

1. If using randomized approach (i.e. using a random tangent vector as initial
    vector for the approximal solve of the trust-regions subproblem), set
    $\eta = \operatorname{randomTVector}(\mathcal{M}, x)$ and multiply it by
    $\sqrt{\sqrt{\operatorname{eps}(Float64)}}$ as long as its norm is greater than
    the current trust-regions radius $\Delta$. If not, set $\eta = \operatorname{zeroTVector}(\mathcal{M}, x)$.
2. Obtain $\eta_k$ by (approximately) solving the trust-regions subproblem with
    the Steihaug-Toint truncated conjugate-gradient method. The problem as well
    as the solution method is described in the
    [`truncatedConjugateGradient`](@ref).
3. If using random tangent vector as initial vector, compare result with the
    Cauchy point. Convergence proofs assume that we achieve at least (a fraction
    of) the reduction of the Cauchy point. The Cauchy point is defined as
    $\eta_{k}^{c} = -\tau_{k}^{c} \frac{\Delta}{\operatorname{norm}(\operatorname{Grad}[f] (x_k))} \operatorname{Grad}[f] (x_k)$ where
    $\tau_{k}^{c} = \begin{cases} 1 & \langle \operatorname{Grad}[f] (x_k), \, \operatorname{Hess}[f] (\eta_k)_ {x_k}\rangle_{x_k} \leqq 0 , \\ \operatorname{min}(\frac{{\operatorname{norm}(\operatorname{Grad}[f] (x_k))}^3}{\Delta \langle \operatorname{Grad}[f] (x_k), \, \operatorname{Hess}[f] (\eta_k)_ {x_k}\rangle_{x_k}}, 1) & \, \text{sonst} \end{cases}$. If
    $f(x_k) + \langle \eta_{k}^{c},\operatorname{Grad}[f] (x_k)\rangle_{x_k}
    +\frac{1}{2}\langle \eta_{k}^{c}, \operatorname{Hess}[f] (\eta_{k}^{c})_ {x_k}\rangle_{x_k}
    < f(x_k) + \langle\ eta_k, \operatorname{Grad}[f] (x_k) \rangle_{x_k}
    +\frac{1}{2} \langle \eta_k, \operatorname{Hess}[f] (\eta_k)_ {x_k} \rangle_{x_k}$
    replace the update vector $\eta_k$ with the cauchy point $\eta_{k}^{c}$.
4. Set ${x}^{* } = \operatorname{retraction}(\mathcal{M}, x_k, \eta_k)$
5. Set $\rho = \frac{f(x_k)-f({x}^{* })}{m_{k}(x_k)-m_{k}({x}^{* })}$, where $f$
    is the cost function and
    $m_{k}({x}^{* })=m_{k}(x_k)+\langle\eta_k,\operatorname{Grad}[f] (x_k)\rangle_{x_k}
    +\frac{1}{2}\langle\eta_k,\operatorname{Hess}[f] (\eta_k)_ {x_k}\rangle_{x_k}$
    describes the quadratic model function with $m_{k}(x_k) = f(x_k)$.
6. Then the trust-region radius is updated. If $\rho < \frac{1}{4}& or
    $m_{k}(x_k)-m_{k}({x}^{* }) \leqq 0$ or $\rho = \pm \infty& set
    $\Delta =\frac{1}{4} \Delta$. Else if $\rho > \frac{3}{4}$ and
    the Steihaug-Toint truncated conjugate-gradient method stopped because of
    a negative curvature or exceeding the trust-region ($\operatorname{norm}
    (\eta_k) \geqq \Delta$) set $\Delta = \operatorname{min}(2 \Delta, \bar{\Delta})$.
    If none of the two cases applies, the trust-region radius $\Delta$ remains
    unchanged.
7. The last step is to decide if the new point ${x}^{* }$ ist accepted. If
    $m_{k}(x_k)-m_{k}({x}^{* }) \geqq 0$ and $\rho > \rho_{prime}$ set
    $x_k = {x}^{* }$.
8. Set $k = k+1$.


## Result

The result is given by the last computed $x_k$.

## Interface

```@docs
trustRegions
```

## Options

```@docs
TrustRegionsOptions
```
