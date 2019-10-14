# [The Riemannian Trust-Regions Solver](@id trustRegions)

The aim is to solve an optimization problem on a manifold

```math
min_{x \in \mathcal{M}} f(x)
```

by using the Riemannian trust-regions solver. It is number one choice for smooth
optimization. This trust-region method uses the Steihaug-Toint truncated
conjugate-gradient method to solve the inner minimization problems called the
trust-regions subproblem. This inner solve can be preconditioned: simply provide
a preconditioner.

## Initialization

Initialize $x_0 = x$ if an initial point $x$ is given by the caller or set
$x_0 = \operatorname{randomMPoint}(\mathcal{M})$, $\Delta =\frac{1}{2} \bar{\Delta}$
where $\bar{\Delta}$ is the maximum radius the trust-region can have. Usually
one uses the root of the manifold dimension.

## Iteration

Repeat until a convergence criterion is reached

1. If using randomized approach (i.e. using a random tangent vector as initial
    vector for the approximal solve of the trust-regions subproblem), set
    $\eta = \operatorname{randomTVector}(\mathcal{M}, x)$ and multiply it by
    $\sqrt{\sqrt{\operatorname{eps}(Float64)}}$ as long as its norm is greater than
    the current trust-regions radius $\Delta$. If not, set $\eta = \operatorname{zeroTVector}(\mathcal{M}, x)$.
2. Obtain $\eta_k$ by (approximately) solving the trust-regions subproblem.
    The problem as well as the solution method is described in the
    [`truncatedConjugateGradient`](@ref).
3. If using random tangent vector as initial vector, compare result with the
    Cauchy point. Convergence proofs assume that we achieve at least (a fraction
    of) the reduction of the Cauchy point.
4. Set ${x}^{* } = \operatorname{retraction}(\mathcal{M}, x_k, \eta_k)$
5. Set $\rho = \frac{f(x_k)-f({x}^{* })}{m_{k}(x_k)-m_{k}({x}^{* })}$, where $f$
    is the cost function and
    $m_{k}({x}^{* })=m_{k}(x_k)+\langle\eta_k,\operatorname{Grad}[f] (x_k)\rangle_{x_k}
    +\frac{1}{2}\langle\eta_k,\operatorname{Hess}[f] (\eta_k)_ {x_k}\rangle_{x_k}$
    describes the quadratic model function with $m_{k}(x_k) = f(x_k)$.
6. Then the trust-region radius is updated. If $\rho < \frac{1}{4}& or
    $m_{k}(x_k)-m_{k}({x}^{* }) \geq 0$ or $\rho = \pm \infty& set
    $\Delta =\frac{1}{4} \Delta$.

## Result

The result is given by the last computed $x_k$.

## Interface

```@docs
trustRegions
```

## Options

```@docs
TrustRegionOptions
```
