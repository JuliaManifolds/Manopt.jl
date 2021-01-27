# [The Riemannian Trust-Regions Solver](@id trust_regions)

The aim is to solve an optimization problem on a manifold

```math
\operatorname*{min}_{x  ∈  \mathcal{M}} F(x)
```

by using the Riemannian trust-regions solver. It is number one choice for smooth
optimization. This trust-region method uses the Steihaug-Toint truncated
conjugate-gradient method [`truncated_conjugate_gradient_descent`](@ref)
to solve the inner minimization problem called the
trust-regions subproblem. This inner solve can be preconditioned by providing
a preconditioner (symmetric and positive deﬁnite, an approximation of the
inverse of the Hessian of $F$). If no Hessian of the cost function $F$ is
provided, a standard approximation of the Hessian based on the gradient
$\operatorname{grad}F$ with [`approxHessianFD`](@ref) will be computed.

## Initialization

Initialize $x_0 = x$ with an initial point $x$ on the manifold. It can be
given by the caller or set randomly. Set the initial trust-region radius
$\Delta =\frac{1}{8} \bar{\Delta}$ where $\bar{\Delta}$ is the maximum radius
the trust-region can have. Usually one uses
the root of the manifold dimension $\operatorname{dim}(\mathcal{M})$.
For accepting the next iterate and evaluating the new trust-region radius one
needs an accept/reject threshold $\rho'  ∈  [0,\frac{1}{4})$, which is
$\rho' = 0.1$ on default. Set $k=0$.

## Iteration

Repeat until a convergence criterion is reached

1. Set $\eta$ as a random tangent vector if using randomized approach. Else
    set $\eta$ as the zero vector in the tangential space $T_{x_k}\mathcal{M}$.
2. Set $\eta^{* }$ as the solution of the trust-region subproblem, computed by
    the tcg-method with $\eta$ as initial vector.
3. If using randomized approach compare $\eta^{* }$ with the Cauchy point
    $\eta_{c}^{* } = -\tau_{c} \frac{\Delta}{\operatorname{norm}(\operatorname{Grad}[f] (x_k))} \operatorname{Grad}[F] (x_k)$ by the model function $m_{x_k}(\cdot)$. If the
    model decrease is larger by using the Cauchy point, set
    $\eta^{* } = \eta_{c}^{* }$.
4. Set ${x}^{* } = \operatorname{Retr}_{x_k}(\eta^{* })$.
5. Set $\rho = \frac{F(x_k)-F({x}^{* })}{m_{x_k}(\eta)-m_{x_k}(\eta^{* })}$, where
    $m_{x_k}(\cdot)$ describes the quadratic model function.
6. Update the trust-region radius:
    $\Delta = \begin{cases} \frac{1}{4} \Delta & \rho < \frac{1}{4} \,
    \text{or} \, m_{x_k}(\eta)-m_{x_k}(\eta^{* }) \leq 0 \, \text{or}  \,
    \rho = \pm  ∈ fty , \\ \operatorname{min}(2 \Delta, \bar{\Delta}) &
    \rho > \frac{3}{4} \, \text{and the tcg-method stopped because of negative
    curvature or exceeding the trust-region}, \\ \Delta & \, \text{otherwise.}
    \end{cases}$
7. If $m_{x_k}(\eta)-m_{x_k}(\eta^{* }) \geq 0$ and $\rho > \rho'$ set
    $x_k = {x}^{* }$.
8. Set $k = k+1$.

## Result

The result is given by the last computed $x_k$.

## Remarks

To the Initialization: A random point on the manifold.

To step number 1: Using randomized approach means using a random tangent
vector as initial vector for the approximal solve of the trust-regions
subproblem. If this is the case, keep in mind that the vector must be in the
trust-region radius. This is achieved by multiplying
`η` by `sqrt(4,eps(Float64))` as long as
its norm is greater than the current trust-region radius $\Delta$.
For not using randomized approach, one can get the zero tangent vector.

To step number 2: Obtain $\eta^{* }$ by (approximately) solving the
trust-regions subproblem

```math
\operatorname*{arg\,min}_{\eta  ∈  T_{x_k}\mathcal{M}} m_{x_k}(\eta) = F(x_k) +
\langle \operatorname{grad}F(x_k), \eta \rangle_{x_k} + \frac{1}{2} \langle
\operatorname{Hess}[F](\eta)_ {x_k}, \eta \rangle_{x_k}
```

```math
\text{s.t.} \; \langle \eta, \eta \rangle_{x_k} \leq {\Delta}^2
```

with the Steihaug-Toint truncated conjugate-gradient (tcg) method. The problem
as well as the solution method is described in the
[`truncated_conjugate_gradient_descent`](@ref).

To step number 3: If using a random tangent vector as an initial vector, compare
the result of the tcg-method with the Cauchy point. Convergence proofs assume
that one achieves at least (a fraction of) the reduction of the Cauchy point.
The idea is to go in the direction of the gradient to an optimal point. This
can be on the edge, but also before.
The parameter $\tau_{c}$ for the optimal length is defined by

```math
\tau_{c} = \begin{cases} 1 & \langle \operatorname{Grad}[F] (x_k), \,
\operatorname{Hess}[F] (\eta_k)_ {x_k}\rangle_{x_k} \leq 0 , \\
\operatorname{min}(\frac{{\operatorname{norm}(\operatorname{Grad}[F] (x_k))}^3}
{\Delta \langle \operatorname{Grad}[F] (x_k), \,
\operatorname{Hess}[F] (\eta_k)_ {x_k}\rangle_{x_k}}, 1) & \, \text{otherwise.}
\end{cases}
```

To check the model decrease one compares
$m_{x_k}(\eta_{c}^{* }) = F(x_k) + \langle \eta_{c}^{* },
\operatorname{Grad}[F] (x_k)\rangle_{x_k} + \frac{1}{2}\langle \eta_{c}^{* },
\operatorname{Hess}[F] (\eta_{c}^{* })_ {x_k}\rangle_{x_k}$ with
$m_{x_k}(\eta^{* }) = F(x_k) + \langle \eta^{* },
\operatorname{Grad}[F] (x_k)\rangle_{x_k} + \frac{1}{2}\langle \eta^{* },
\operatorname{Hess}[F] (\eta^{* })_ {x_k}\rangle_{x_k}$.
If $m_{x_k}(\eta_{c}^{* }) < m_{x_k}(\eta^{* })$ then is
$m_{x_k}(\eta_{c}^{* })$ the better choice.

To step number 4: $\operatorname{Retr}_{x_k}(\cdot)$ denotes the retraction, a
mapping $\operatorname{Retr}_{x_k}:T_{x_k}\mathcal{M} \rightarrow \mathcal{M}$
wich approximates the exponential map. In some cases it is cheaper to use this
instead of the exponential.

To step number 6: One knows that the [`truncated_conjugate_gradient_descent`](@ref) algorithm stopped for
these reasons when the stopping criteria [`StopWhenCurvatureIsNegative`](@ref),
[`StopWhenTrustRegionIsExceeded`](@ref) are activated.

To step number 7: The last step is to decide if the new point ${x}^{* }$ is
accepted.

## Interface

```@docs
trust_regions
trust_regions!
```

## Options

```@docs
AbstractHessianOptions
TrustRegionsOptions
```

## Approximation of the Hessian

```@docs
approxHessianFD
```
