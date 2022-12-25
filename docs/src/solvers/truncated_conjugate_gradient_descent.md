# [Steihaug-Toint Truncated Conjugate-Gradient Method](@id tCG)

The aim is to solve the trust-region subproblem

```math
\operatorname*{arg\,min}_{η  ∈  T_{x}\mathcal{M}} m_{x}(η) = F(x) +
⟨\operatorname{grad}F(x), η⟩_{x} + \frac{1}{2} ⟨
\mathcal{H}[η], η⟩_{x}
```

```math
\text{s.t.} \; ⟨η, η⟩_{x} \leq {Δ}^2
```

on a manifold by using the Steihaug-Toint truncated conjugate-gradient method, 
abbreviated tCG-method.
All terms involving the trust-region radius use an inner product w.r.t. the
preconditioner; this is because the iterates grow in length w.r.t. the
preconditioner, guaranteeing that we do not re-enter the trust-region.

## Initialization

Initialize ``η_0 = η`` if using randomized approach and
``η`` the zero tangent vector otherwise, ``r_0 = \operatorname{grad}F(x)``,
``z_0 = \operatorname{P}(r_0)``, ``δ_0 = z_0`` and ``k=0``

## Iteration

Repeat until a convergence criterion is reached

1. Set ``α =\frac{⟨r_k, z_k⟩_x}{⟨δ_k, \mathcal{H}[δ_k]⟩_x}`` and
    ``⟨η_k, η_k⟩_{x}^* = ⟨η_k, \operatorname{P}(η_k)⟩_x +
    2α ⟨η_k, \operatorname{P}(δ_k)⟩_{x} +  {α}^2
    ⟨δ_k, \operatorname{P}(δ_k)⟩_{x}``.
2. If ``⟨δ_k, \mathcal{H}[δ_k]⟩_x ≤ 0`` or ``⟨η_k, η_k⟩_x^* ≥ Δ^2``
    return ``η_{k+1} = η_k + τ δ_k`` and stop.
3. Set ``η_{k}^*= η_k + α δ_k``, if
    ``⟨η_k, η_k⟩_{x} + \frac{1}{2} ⟨η_k,
    \operatorname{Hess}[F] (η_k)_{x}⟩_{x} ≤ ⟨η_k^*,
    η_k^*⟩_{x} + \frac{1}{2} ⟨η_k^*,
    \operatorname{Hess}[F] (η_k)_ {x}⟩_{x}``
    set ``η_{k+1} = η_k`` else set ``η_{k+1} = η_{k}^*``.
4. Set ``r_{k+1} = r_k + α \mathcal{H}[δ_k]``,
     ``z_{k+1} = \operatorname{P}(r_{k+1})``,
     ``β = \frac{⟨r_{k+1},  z_{k+1}⟩_{x}}{⟨r_k, z_k
   ⟩_{x}}`` and ``δ_{k+1} = -z_{k+1} + β δ_k``.
5. Set ``k=k+1``.

## Result

The result is given by the last computed ``η_k``.

## Remarks

The ``\operatorname{P}(⋅)`` denotes the symmetric, positive deﬁnite
preconditioner. It is required if a randomized approach is used i.e. using
a random tangent vector ``η_0`` as the initial
vector. The idea behind it is to avoid saddle points. Preconditioning is
simply a rescaling of the variables and thus a redefinition of the shape of
the trust region. Ideally ``\operatorname{P}(⋅)`` is a cheap, positive
approximation of the inverse of the Hessian of ``F`` at ``x``. On
default, the preconditioner is just the identity.

To step number 2: obtain ``τ`` from the positive root of
``\left\lVert η_k + τ δ_k \right\rVert_{\operatorname{P}, x} = Δ``
what becomes after the conversion of the equation to

````math
 τ = \frac{-⟨η_k, \operatorname{P}(δ_k)⟩_{x} +
 \sqrt{⟨η_k, \operatorname{P}(δ_k)⟩_{x}^{2} +
 ⟨δ_k, \operatorname{P}(δ_k)⟩_{x} ( Δ^2 -
 ⟨η_k, \operatorname{P}(η_k)⟩_{x})}}
 {⟨δ_k, \operatorname{P}(δ_k)⟩_{x}}.
````

It can occur that ``⟨δ_k, \operatorname{Hess}[F] (δ_k)_{x}⟩_{x}
= κ ≤ 0`` at iteration ``k``. In this case, the model is not strictly
convex, and the stepsize ``α =\frac{⟨r_k, z_k⟩_{x}}
{κ}`` computed in step 1. does not give a reduction in the model function
``m_x(⋅)``. Indeed, ``m_x(⋅)`` is unbounded from below along the
line ``η_k + α δ_k``. If our aim is to minimize the model within
the trust-region, it makes far more sense to reduce ``m_x(⋅)`` along
``η_k + α δ_k`` as much as we can while staying within the
trust-region, and this means moving to the trust-region boundary along this
line. Thus, when ``κ ≤ 0`` at iteration k, we replace
``α = \frac{⟨r_k, z_k⟩_{x}}{κ}`` with ``τ`` described as above.
The other possibility is that ``η_{k+1}`` would lie outside the trust-region at
iteration k (i.e. ``⟨η_k, η_k⟩_{x}^{* } ≥ {Δ}^2``
that can be identified with the norm of ``η_{k+1}``). In
particular, when ``\operatorname{Hess}[F] (⋅)_{x}`` is positive deﬁnite
and ``η_{k+1}`` lies outside the trust region, the solution to the
trust-region problem must lie on the trust-region boundary. Thus, there
is no reason to continue with the conjugate gradient iteration, as it
stands, as subsequent iterates will move further outside the trust-region
boundary. A sensible strategy, just as in the case considered above, is to
move to the trust-region boundary by finding ``τ``.


Although it is virtually impossible in practice to know how many iterations are
necessary to provide a good estimate ``η_{k}`` of the trust-region subproblem, 
the method stops after a certain number of iterations, which is realised by 
[`StopAfterIteration`](@ref). In order to increase the convergence rate of 
the underlying trust-region method, see 
[`trust_regions`](@ref), a typical stopping criterion 
is to stop as soon as an iteration ``k`` is reached for which

```math
  \Vert r_k \Vert_x \leqq \Vert r_0 \Vert_x \min \left( \Vert r_0 \Vert^{θ}_x, κ \right 
```

holds, where ``0 < κ < 1`` and ``θ > 0`` are chosen in advance. This is 
realized in this method by [`StopIfResidualIsReducedByFactorOrPower`](@ref). 
It can be shown shown that under appropriate conditions the iterates ``x_k``
of the underlying trust-region method converge to nondegenerate critical
points with an order of convergence of at least ``\min \left( θ + 1, 2 \right)``, 
see [[Absil, Mahony, Sepulchre, 2008](#AbsilMahonySepulchre2008)]. 
The method also aborts if the curvature of the model is negative, i.e. if 
``\langle \delta_k, \mathcal{H}[δ_k] \rangle_x \leqq 0``, which is realised by 
[`StopWhenCurvatureIsNegative`](@ref). If the next possible approximate 
solution ``η_{k}^{*}`` calculated in iteration ``k`` lies outside the 
trust region, i.e. if ``\lVert η_{k}^{*} \rVert_x \geq Δ``, then the method 
aborts, which is realised by [`StopWhenTrustRegionIsExceeded`](@ref). 
Furthermore, the method aborts if the new model value evaluated at ``η_{k}^{*}`` 
is greater than the previous model value evaluated at ``η_{k}``, which 
is realised by [`StopWhenModelIncreased`](@ref). 


## Interface

```@docs
  truncated_conjugate_gradient_descent
  truncated_conjugate_gradient_descent!
```

## State

```@docs
TruncatedConjugateGradientState
```

## Additional Stopping Criteria

```@docs
StopWhenResidualReducedByFactorOrPower
StopWhenTrustRegionIsExceeded
StopWhenCurvatureIsNegative
StopWhenModelIncreased
```

## Literature 

```@raw html
<ul>
<li id="AbsilMahonySepulchre2008">[<a>Absil, Mahony, Sepulchre, 2008</a>]
  Absil, Pierre-Antoine and Mahony, Robert and Sepulchre, Rodolphe: 
  <emph> Optimization Algorithms on Matrix Manifolds </emph>
  Mathematics of Computation - Math. Comput., Volume 78.
  doi: <a href="https://doi.org/10.1515/9781400830244">10.1515/9781400830244</a>,
</li>
</ul>
```