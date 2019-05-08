
```@meta
CurrentModule = Manopt
```

```@docs
Manopt.Manopt
```

`Manopt.jl` provides a framework for optimization on manifolds.
Based on [Manopt](https://manopt.org) and
[MVIRT](https://ronnybergmann.net/mvirt/), both implemented in Matlab,
this toolbox aims to provide an easy access to optimization methods on manifolds
for [Julia](https://julialang.org), including example data and visualization methods.

The package provides a properly typed system for Manifolds, points
on manifolds and tangent vectors, see [Riemannian manifolds](@ref RiemannianManifolds)
for an introduction to the types and the available manifolds.

All optimization tasks are described by a [`Problem`](@ref) and corresponding
[`Options`](@ref), see the page about [plans](@ref planSection).

Based on these, the solvers are implemented, see [Solvers](@ref Solvers) for an
introduction and a list of algorithms.

## Notation

During this documentation, we distinguish variables `x` and mathematical symbols $x$ from time to time.
| Symbols | used for
|:---|:---
$\mathcal M, \mathcal N$ | a manifold
$d,d_1,\ldots,d_n$ | dimension(s) of a manifold
$x,y,z,x_1,\ldots,x_n$ | points on a manifold
$T_x\mathcal M$ | the tangent space to $x\in\mathcal M$
$\xi,\nu,\eta,\xi_1,\ldots,\xi_n$ | tangent vectors, might be extended by the base point, i.e. $\xi_x$
$\log_xy$ | logarithmic map
$\exp_x\xi$ | exponential map
$g(t; x,y)$ | geodesic connecting $x,y\in\mathcal M$ with $t\in [0,1]$
$\langle \cdot, \cdot\rangle_x$ | Riemannian inner product on $T_x\mathcal M$
$\operatorname{PT}_{x\to y}(\xi$ | parallel transport of $\xi\inT_x\mathcal M$ from $x$ to $y$ along $g(\cdot;x,y)$

## Literature

[AMS08] P.-A. Absil, R. Mahony and R. Sepulchre, Optimization Algorithms on
Matrix Manifolds, Princeton University Press, 2008,
[open access](http://press.princeton.edu/chapters/absil/)
