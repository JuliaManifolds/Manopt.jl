
```@meta
CurrentModule = Manopt
```

```@docs
Manopt.Manopt
```
For a function $f\colon\mathcal M \to \mathbb R$ defined on a [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold) $\mathcal M$ we aim to solve

$\operatorname*{argmin}_{x\in\mathcal M} f(x),$

or in other words: find the point $x$ on the manifold, where $f$ reaches its minimal function value.

`Manopt.jl` provides a framework for optimization on manifolds.
Based on [Manopt](https://manopt.org) and
[MVIRT](https://ronnybergmann.net/mvirt/), both implemented in Matlab,
this toolbox aims to provide an easy access to optimization methods on manifolds
for [Julia](https://julialang.org), including example data and visualization methods.

If you want to delve right into `Manopt.jl` check out the
[Getting Started: Optimize!](@ref Optimize) tutorial.

The package can be roughly split into four parts

**1. Manifolds** Manifolds consist of three elements: a [`Manifold`](@ref) type
that stores general information about the manifold, for example a name, or for
example in order to generate a [`randomMPoint`](@ref), an [`MPoint`](@ref) storing
data to represent a point on the manifold, for example a vector or a matrix, and
a [`TVector`](@ref) stroing data to represent a point in a tangent space $T_x\mathcal M$ of
such an [`MPoint`](@ref).

**2. Functions on Manifolds**
Several functions arranged in groups are available, for example
[cost functions](@ref CostFunctions), [differentials](@ref DifferentialFunctions),
and [gradients](@ref GradientFunctions) as well as
[proximal maps](@ref proximalMapFunctions), but also several [jacobi Fields](@ref JacobiFieldFunctions) and their [adjoints](@ref adjointDifferentialFunctions)

**3. Optimization Algorithms (Solvers)**
For every optimization algorithm, a [solver](@ref Solvers) is implemented based on a [`Problem`](@ref) that describes the problem at hand, for example a [`GradientProblem`](@ref) in general and specific [`Options`](@ref) that set up the solver, i.e. parameters and initial values.

**4. Vizualization**
To visualize and interprete results, `Manopt.jl` aims to provide both easy plot functions as well as [exports](@ref Exports). Furthermore a system to get [debug](@ref DebugOptions) during the iterations of
an algorithms as well as [record](@ref RecordOptions) capabilities, i.e. to record a specified tuple of
values per iteration.

## Notation

During this documentation, we distinguish variables `x` and mathematical symbols $x$ from time to time.

| Symbol | used for
|:---|:---|
$\mathcal M, \mathcal N$ | a manifold
$d,d_1,\ldots,d_n$ | dimension(s) of a manifold
$x,y,z,x_1,\ldots,x_n$ | points on a manifold
$T_x\mathcal M$ | the tangent space of $x\in\mathcal M$
$\xi,\nu,\eta,\xi_1,\ldots,\xi_n$ | tangent vectors, might be extended by the base point, i.e. $\xi_x$
$\log_xy$ | logarithmic map
$\exp_x\xi$ | exponential map
$g(t; x,y)$ | geodesic connecting $x,y\in\mathcal M$ with $t\in [0,1]$
$\langle \cdot, \cdot\rangle_x$ | Riemannian inner product on $T_x\mathcal M$
$\operatorname{PT}_{x\to y}\xi$ | parallel transport of $\xi\in T_x\mathcal M$ from $x$ to $y$ along $g(\cdot;x,y)$

## Literature

[AMS08] P.-A. Absil, R. Mahony and R. Sepulchre, Optimization Algorithms on
Matrix Manifolds, Princeton University Press, 2008,
[open access](http://press.princeton.edu/chapters/absil/)
