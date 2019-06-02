# Welcome to Manopt.jl

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
this toolbox provide an easy access to optimization methods on manifolds
for [Julia](https://julialang.org), including example data and visualization methods.

If you want to delve right into `Manopt.jl` check out the
[Getting Started: Optimize!](@ref Optimize) tutorial.

`Manopt.jl` makes it easy to use an algorithm for your favorite
manifold as well as a manifold for your favorite algorithm. It already provides
many manifolds and algorithms, which can easily be enhanced, for example to
[record](@ref RecordOptions) certain data or
[display information](@ref DebugOptions) throughout iterations.

## Main Features

**1. Manifolds**
Manifolds consist of three elements: a [`Manifold`](@ref) type that stores
general information about the manifold, for example a name, or for example in
order to generate a [`randomMPoint`](@ref), an [`MPoint`](@ref) storing data to
represent a point on the manifold, for example a vector or a matrix, and a
[`TVector`](@ref) string data to represent a point in a tangent space
$T_x\mathcal M$ of such an [`MPoint`](@ref). If a manifold has certain
properties, for example if it is a [matrix manifold](@ref MatrixManifold) or a [Lie
group](@ref LieGroup), see for example the binary operator [`⊗`](@ref). For a
list of available manifolds, see [the list of manifolds](@ref Manifolds)

**2. Functions on Manifolds**
Several functions are available, implemented on an arbitrary manifold, [cost
functions](@ref CostFunctions), [differentials](@ref DifferentialFunctions), and
[gradients](@ref GradientFunctions) as well as [proximal maps](@ref
proximalMapFunctions), but also several [jacobi Fields](@ref
JacobiFieldFunctions) and their [adjoints](@ref adjointDifferentialFunctions).

**3. Optimization Algorithms (Solvers)**
For every optimization algorithm, a [solver](@ref Solvers) is implemented based
on a [`Problem`](@ref) that describes the problem to solve and its
[`Options`](@ref) that set up the solver, store interims values. Together they
form a [plan](@ref planSection).

**4. Visualization**
To visualize and interpret results, `Manopt.jl` aims to provide both easy plot
functions as well as [exports](@ref Exports). Furthermore a system to get
[debug](@ref DebugOptions) during the iterations of an algorithms as well as
[record](@ref RecordOptions) capabilities, i.e. to record a specified tuple of
values per iteration, most prominently [`RecordCost`](@ref) and
[`RecordIterate`](@ref). Take a look at the
[Getting Started: Optimize!](@ref Optimize) tutorial how to easily activate
this.

All four parts are accompanied by a documentation that can also be accessed from
within `Julia REPL` and provides detailed information, e.g. the formula for an
[exponential or logarithmic map on the manifold of symmetric positive definite matrices](@ref SymmetricPositiveDefiniteManifold) or literature references for an algorithm like [`cyclicProximalPoint`](@ref).

## Notation

During this documentation, we refer to a variable with e.g. both `x` and $x$
depending on whether the context refers to a code fragment or a mathematical
formula, respectively.

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

## Contribute

If you notice a typo, have a question or would like even to contribute, give me a note
at `manopt@ronnybergmann.net` or visit the [GitHub repository](https://github.com/kellertuer/Manopt.jl/) to clone/fork the repository.

## Literature

```@raw html
<ul><li id="AbsilMahonySepulchre2008">
    [<a>Absil, Mahony, Sepulchre, 2008</a>]
    P.-A. Absil, R. Mahony and R. Sepulchre,
    <emph>Optimization Algorithms on
    Matrix Manifolds</emph>, Princeton University Press, 2008,
    doi: <a href="https://doi.org/10.1515/9781400830244">10.1515/9781400830244</a>,
    <a href="http://press.princeton.edu/chapters/absil/">open access</a>.
</li></ul>
```
