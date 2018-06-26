# Manifold optimization
The `Manopt.jl` Julia package provides all necessary tools to perform
optimization on manifolds and apply these methods to manifold-valued image
processing. This file intends to fix notations both mathematical and with
respect to source code throughout this package. For an introduction to
manifolds, we refer to [AMS08].
This package further aims to unify Manopt and MVIRT, i.e. to find a “Julia way”
to perform all algorithms given in these packages with all benefits from Julia.

## Manifolds
All manifolds inherit from `Manifold` to store their main properties, which is
most prominently the manifold dimension and the name of the manifold. This will
be extended in the future, for example ba properties denoting whether the
manifold is explicitly given in the sense of a closed form exponential and
logarithmic map for example, or only approximately.

Furthermore there are two types accompanying each manifold – a point on the
manifold inheriting from `MPoint` and the tangential vector `MTVector`. For both
the term manifold is shortened to `M` for concise naming. Each manifold also
inherits such a short abbreviation, see `Abbr.` in the following table.
Furthermore there is an abstract type (struct) `MatrixManifold` (Abbreviation `MM`) that
overloads the operands `*,/,+,-` for the `MMPoint`. Furthermore each `MMPoint`is
indeed a matrix, so they also possess an array to store its decomposition, namely
for $\mathbf{A}\in\mathbb R^{n,m}$ its singular value decomposition
$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}$.
Compared to Matlab, we do not implicitly do vectorization but use explicitly a
`ProdctManifold` (with `ProdMPoint` and `ProdMTVector`) consisting of an array of manifolds,
as well as the product manifold consisting of only one manifold, i.e. the `PowerManifold`
(with `PowMPoint` and `PowMTVector`).

File | Abbr. |  supertype |  Manifold $\mathcal M$ | Comment
-----|-------|------------|-----------------------|---------
`Circle.jl`  | `S1`| `M` | $1$-Sphere $\mathbb S^1$ | represented as angles $p_i\in[-\pi,\pi)$
`Euclidean.jl` | `Rn` | `M` |  $n$-dimensional Euclidean space $\mathbb R^n$
`Manifold.jl`| `M`| | the (abstract) base manifold $\mathcal M$ |  
`MatrixManifold` | `MM` | `M` |  The manifold, where points are represented by matrices |
`PowerManifold.jl` | `PowM` | `M` | $\mathcal M^n$ | where $n$ can be a vector |
`ProductManifold.jl` | `ProdM` | `M` | $\mathcal M_1\times \mathcal M_2\times\cdot \mathcal M_n$ | might be arranged in any array |
`SymmetricPositiveDefinite.jl` | `SPD` | `MM` |  $n\times n$ symmetric positive matrices | using the affine metric
`Sphere.jl` | `Sn` | `M` |  $n$-dimensional Sphere $\mathbb S^n$ | embedded in $\mathbb R^{n+1}$

## A summary of notations
The Riemannian Manifold will always be denoted by $\mathcal M$ its dimension
by $d\in\mathbb N$, and points on $\mathcal M$ by $p,q$ or $p_i$,
$i\in\mathcal I$, for more than 2 points.We denote the tangential space at
$p\in\mathcal M$ by $\mathrm{T}_p\mathcal M$ and the Riemannian metric by
$g_p(\xi,\nu) = \langle \xi,\nu\rangle_p$, where we denote tangential vectors
$\xi,\nu\in\mathrm{T}_p\mathcal M$ by small greek letters, and in most cases
omit denoting the basis point $p$, just if a confucion might arise or we
would like to emphasize the tangential space, we write $\xi_p,\nu_p$.

With capital letters $X,Y,Z$ we refer to vector fields on a manifold,
$X\colon \mathcal M \to \mathrm{T}\mathcal M$, such that $X(p) = X_p
\in \mathrm{T}\mathcal M$. Whenever we require to look at charts, they are
indicated by $\varphi,\psi\colon\mathcal U \to\mathbb R^d$,
$\mathcal U\subset\mathcal M$ with e.g. $x=\varphi(p)$, $y=\varphi(q)$.
When explicitly speaking of matrices, we employ $A,B,C\in\mathbb R^{n,n}$.

We denote the exponential map by $\exp_p\colon\mathrm{T}_p\mathcal M\to\mathcal
M$ and its (locally defined) inverse by $\log_p\colon\mathcal M
\to\mathrm{T}_p\mathcal M$.

Finally functions are denoted by $f,g\colon\Omega\to\mathcal M$,
$\Omega\subset\mathbb R^m$, where we use the short hand notation of $f_i =
f(x_i)$, $i\in\mathcal I$, for given fixed samples of a function $f$ and
functions defined on a manifold by $F,G\colon\mathcal M\to\mathbb R^m$. The
notion of samples valued is especially employed, when a Function $F$ depends
on fixed given data and variable data, e.g. $F(p;f) = d_{\mathcal M}^2(p,f)$
is meant as a function $F\colon \mathcal M\to\mathbb R$ defined for some fixed
(sampled) data $f\in\mathcal M$.

## The general Approach to Algorithms
Algorithms should be implemented for any subtype of `Manifold`, i.e. a function taking a manifold point as an argument should be written in the form `function F{MP <: MPoint}(x::MP)`. Furthermore, for performance issues, the inner functions that
are required to run often and fast, should (for now, `Julia 0.6`) contain
optional arguments (`...kwargs`), since that reduces the performance per call.
The recommendation.

## Literature
[AMS08] P.-A. Absil, R. Mahony and R. Sepulchre, Optimization Algorithms on
Matrix Manifolds, Princeton University Press, 2008,
[open access](http://press.princeton.edu/chapters/absil/)
