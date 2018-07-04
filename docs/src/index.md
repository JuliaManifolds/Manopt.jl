# Manopt.jl
## Optimization on Manifolds
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
manifold inheriting from `MPoint` and the tangential vector `TVector`. For both
the term manifold is shortened to `M` for concise naming. Each manifold also
inherits such a short abbreviation, see `Abbr.` in the following table.
Furthermore there is an abstract type (struct) `MatrixManifold` (Abbreviation `MM`) that
overloads the operands `*,/,+,-` for the `MatPoint`. Furthermore each `MatPoint`is
indeed a matrix, so they also possess an array to store its decomposition, namely
for $\mathbf{A}\in\mathbb R^{n,m}$ its singular value decomposition
$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}$.
Compared to Matlab, we do not implicitly do vectorization but use explicitly a
`ProdctManifold` (with `ProdMPoint` and `ProdTVector`) consisting of an array of manifolds,
as well as the product manifold consisting of only one manifold, i.e. the `PowerManifold`
(with `PowMPoint` and `PowTVector`).

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

## Organization of Code
Despite the given structure by Julia (folders `src/`, `docs/`, `test/`), this
package consists of the following structre:
* a folder in the main directory `Tutorials/` containing all Tutorials as Juypter
  notebooks (thats why they are not wihtin `src/`) that should cover all main
  algorithms and introduce the data structures (`Manifold`, `MPoint`, `TVector`
  structure)
* the `src/` folder itself structures the parts of the Toolbox as follows (roughly
  in order of importance). The main file including all following is `Manopt.jl`
  encapsulating all `include`s in a module. That also means, that all files
  seperately handle `export`s and `import`s. * `manifold/` contains a file for
  each manifold implementing the subtypes. The main types are defined in
  `Manifold.jl` as well as operators in the types as well as fallbacks to provide
  errors for not implemented cases of types or non-fitting cases (like `exp` with
  a point from manifold A and a tangent vector from manifold B)
  * `solvers/` contains all solving algorithms like `steepestDescent`.
    These should always be available in two formats: One based on a `problem`
    structure, such that they can be called by just one argument, for example
    when checking a range of parameters and such that you only need to change one
    value in the struct (which is defined in `problems/`). The second version should
    be one with only mandatory parameters, where all others are (`;...)`) optional
    key-value parameters and also providing thorough checks of the input and internally
    call the first ones. While
    the first one might be harder to call, the less checks and without key-value
    makes them faster after compilation, which is also beneficiary for the second ones
    since they internally call them. The second ones provide the easy-to-use
    functions for new users, where all optional parameters are set so values that
    are for a start in a certain sense _reasonable_.
  * `algorithms/` contains all smaller parts or helpers for the solvers, for
    example the Armijo line search
  * `examples/` contains examples that are not yet tutorials or jupyter notebooks
    (maybe removed in future, when all examples are Tutorials)
  * `helpers/` contains small helpers like the debug functionals
  * `plots/` contains all plotting functions
  * `problem/` contains all structures for the internal algorithms or solvers

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
Algorithms should be implemented for any subtype of `Manifold`, i.e. a function
taking a manifold point as an argument should be written in the form `function
F{MP <: MPoint}(x::MP)`. Furthermore, for performance issues, the inner
functions that are required to run often and fast, should not (for now, `Julia 0.6`)
contain optional arguments (`...kwargs`), since that reduces the performance per
call. The recommendation is hence to write a classical function using a `struct problem`
(see `problem/problem.jl` for a start).

## Verbosity and Debug
There is a global `verbosity` level within the algorithms that steers the amount
of output. It ranges from `verbosity=0` meaning no output at all to `verbosity=5`
including times and a lot of debug. The rough course is as follows

| Level | Additional Output     |
|-------|-----------------------|
|   1   | Main start and result |
|   2   | not yet used          |
|   3   | End criteria of algorithms etc. |
|   4   | Time measurements |
|   5   | Iteration interims values |

For the last Level, an individual `debug` field in the structs provides the
possibility to add an own function into the iteration. This function always takes
just one argument, namnely a `Dict{String, Any}` dictionary, i.e. a hashmap to
store any data passed to a debug function. If values are present in the `debugSettings`
they should be updated in the algorithm.

Maybe something similar using a `record` field would also be nice.

### A general remark
In general, one should additionally provide an interface that uses keyword-argument lists (kwargs) as a Dictionary (similarly to the matlab usage),
having the same name, checking input and passing it to the struct-based function. That way the basic function (with struct) is compiled to a function being quite fast, and the wrapper makes the function easily accessible with `kwargs` for a user; see e.g. https://stackoverflow.com/a/39602289/1820236. This might also be a good idea for the plot functions, because we can filter dictionaries and pass parts to the internally used plot functions, making it possible to provide line styles but also appearance of e.g. the sphere as options.

## Tests
Every new function or manifold should be accompanied by a test suite, placed
within `test/` (the script `runtests.jl` takes care of running all test cases
within that folder automatically).
A plan is to use Travor CI to check for code coverage and all tests passing
automatically, soon.

## Literature
[AMS08] P.-A. Absil, R. Mahony and R. Sepulchre, Optimization Algorithms on
Matrix Manifolds, Princeton University Press, 2008,
[open access](http://press.princeton.edu/chapters/absil/)
