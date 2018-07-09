# Manifolds within `Manopt.jl`

All manifolds inherit from `Manifold` to store their main properties, which is
most prominently the manifold dimension and the name of the manifold. This will
be extended in the future, for example properties denoting whether the
manifold is explicitly given in the sense of a closed form exponential and
logarithmic map for example, or only approximately.

Furthermore there are two types accompanying each manifold – a point on the
manifold inheriting from `MPoint` and the tangential vector `TVector`. For both
the term manifold is shortened to `M` for concise naming. Each manifold also
inherits such a short abbreviation, see `Abbr.` in the following table.

|  Manifold $\mathcal M$ | File | Abbr. | Comment
-------------------------|------|-------|---------
A manifold $\mathcal M$ | `Manifold.jl`| `M`| | the (abstract) base manifold $\mathcal M$
$1$-sphere $\mathbb S^1$  | `Circle.jl`  | `S1`| represented as angles $p_i\in[-\pi,\pi)$
[$n$-sphere $\mathbb S^n$](@ref) | `Sn` | `M` | embedded in $\mathbb R^{n+1}$
Euclidean space | `Euclidean.jl` | `Rn` |  $n$-dimensional Euclidean space $\mathbb R^n$
symmetric positive definite matrices | `SymmetricPositiveDefinite.jl` | `SPD` |  $n\times n$ symmetric positive matrices using the affine metric

A Riemannian manifold in `Manopt.jl` consist of three types:
```@docs
Manifold
MPoint
TVector
```
## Functions that need to be implemented for Manifolds
If you plan to implement a new manifold within `Manopt.jl`, the following
functions should be implemented. If you only implement a few of these functions,
not all algorithms might work.
all these functions have a fallback providing an error message if the function is
not (yet) implemented.

```@docs
  exp
  log
  norm
  dist
  dot
  parallelTransport
  retraction
  addNoise
```
## Functions implemented for a general manifold
the following base functions are implemented for general manifolds and are based on the functions from the last section

```@docs
  mean
  median
  geodesic
```
## Traits for special types of Manifolds

* With `@isLieGroupM`, `@isLieGroupP`, and `@isLieGroupV`
  a manifold, point, and vector, respectively is (belongs to) a Lie group manifold.
* With `@isMatrixM`, `@isMatrixP`, and `@isMatrixV`
  a manifold, point, and vector, respectively is (belongs to) a matrix manifold.

## Special Manifolds to extend the above Basic manifolds

* `PowerManifold.jl` (Abbr. `PowM`) introduces a power manifold
  $\mathcal M^n$, where $n$ can be a vector.
* `ProductManifold.jl` (Abbr. `ProdM`) introduces a product manifold
  $\mathcal M_1\times \mathcal M_2\times\cdot \mathcal M_n$ and corresponding
  point and vector types. The manifolds may also be arranged in a general array
  instead of in a vector.
