```@meta
CurrentModule = Manopt
```
# Manifolds within `Manopt.jl`
All manifolds inherit from `Manifold` to store their main properties, which is
most prominently the manifold dimension and the name of the manifold. This will
be extended in the future, for example properties denoting whether the
manifold is explicitly given in the sense of a closed form exponential and
logarithmic map for example, or only approximately.

A Riemannian manifold in `Manopt.jl` consist of three types:
```@docs
Manifold
MPoint
TVector
```

Furthermore there are two types accompanying each manifold – a point on the
manifold inheriting from `MPoint` and the tangential vector `TVector`. For both
the term manifold is shortened to `M` for concise naming. Each manifold also
inherits such a short abbreviation, see `Abbr.` in the following table.

|  Manifold $\mathcal M$ | File | Abbr. | Comment
-------------------------|------|-------|---------
A manifold $\mathcal M$ | `Manifold.jl`| `M`| | the (abstract) base manifold $\mathcal M$
$1$-sphere $\mathbb S^1$  | `Circle.jl`  | `S1`| represented as angles $x\in[-\pi,\pi)$
[$n$-sphere $\mathbb S^n$](@ref) | `Sphere.jl` | `Sn` | embedded in $\mathbb R^{n+1}$
Euclidean space $\mathbb R^n$ | `Euclidean.jl` | `Rn` |  $n$-dimensional Euclidean space $\mathbb R^n$
symmetric positive definite matrices $\mathcal P(n)$ | `SymmetricPositiveDefinite.jl` | `SPD` |  $n\times n$ symmetric positive matrices using the affine metric
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

## Functions that need to be implemented for Manifolds
If you plan to implement a new manifold within `Manopt.jl`, the following
functions should be implemented. If you only implement a few of these functions,
not all algorithms might work.
all these functions have a fallback providing an error message if the function is
not (yet) implemented.

```@docs
addNoise
distance
Manopt.dot
Manopt.exp
getValue
Manopt.log
manifoldDimension
norm
parallelTransport
```
## Functions implemented for a general manifold
the following base functions are implemented for general manifolds and are
based on the functions from the last section

```@docs
geodesic
Manopt.mean
Manopt.median
variance
```

## A decorator for validation
In order to validate tangent vectors, the decorator pattern `TVectorE` is available for any subtype of `TVector`
as follows
```@docs
TVectorE
```
together with a small helper `MPointE` that indicates to the `log` to return an
extended tangent vector
```@docs
MPointE
```
For data decorated with the extended feature, the manifold functions `exp`,
`log`, `dot`, and `norm` are prepended by a check of the basis point of the involved tangent vectors to be correct, e.g. for `exp(x,ξ)` `ξ.base` has to be equal to `x` in value. This is
performed by the internal function
```@docs
checkBase
```
that checks, that the values of bases for tangent vectors are the same or that
for a tuple $(x,\xi)$ the point $x$ is the base point of $\xi$. Furthermore, to
access the inner value of `MPointE` and the base stored in `TVectorE` you can use
```@docs
getBase
```
