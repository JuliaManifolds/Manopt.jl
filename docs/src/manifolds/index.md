```@meta
CurrentModule = Manopt
```
# Riemannian Manifolds
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
manifold inheriting from [`MPoint`](@ref) and the tangential vector [`TVector`](@ref). For both
the term manifold is shortened to `M` for concise naming. Each manifold also
inherits such a short abbreviation, see `Abbr.` in the following table.

|  Manifold $\mathcal M$ | File | Abbr. | Comment
-------------------------|------|-------|---------
A manifold $\mathcal M$ | `Manifold.jl`| `M`| | the (abstract) base manifold $\mathcal M$
[$1$-sphere $\mathbb S^1$](@ref CircleManifold)  | `Circle.jl`  | `S1`| represented as angles $x\in[-\pi,\pi)$
[$n$-dim. Hyperbolic space $\mathbb H^n$](@ref HyperbolicManifold) | `Hyperbolic.jl` | `Hn` | embedded in $\mathbb R^{n+1}$
[$n$-sphere $\mathbb S^n$](@ref SphereManifold) | `Sphere.jl` | `Sn` | embedded in $\mathbb R^{n+1}$
[Euclidean space $\mathbb R^n$](@ref EuclideanSpace) | `Euclidean.jl` | `Rn` |  $n$-dimensional Euclidean space $\mathbb R^n$
[symmetric matrices $\mathcal{Sym}(n)$](@ref SymmetricManifold) | `Symmetric.jl` | `Sym` | $n\times n$ symmetric matrices
[symmetric positive definite matrices $\mathcal P(n)$](@ref SymmmetricPositiveDefiniteManifold) | `SymmetricPositiveDefinite.jl` | `SPD` | $n\times n$ symmetric positive matrices using the affine metric
## Special Types of Manifolds
Special types of manifolds are introduced by [SimpleTraits.jl](https://github.com/mauro3/SimpleTraits.jl)
### Embedded Manifold
```@docs
IsEmbeddedM
IsEmbeddedP
IsEmbeddedV
```

### Lie Group Manifold
```@docs
IsLieGroupM
IsLieGroupP
IsLieGroupV
```

### Matrix Manifold
```@docs
IsMatrixM
IsMatrixP
IsMatrixV
```

Further special manifolds can be created combining existing ones, see [Combined Manifolds](@ref CombinedManifolds)

## Functions that need to be implemented for a Manifold
If you plan to implement a new manifold within `Manopt.jl`, the following
functions should be implemented. If you only implement a few of these functions,
not all algorithms might work.
all these functions have a fallback providing an error message if the function is
not (yet) implemented.
Otherwise, for example, if the field of the inner representant of `MPoint`
or [`TVector`](@ref) is the field `.value` of your struct, [`getValue`](@ref) directly
works.

```@docs
addNoise(::mT,::T,::Number) where {mT <: Manifold, T <: MPoint}
distance(::M,::P,::P) where {M<:Manifold, P<:MPoint}
dot(::mT,::P,::T,::S) where {mT <: Manifold, P <: MPoint, T <: TVector, S <: TVector}
exp(::mT,::P,::T,::N) where {mT<:Manifold, P<:MPoint, T<:TVector, N<:Number}
getValue
log(::mT,::P,::Q) where {mT<:Manifold, P<:MPoint, Q<:MPoint}
manifoldDimension(::P) where {P <: MPoint}
manifoldDimension(::mT) where {mT <: Manifold}
norm(::mT,::P,::T) where {mT<:Manifold, P<: MPoint, T<:TVector}
parallelTransport(::mT,::P,::Q,::T) where {mT <: Manifold, P <: MPoint, Q <: MPoint, T <: TVector}
tangentONB(::mT, ::P, ::Q) where {mT <: Manifold, P <: MPoint, Q <: MPoint}
typicalDistance(M::mT) where {mT <: Manifold}
zeroTVector(::mT, ::P) where {mT <: Manifold, P <: MPoint}
```
## Functions implemented for a general manifold
the following base functions are implemented for general manifolds and are
based on the functions from the last section

```@docs
adjointJacobiField
geodesic
jacobiField
Manopt.mean
Manopt.median
midPoint
reflection
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
getVector
```
