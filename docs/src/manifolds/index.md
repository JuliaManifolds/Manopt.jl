# [Riemannian Manifolds](@id RiemannianManifolds)

```@meta
CurrentModule = Manopt
```

All manifolds inherit from [`Manifold`](@ref) to store their main properties, which is
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

## [List of available Manifolds](@id Manifolds)

Furthermore there are two types accompanying each manifold – a point on the
manifold inheriting from [`MPoint`](@ref) and the tangent vector
[`TVector`](@ref). For both the term manifold is shortened to `M` for concise
naming. Each manifold also inherits such a short abbreviation, see `Abbr.` in
the following table.

|  Manifold $\mathcal M$ | File | Abbr. | Comment
|:-----------------------|:-----|:-----:|:--------
A manifold $\mathcal M$ | `Manifold.jl`| `M`| | the (abstract) base manifold $\mathcal M$ | collects general functions and types
[$1$-sphere $\mathbb S^1$](@ref CircleManifold)  | `Circle.jl`  | `S1`| represented as angles $x\in[-\pi,\pi)$
[Euclidean space $\mathbb R^n$](@ref EuclideanSpace) | `Euclidean.jl` | `Rn` |  $n$-dimensional Euclidean space $\mathbb R^n$
[Grassmannian manifold $\mathrm{Gr}(k,n)$](@ref GrassmannianManifold) | `Grassmannian.jl` | `Gr` | embedded in $\mathbb R^{n\times k}$
[$n$-dim. Hyperbolic space $\mathbb H^n$](@ref HyperbolicManifold) | `Hyperbolic.jl` | `Hn` | embedded in $\mathbb R^{n+1}$
[special orthogonal group $\mathrm{SO}(n)$](@ref SOn) | `Rotations.jl` | `SO` | represented as rotation matrices
[$n$-sphere $\mathbb S^n$](@ref SphereManifold) | `Sphere.jl` | `Sn` | embedded in $\mathbb R^{n+1}$
[Stiefel $\mathrm{St}(k,n)$](@ref StiefelManifold) | `Stiefel.jl`| `St` |  contains both the real- ad the complex-valued case
[symmetric matrices $\mathcal{Sym}(n)$](@ref SymmetricManifold) | `Symmetric.jl` | `Sym` | $n\times n$ symmetric matrices
[symmetric positive definite matrices $\mathcal P(n)$](@ref SymmetricPositiveDefiniteManifold) | `SymmetricPositiveDefinite.jl` | `SPD` | $n\times n$ symmetric positive matrices using the affine metric

If you're missing your favorite manifold, [give us a note on Github](https://github.com/kellertuer/Manopt.jl/issues).

## Special Types of Manifolds

### Manifolds build upon Manifolds

| Manifold $\mathcal M$ | File | Abbr. | Comment
-------------------------|------|-------|---------
Power manifold           | `Power.jl` | `Pow` | Builds $\mathcal N^{d_1\times\cdot\times d_k}$ of any manifold $\mathcal N$
Product manifold         | `Product.jl` | `Prod` | Build the product manifold $\mathcal N_1\times\cdots\times\mathcal N_k$ of manifolds
Tangent bundle           | `TangentBundle.jl` | `TB` | tangent bundle of a manifold, i.e. the set of all tuples $(x,\xi), \xi \in T_x\mathcal M$, $x\in\mathcal M$ with the induced metric.

for more details see [Combined Manifolds](@ref CombinedManifolds)

### Special Properties of Manifolds

Special types of manifolds are introduced by
[SimpleTraits.jl](https://github.com/mauro3/SimpleTraits.jl). They can be used
to clarify that a manifold possesses a certain property. For example two points
on a matrix manifold can be multiplied, though the result is not necessarily a
point on the manifold anymore. Traits have to goals here: Provide functions that
are common for all manifolds of such a type (e.g. the [`⊗`](@ref) for Lie
groups) as a common interface and to specify certain functions or solvers for
these certain types, that for example take advantage of [`⊗`](@ref) then.

#### Embedded Manifold

```@docs
IsEmbeddedM
IsEmbeddedP
IsEmbeddedV
```

#### [Lie Group Manifold](@id LieGroup)

```@docs
IsLieGroupM
IsLieGroupP
IsLieGroupV
⊗
```

#### [Matrix Manifold](@id MatrixManifold)

```@docs
IsMatrixM
IsMatrixP
IsMatrixTV
```

## Functions that need to be implemented for a Manifold

If you plan to implement a new manifold within `Manopt.jl`, the following
functions should be implemented. If you only implement a few of these functions,
not all algorithms might work.
all these functions have a fallback providing an error message if the function is
not (yet) implemented.
Otherwise, for example, if the field of the inner representant of [`MPoint`](@ref)
or [`TVector`](@ref) is the field `.value` of your data structure, the default
implementation of [`getValue`](@ref) directly works.
In the following list `M <: Manifold` the manifold type
 represents the manifold `Q`,`P <: MPoint` the type of a point on the new manifold,
`T <: TVector` a corresponding tangent vector in a suitable tangent space,

```@docs
addNoise(M::mT, x::P, options...) where {mT <: Manifold, P <: MPoint}
distance(M::mT, x::T, y::T) where {mT <: Manifold, T <: MPoint}
dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPoint, T <: TVector, S <: TVector}
exp(M::mT, x::P, ξ::T,t::Float64=1.0) where {mT<:Manifold, P<:MPoint, T<:TVector, N<:Number}
getValue(ξ::P) where {P <: MPoint}
getValue(ξ::T) where {T <: TVector}
log(::mT,::P,::Q) where {mT<:Manifold, P<:MPoint, Q<:MPoint}
manifoldDimension(::P) where {P <: MPoint}
manifoldDimension(::mT) where {mT <: Manifold}
norm(::mT,::P,::T) where {mT<:Manifold, P<: MPoint, T<:TVector}
parallelTransport(::mT,::P,::Q,::T) where {mT <: Manifold, P <: MPoint, Q <: MPoint, T <: TVector}
randomMPoint(M::mT,s::Symbol,options...) where {mT <: Manifold}
randomTVector(M::mT,p::P,s::Symbol,options...) where {mT <: Manifold, P<: MPoint}
tangentONB(::mT, ::P, ::Q) where {mT <: Manifold, P <: MPoint, Q <: MPoint}
tangentONB(::mT, ::P, ::T) where {mT <: Manifold, P <: MPoint, T <: TVector}
typeofMPoint(::T) where {T <: TVector}
typeofMPoint(::Type{T}) where {T <: TVector}
typeofTVector(::P) where {P <: MPoint}
typeofTVector(::Type{P}) where {P <: MPoint}
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

## A decorator for checks and validation

In order to check and/or validate tangent vectors, the decorator pattern
[`TVectorE`](@ref) is available for any subtype of [`TVector`](@ref)
as follows

```@docs
TVectorE
```

together with a small helper [`MPointE`](@ref) that indicates to the `log` to returns an
extended tangent vector as soon as one of its arguments is an extended manifold point.

```@docs
MPointE
```

for these two data items, the following additional features that are activated

### Inheritance

Basic functions like `exp`, `log`, `parallelTransport`, and [`randomTVector`](@ref)
return an extended [`MPointE`](@ref) or [`TVectorE`](@ref) whenever one of its arguments is an
extended input. This enables, that setting (only) one input for a calculation
to an extended version, this property propagates this into all the algorithm.

Note that this might increase memory usage and hence reduce performance, since
for any [`TVectorE`](@ref) internally stores both a [`TVector`](@ref) as well as
its base [`MPoint`](@ref) (as extension).

### Checks

```@docs
checkBasePoint
```

For extended data decorators, whenever possible in the basic functions listed above
a [`checkBasePoint`](@ref) completely automatically performed. For example, when calling
`exp(M,x,ξ)`, as soon as `ξ` is an extended vector, `checkBasePoint(x,ξ)` is called
before performing the original exponential map.

This way, as many checks are performed, whether corresponding points and vectors
or two vectors involved have the correct base points.

### Validation

Every extended type carries a further boolean `validation`, whose default is
`true`, i.e. to perform validation. activating validation one needs to implement
the following two functions, otherwise, a lot of warnings might occur.

```@docs
validateMPoint(::mT, ::P) where {mT <: Manifold, P <: MPoint}
validateTVector(::mT,::P,::T) where {mT<:Manifold, P<: MPoint, T<:TVector}
```

whenever possible (see [`checkBasePoint`](@ref) above). Since such a validation might not be
available for your favorite manifold, you can deactivate validation by setting
the boolean to `false`. Every new extended type inherits the false, whenever one
of its part (i.e. either the [`TVector`](@ref) _or_ [`MPoint`](@ref)) has
validation set to false.

So while checking as often as possible, this feature can easily be deactivated.

### Internals

To access the inner value of [`MPointE`](@ref) and the base stored in [`TVectorE`](@ref)
you can
use

```@docs
strip
getBasePoint
```

Furthermore the following functions are
mapping to the internally stored data and
encapsulate the results with the extended variant if applicable

* [`getValue`](@ref)
* [`addNoise`](@ref)
* [`distance`](@ref)
* [`dot`](@ref)
* [`exp`](@ref)
* [`getValue`](@ref)
* [`log`](@ref)
* [`manifoldDimension`](@ref)
* [`norm`](@ref)
* [`parallelTransport`](@ref)
* [`randomMPoint`](@ref)
* [`randomTVector`](@ref)
* [`tangentONB`](@ref)
* [`typicalDistance`](@ref)
* [`zeroTVector`](@ref)

as well as mathematical operators on tangent vectors and comparison operators.
