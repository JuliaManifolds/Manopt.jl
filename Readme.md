# Manopt.jl
<div align="right">
   Ronny Bergmann <bergmann@mathematik.uni-kl.de>
</div>

A Julia package to perform Image Processing on Images and Data having values
on a manifold.
the algorithms are based on the [Manfiold-valued Image Processing Toolbox](http://www.mathematik.uni-kl.de/imagepro/members/bergmann/mvirt/)
(MVIRT) and the [Manopt](http://www.manopt.org) toolbox, both available as Matlab Source code.

## Optimization
Formulate your optimization problem using the `Problem` struct. These are available for
several main approaches
* Gradients
* Hessian
* proximal maps
and combinations of these.
Furthermore proximal maps can be applied to only parts of power manifolds if
one works on an array of manifold-valued data (which implicitly specifies the
dimesnion of the power-manifold)

The problem will also include `stoppingcriterium` and the `stoppingcriterionfunction`.
Furthermore the cost function will also be included as well as a `verbosity`
level.

## Manifolds
  The manifold itself is the central type of this Julia package concerning
  data types. We split the manifold into three types:
  * `Manifold` containing information that either is not available when no
  points are present, e.g. when calling a `random(M::Manifold,arraysize)` type
  function or which should not be stored in all data items. Still, all base
  functions depend on an instance of this type, e.g. the metric (`dot`) might be
  implemented differently (Affine, Log-Euclidean for SPDs) and then `dot(M,ξ,ν)`
  can be used to switch the metric by chaning the correspoing data in `M`
  (e.g. the metric is storad as an SPD matrix itself within `M::Manifod`).
  * `MPoint` a point on the manifold, where the naming convention is to shorten
  the manifold to `M` or a specific manifold to a unique abbreviation
  (e.g. `Sn` for the sphere)
  * `TVector` a manifold tangential vector with the same naming convention as
  above replacing again the `M` shorthand for manifold

	For an overview of already available manifolds see [Documentation.md](Documentation.md)

## Functions available for Manifolds

Implemented on the most general level for `Manifold`
* `mean(M,Vector<MPoint> F)` compute the mean (Riemannian center of mass)
of a set of points on `M`
* `median(M,Vector<MPoint> F)` compute the median based on the variational minimizer of the distances between all points on `M`
* `geodesic(M,p,q)` the function of the geodesic between `p` and `q`
* `geodesic(M,p,q,t)` evaluate the geodesic between `p` and `q` at `t`
* `geodesic(M,p,q,n)` evaluate the geodesic between `p` and `q` at n equistpaced points from \([0,1]\)
* `geodesic(M,p,q,t)` evaluate the geodesic between `p` and `q` at points from teh vector `v`

Available as functions (fallbacks in the abstract case)
  n general and to be implemented on you favourite manifold:
* `exp(M,x,ξ)` the exponential map from the tangential space at `x` to the manifold
* `log(M,x,y)` the (locally defined) inverse of the logarithmic map
* `norm(M,x,ξ)` the norm of a tangential vector at x
* `dist(M,x,y)` the geodesic distance between two points on the manifold
* `dot(M,ξ,ν)` the Riemannian metric of two vectors.
* `retraction(M,x,ξ)` the retraction on the manifold.
* `parallelTransport(M,x,y,ξ)` transports the tangential vector

note that in the general case, the first argument is always the manifold. This can be used to easily change for example the metric. If you have a specific manifold you may also introduce the shorthands `exp(x,y)` for ease of usage.

## Algorithms
* `TV_Regularization` available as `CPPA` based algorithm

## Proximal Maps
* `proxTV`
* `proxTVSquared`
* `proxDistance`
* `proxDistanceSquared`
