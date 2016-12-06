# Manifold-valued Image Processing
<div align="right">
   Ronny Bergmann <bergmann@mathematik.uni-kl.de>
</div>

A Julia package to perform Image Processing on Images and Data having values
on a manifold.
the algorithms are based on the [Manfiold-valued Image Processing Toolbox](http://www.mathematik.uni-kl.de/imagepro/members/bergmann/mvirt/)
(MVIRT) available as Matlab Source code.

## Manfiolds
* `Manifold.jl` the general manifold containing functions and algorithms applicable to all manifolds
* `Sn.jl` the n-dimensional Sphere represented by n+1-dimensional vectors
* `S1.jl` the 1-dimensional Sphere, i.e. the cirvle, represented by phase values

## Algorithms
* `TV_Regularization` available as `CPPA` based algorithm

## Proximal Maps
* `TV`
* `TVSquared`
