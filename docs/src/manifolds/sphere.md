# [The $n$-sphere $\mathbb S^n$ embedded in $\mathbb R^{n+1}$](@id SphereManifold)
The Sphere $\mathcal M = \mathbb S^n$ posesses the following instances of the
abstract types [`Manifold`](@ref), [`MPoint`](@ref), and [`TVector`](@ref).
```@docs
Sphere
SnPoint
SnTVector
```
# Specific Functions
```@docs
distance(::Sphere,::SnPoint,::SnPoint)
dot(::Sphere,::SnPoint,::SnTVector,::SnTVector)
exp(::Sphere,::SnPoint,::SnTVector,::Float64)
log(::Sphere,::SnPoint,::SnPoint)
manifoldDimension(::SnPoint)
manifoldDimension(::Sphere)
norm(::Sphere,::SnPoint,::SnTVector)
parallelTransport(::Sphere,::SnPoint,::SnPoint,::SnTVector)
```
