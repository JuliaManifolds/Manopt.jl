#
#      Sn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
import Base.LinAlg: norm, dot
import Base: exp, log, show
export Hyperbolic, HnPoint, HnTVector, getValue
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
#
# Type definitions
#

doc"""
    Hyperbolic <: Manifold
The manifold $\mathcal M = \mathbb H^n$ is the set

$\mathbb H^n = \Bigl\{x\in\mathbb R^{n+1}\Big|\langle x,x \rangle_{\mathrm{M}}= -x_{n+1}^2 + \displaystyle\sum_{k=1}^n x_k^2 = -1, x_{n+1} > 0\Bigr\},$
where $\langle\cdot,\cdot\rangle_{\mathrm{M}}$ denotes the Minkowski inner product.

This manifold is a matrix manifold (see [`IsMatrixM`](@ref)) and embedded (see
[`IsEmbeddedM`](@ref)).
Its abbreviation is `Hn`.
"""
struct Hyperbolic <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Hyperbolic(dimension::Int) = new("$dimension-Hyperbolic Space",dimension,"Hn($(dimension-1))")
end
doc"""
    HnPoint <: MPoint
A point $x$ on the manifold $\mathcal M = \mathbb H^n$ represented by a vector
$x\in\mathbb R^{n+1}$ with Minkowski inner product
$\langle x,x\rangle_{\mathrm{M}} = -1$.
"""
struct HnPoint <: MPoint
  value::Vector
  HnPoint(value::Vector) = new(value)
end
getValue(x::HnPoint) = x.value;

doc"""
    HnTVector <: TVector
A tangent vector $\xi \in T_x\mathcal M$ on the manifold
$\mathcal M = \mathbb H^n$.
"""
struct HnTVector <: TVector
  value::Vector
  HnTVector(value::Vector) = new(value)
end
getValue(ξ::HnTVector) = ξ.value;
# Traits
# ---
# (a) Sn is a MatrixManifold
@traitimpl IsMatrixM{Hyperbolic}
@traitimpl IsMatrixP{HnPoint}
@traitimpl IsMatrixV{HnTVector}
# (b) Sn is a MatrixManifold
@traitimpl IsEmbeddedM{Hyperbolic}
@traitimpl IsEmbeddedP{HnPoint}
@traitimpl IsEmbeddedV{HnTVector}

# Functions
# ---
doc"""
    distance(M,x,y)
Compute the Riemannian distance on $\mathcal M=\mathbb H^n$ embedded in
$\mathbb R^{n+1}$ can be computed as

$ d_{\mathbb H^n}(x,y) = \operatorname{acos} \bigl(\langle x,y\rangle_{\mathrm{M}}\bigr), $

where $\langle x,y\rangle_{\mathrm{M}} = -x_{n+1}y_{n+1} +
\displaystyle\sum_{k=1}^n x_ky_k$ denotes the Minkowski inner product
on $\mathbb R^{n+1}$.
"""
distance(M::Hyperbolic,x::HnPoint,y::HnPoint) = acos(-dotM(getValue(x), getValue(y) ))

doc"""
    dot(M,x,ξ,ν)
Compute the Riemannian inner product for two [`HnTVector`](@ref)s `ξ` and `ν`
from $T_x\mathcal M$ of the [`Hyperpolic Space`](@ref Hyperbolic)` M` given by
$\langle \xi, \nu \rangle_x = \langle \xi,\nu \rangle$, i.e. the inner product
in the embedded space $\mathbb R^{n+1}$.
"""
dot(M::Hyperbolic, x::HnPoint, ξ::HnTVector, ν::HnTVector) = dotM( getValue(ξ), getValue(ν) )

doc"""
    exp(M,x,ξ,[t=1.0])
Compute the exponential map on the [`Sphere`](@ref)` M`$=\mathbb S^n$ with
respect to the [`SnPoint`](@ref)` x` and the [`SnTVector`](@ref)` ξ`, which can
be shortened with `t` to `tξ`. The formula reads

$\exp_x\xi = \cosh(\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}})x + \sin(\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}})\frac{\xi}{\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}}}.$
"""
function exp(M::Hyperbolic,x::HnPoint,ξ::HnTVector,t::Float64=1.0)
  len = sqrt(dotM( getValue(ξ), getValue(ξ) ));
  if len < eps(Float64)
  	return x
  else
  	return HnPoint( cos(t*len)*getValue(x) + sin(t*len)/len*getValue(ξ) )
  end
end
doc"""
    log(M,x,y)
Compute the logarithmic map on the [`Sphere`](@ref)
$\mathcal M=\mathbb S^n$, i.e. the [`SnTVector`](@ref) whose corresponding
[`geodesic`](@ref) starting from [`SnPoint`](@ref)` x` reaches the
[`SnPoint`](@ref)` y` after time 1 on the [`Sphere`](@ref)` M`.
The formula reads for $x\neq -y$

$\log_x y = d_{\mathbb S^n}(x,y)\frac{y-\langle x,y\rangle x}{\lVert y-\langle x,y\rangle x \rVert_2}.$
"""
function log(M::Hyperbolic,x::HnPoint,y::HnPoint)
  scp = dotM( getValue(x), getValue(y) )
  ξvalue = getValue(y) + scp*getValue(x)
  ξvnorm = sqrt(dotM(getValue(x),getVaklue(y))-1);
  if (ξvnorm > eps(Float64))
    value = ξvalue*acos(-scp)/ξvnorm;
  else
    value = zeros( getValue(x) )
  end
  return HnTVector(value)
end
doc"""
    manifoldDimension(x)
returns the dimension of the [`Sphere`](@ref)` M`$=\mathbb S^n$, the
[`SnPoint`](@ref)` x`, itself embedded in $\mathbb R^{n+1}$, belongs to.
"""
manifoldDimension(x::HnPoint)::Integer = length( getValue(x) )-1
doc"""
    manifoldDimension(M)
returns the dimension of the [`Sphere`](@ref)` M`.
"""
manifoldDimension(M::Hyperbolic)::Integer = M.dimension
doc"""
    norm(M,x,ξ)
Computes the norm of the [`SnTVector`](@ref)` ξ` in the tangent space
$T_x\mathcal M$ at [`SnPoint`](@ref)` x` of the [`Sphere`](@ref)` M`.
"""
norm(M::Hyperbolic, x::HnPoint, ξ::HnTVector) = sqrt(dot(x,ξ,ξ))
doc"""
    parallelTransport(M,x,y,ξ)
Compute the paralllel transport of the [`SnTVector`](@ref) `ξ` from
the tangent space $T_x\mathcal M$ at [`SnPoint`](@ref)` x` to
$T_y\mathcal M$ at [`SnPoint`](@ref)` y` on the [`Sphere`](@ref)` M` provided
that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
The formula reads

$P_{x\to y}(\xi) = \xi - \frac{\langle \log_xy,\xi\rangle_x}{d^2_{\mathbb H^n}(x,y)}
\bigl(\log_xy + \log_yx \bigr).$
"""
function parallelTransport(M::Hyperbolic, x::HnPoint, y::HnPoint, ξ::HnTVector)
  ν = log(M,x,y);
  νL = norm(M,x,ν);
  if νL > 0
    ν = ν/νL;
	return HnTVector( getValue(ξ) - dot(M,x,ν,ξ)*( getValue(ν) + getValue(log(M,y,x))/νL) );
  else
    # if length of ν is 0, we have p=q and hence ξ is unchanged
    return ξ;
  end
end
# Display
# ---
show(io::IO, p::HnPoint) = print(io, "Hn($( getValue(p) ))")
show(io::IO, ξ::HnTVector) = print(io, "HnT($( getValue(ξ) ))")
# Helper
#
doc"""
    dotM(a,b)
computes the Minkowski inner product of two Vectors `a` and `b` of same length
`n+1`, i.e.

$\langle a,b\rangle_{\mathrm{M}} = -a_{n+1}b_{n+1} +
\displaystyle\sum_{k=1}^n a_kb_k.$
"""
dotM(a::Vector,b::Vector) = -a[end]*b[end] + sum( a[1:end-1].*b[1:end-1] )
