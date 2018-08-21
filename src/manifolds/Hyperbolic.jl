#
#      Hn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
import LinearAlgebra: norm, dot
import Base: exp, log, show
export Hyperbolic, HnPoint, HnTVector, getValue
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export zeroTVector
#
# Type definitions
#

@doc doc"""
    Hyperbolic <: Manifold
The manifold $\mathbb H^n$ is the set

$\mathbb H^n = \Bigl\{x\in\mathbb R^{n+1}\Big|\langle x,x \rangle_{\mathrm{M}}= -x_{n+1}^2 + \displaystyle\sum_{k=1}^n x_k^2 = -1, x_{n+1} > 0\Bigr\},$
where $\langle\cdot,\cdot\rangle_{\mathrm{M}}$ denotes the Minkowski inner product,
and this inner product in the embedded space as Riemannian metric in the tangent bundle $T\mathbb H^n$.


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
@doc doc"""
    HnPoint <: MPoint
A point $x$ on the manifold $\mathbb H^n$ represented by a vector
$x\in\mathbb R^{n+1}$ with Minkowski inner product
$\langle x,x\rangle_{\mathrm{M}} = -1$.
"""
struct HnPoint <: MPoint
  value::Vector
  HnPoint(value::Vector) = new(value)
end
getValue(x::HnPoint) = x.value;

@doc doc"""
    HnTVector <: TVector
A tangent vector $\xi \in T_x\mathcal M$ on the manifold
$\mathbb H^n$.
"""
struct HnTVector <: TVector
  value::Vector
  HnTVector(value::Vector) = new(value)
end
getValue(ξ::HnTVector) = ξ.value;
# Traits
# ---
# (a) Hn is a MatrixManifold
@traitimpl IsMatrixM{Hyperbolic}
@traitimpl IsMatrixP{HnPoint}
@traitimpl IsMatrixV{HnTVector}
# (b) Hn is a MatrixManifold
@traitimpl IsEmbeddedM{Hyperbolic}
@traitimpl IsEmbeddedP{HnPoint}
@traitimpl IsEmbeddedV{HnTVector}

# Functions
# ---
@doc doc"""
    distance(M,x,y)
Compute the Riemannian distance on the [`Hyperbolic Space`](@ref Hyperbolic) $\mathbb H^n$ embedded in
$\mathbb R^{n+1}$ can be computed as

$ d_{\mathbb H^n}(x,y) = \operatorname{acosh} \bigl(-\langle x,y\rangle_{\mathrm{M}}\bigr), $

where $\langle x,y\rangle_{\mathrm{M}} = -x_{n+1}y_{n+1} +
\displaystyle\sum_{k=1}^n x_ky_k$ denotes the Minkowski inner product
on $\mathbb R^{n+1}$.
"""
distance(M::Hyperbolic,x::HnPoint,y::HnPoint) = acosh(-dotM(getValue(x), getValue(y) ))

@doc doc"""
    dot(M,x,ξ,ν)
Compute the Riemannian inner product for two [`HnTVector`](@ref)s `ξ` and `ν`
from $T_x\mathcal M$ of the [`Hyperpolic Space`](@ref Hyperbolic) $\mathbb H^n$ given by
$\langle \xi, \nu \rangle_x = \langle \xi,\nu \rangle$, i.e. the inner product
in the embedded space $\mathbb R^{n+1}$.
"""
dot(M::Hyperbolic, x::HnPoint, ξ::HnTVector, ν::HnTVector) = dotM( getValue(ξ), getValue(ν) )

@doc doc"""
    exp(M,x,ξ,[t=1.0])
Computes the exponential map on the [`Hyperpolic Space`](@ref Hyperbolic) $\mathbb H^n$ with
respect to the [`HnPoint`](@ref)` x` and the [`HnTVector`](@ref)` ξ`, which can
be shortened with `t` to `tξ`. The formula reads

$\exp_x\xi = \cosh(\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}}})x + \operatorname{sinh}(\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}}})\frac{\xi}{\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}}}}.$
"""
function exp(M::Hyperbolic,x::HnPoint,ξ::HnTVector,t::Float64=1.0)
  len = sqrt(dotM( getValue(ξ), getValue(ξ) ));
  if len < eps(Float64)
  	return x
  else
  	return HnPoint( cosh(t*len)*getValue(x) + sinh(t*len)/len*getValue(ξ) )
  end
end
@doc doc"""
    log(M,x,y)
Computes the logarithmic map on the [`Hyperbolic`](@ref) $\mathbb H^n$,
i.e., the [`HnTVector`](@ref) whose corresponding
[`geodesic`](@ref) starting from [`HnPoint`](@ref)` x` reaches the
[`HnPoint`](@ref)` y` after time 1 on the [`Hyperpolic Space`](@ref Hyperbolic) $\mathbb H^n$.
The formula reads for $x\neq -y$

$\log_x y = d_{\mathbb H^n}(x,y)\frac{y-\langle x,y\rangle_{\mathrm{M}} x}{\lVert y-\langle x,y\rangle_{\mathrm{M}} x \rVert_2}.$
"""
function log(M::Hyperbolic,x::HnPoint,y::HnPoint)
  scp = dotM( getValue(x), getValue(y) )
  ξvalue = getValue(y) + scp*getValue(x)
  ξvnorm = sqrt(dotM(getValue(x),getVaklue(y))-1);
  if (ξvnorm > eps(Float64))
    value = ξvalue*acosh(-scp)/ξvnorm;
  else
    value = zeros( getValue(x) )
  end
  return HnTVector(value)
end
@doc doc"""
    manifoldDimension(x)
returns the dimension of the [`Hyperbolic Space`](@ref Hyperbolic) $\mathbb H^n$, the
[`HnPoint`](@ref)` x`, itself embedded in $\mathbb R^{n+1}$, belongs to.
"""
manifoldDimension(x::HnPoint)::Integer = length( getValue(x) )-1
@doc doc"""
    manifoldDimension(M)
returns the dimension of the [`Hyperbolic Space`](@ref Hyperbolic) $\mathbb H^n$.
"""
manifoldDimension(M::Hyperbolic)::Integer = M.dimension
@doc doc"""
    norm(M,x,ξ)
Computes the norm of the [`HnTVector`](@ref)` ξ` in the tangent space
$T_x\mathcal M$ at [`HnPoint`](@ref)` x` of the
[`Hyperbolic Space`](@ref Hyperbolic) $\mathbb H^n$.
"""
norm(M::Hyperbolic, x::HnPoint, ξ::HnTVector) = sqrt(dot(x,ξ,ξ))
@doc doc"""
    parallelTransport(M,x,y,ξ)
Compute the paralllel transport of the [`HnTVector`](@ref)` ξ` from
the tangent space $T_x\mathcal M$ at [`HnPoint`](@ref)` x` to
$T_y\mathcal M$ at [`HnPoint`](@ref)` y` on the
[`Hyperbolic Space`](@ref Hyperbolic) $\mathbb H^n$ provided
that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
The formula reads

$P_{x\to y}(\xi) = \xi - \frac{\langle \log_xy,\xi\rangle_x}
{d^2_{\mathbb H^n}(x,y)}\bigl(\log_xy + \log_yx \bigr).$
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
@doc doc"""
    typicalDistance(M)
returns the typical distance on the [`Hyperbolic`](@ref)` Hn`: $\sqrt(n)$.
"""
typicalDistance(M::Hyperbolic) = sqrt(M.dimension);
@doc doc"""
    ξ = zeroTVector(M,x)
returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`HnPoint`](@ref) $x\in\mathbb H^n$ on the [`Hyperbolic`](@ref)` Hn`.
"""
zeroTVector(M::Hyperbolic, x::HnPoint) = HnTVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, p::HnPoint) = print(io, "Hn($( getValue(p) ))")
show(io::IO, ξ::HnTVector) = print(io, "HnT($( getValue(ξ) ))")
# Helper
#
@doc doc"""
    dotM(a,b)
computes the Minkowski inner product of two Vectors `a` and `b` of same length
`n+1`, i.e.

$\langle a,b\rangle_{\mathrm{M}} = -a_{n+1}b_{n+1} +
\displaystyle\sum_{k=1}^n a_kb_k.$
"""
dotM(a::Vector,b::Vector) = -a[end]*b[end] + sum( a[1:end-1].*b[1:end-1] )
