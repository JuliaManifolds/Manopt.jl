#
#      Sn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
import Base.LinAlg: norm, dot
import Base: exp, log, show
export Sphere, SnPoint, SnTVector,show, getValue
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
#
# Type definitions
#

doc"""
    Sphere <: MatrixManifold
The manifold $\mathcal M = \mathbb S^n$ of unit vectors in $\mathbb R^{n+1}$
"""
struct Sphere <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Sphere(dimension::Int) = new("$dimension-Sphere",dimension,"S$(dimension-1)")
end
doc"""
    SnPoint <: MPoint
A point $x$ on the manifold $\mathcal M = \mathbb S^n$ represented by a unit
vector from $\mathbb R^{n+1}$
"""
struct SnPoint <: MPoint
  value::Vector
  SnPoint(value::Vector) = new(value)
end
getValue(x::SnPoint) = x.value;

doc"""
    SnTVector <: TVector
A tangent vector $\xi \in T_x\mathcal M$ on the manifold
$\mathcal M = \mathbb S^n$. For the representation the tangent space can be
given as $T_x\mathbb S^n = \bigl\{\xi \in \mathbb R^{n+1}
\big| \langle x,\xi\rangle = 0\bigr\}$, where $\langle\cdot,\cdot\rangle$
denotes the Euclidean inner product on $\mathbb R^{n+1}$.
"""
struct SnTVector <: TVector
  value::Vector
  SnTVector(value::Vector) = new(value)
end
getValue(ξ::SnTVector) = ξ.value;
# Traits
# ---
# (a) Sn is a MatrixManifold
@traitimpl IsMatrixM{Sphere}
@traitimpl IsMatrixP{SnPoint}
@traitimpl IsMatrixV{SnTVector}

# Functions
# ---
doc"""
    distance(M,x,y)
Compute the Riemannian distance on $\mathcal M=\mathbb S^n$ embedded in
$\mathbb R^{n+1}$ can be computed as

$ d_{\mathcal S^n}(x,y) = \operatorname{acos} \bigl(\langle x,y\rangle\bigr), $

where $\langle\cdot,\cdot\rangle$ denotes the Euclidean inner product
on $\mathbb R^{n+1}$.
"""
distance(M::Sphere,x::SnPoint,y::SnPoint) = acos(dot(getValue(x), getValue(y) ))

doc"""
    dot(M,x,ξ,ν)
Compute the Riemannian inner product for two [`SnTVector`](@ref)s `ξ` and `ν`
from $T_x\mathcal M$ of the [`Sphere`](@ref) `M` given by
$\langle \xi, \nu \rangle_x = \langle \xi,\nu \rangle$, i.e. the inner product
in the embedded space $\mathbb R^{n+1}$.
"""
dot(M::Sphere, x::SnPoint, ξ::SnTVector, ν::SnTVector) = dot( getValue(ξ), getValue(ν) )

doc"""
    exp(M,x,ξ,[t=1.0])
Compute the exponential map on the [`Sphere`](@ref) $\mathcal M=\mathbb S^n$ with
respect to the [`SnPoint`](@ref) `x` and the [`SnTVector`](@ref) `ξ`, which can
be shortened with `t` to `tξ`. The formula reads

$\exp_x\xi = \cos(\lVert\xi\rVert_2)x + \sin(\lVert\xi\rVert_2)\frac{\xi}{\lVert\xi\rVert_2}.$
"""
function exp(M::Sphere,x::SnPoint,ξ::SnTVector,t::Float64=1.0)
  len = norm( getValue(ξ) )
	if len < eps(Float64)
  	return x
	else
  	return SnPoint(cos(t*len)*getValue(x) + sin(t*len)/len*getValue(ξ) )
	end
end
doc"""
    log(M,x,y)
Compute the logarithmic map on the [`Sphere`](@ref)
$\mathcal M=\mathbb S^n$, i.e. the [`SnTVector`](@ref) whose corresponding
[`geodesic`](@ref) starting from [`SnPoint`](@ref) `x` reaches the
[`SnPoint`](@ref) `y` after time 1 on the [`Sphere`](@ref) `M`.
The formula reads for $x\neq -y$

$\log_x y = d_{\mathbb S^n}(x,y)\frac{y-\langle x,y\rangle x}{\lVert y-\langle x,y\rangle x \rVert_2}.$
"""
function log(M::Sphere,x::SnPoint,y::SnPoint)
  scp = dot( getValue(x), getValue(y) )
  ξvalue = getValue(y) - scp*getValue(x)
  ξvnorm = norm(ξvalue)
  if (ξvnorm > eps(Float64))
    value = ξvalue*acos(scp)/ξvnorm;
  else
    value = zeros( getValue(x) )
  end
  return SnTVector(value)
end
doc"""
    manifoldDimension(x)
returns the dimension of the [`Sphere`](@ref) $\mathbb S^n$, the
[`SnPoint`](@ref) `x` embedded in $\mathbb R^{n+1}$.
"""
manifoldDimension(x::SnPoint)::Integer = length( getValue(x) )-1
doc"""
    manifoldDimension(M)
returns the dimension of the [`Sphere`](@ref) `M`.
"""
manifoldDimension(M::Sphere)::Integer = M.dimension
doc"""
    norm(M,x,ξ)
Computes the norm of the [`SnTVector`](@ref) `ξ` in the tangent space
$T_x\mathcal M$ at [`SnPoint`](@ref) `x` of the [`Sphere`](@ref) `M`.
"""
norm(M::Sphere, x::SnPoint, ξ::SnTVector) = norm( getValue(ξ) )
doc"""
    parallelTransport(M, x, y, ξ)
Compute the paralllel transport of the [`SnTVector`](@ref) `ξ` from
the tangent space $T_x\mathcal M$ at [`SnPoint`](@ref) `x` to
$T_y\mathcal M$ at [`SnPoint`](@ref) `y` on the [`Sphere`](@ref) `M` provided
that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
The formula reads

$P_{x\to y}(\xi) = \xi - \frac{\langle \log_xy,\xi\rangle_x}{d^2_{\mathbb S^n}(x,y)}
\bigl(\log_xy + \log_yx \bigr).$
"""
function parallelTransport(M::Sphere, x::SnPoint, y::SnPoint, ξ::SnTVector)
  ν = log(M,x,y);
	νL = norm(M,x,ν);
	if νL > 0
	  ν = ν/νL;
		return SnTVector( getValue(ξ) - dot(M,x,ν,ξ)*( getValue(ν) + getValue(log(M,y,x))/νL) );
  else
	  # if length of ν is 0, we have p=q and hence ξ is unchanged
	  return ξ;
	end
end
# Display
# ---
show(io::IO, M::Sphere) = print(io, "The Manifold $(M.name).")
show(io::IO, p::SnPoint) = print(io, "Sn($( getValue(p) ))")
show(io::IO, ξ::SnTVector) = print(io, "SnT($( getValue(ξ) ))")
