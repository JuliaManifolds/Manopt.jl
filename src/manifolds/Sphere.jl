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
    SnVector <: TVector
A tangent vector $\xi \in T_x\mathcal M$ on the manifold
$\mathcal M = \mathbb S^n$ represented by a vector $\mathbb R^{n+1}$
that is orthogonal to the unit vector representing the base point $x$.
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
distance(M::Sphere,x::SnPoint,y::SnPoint) = acos(dot(getValue(x), getValue(y) ))
dot(M::Sphere, p::SnPoint, ξ::SnTVector, ν::SnTVector) = dot( getValue(ξ), getValue(ν) )
function exp(M::Sphere,x::SnPoint,ξ::SnTVector,t::Float64=1.0)
  len = norm( getValue(ξ) )
	if len < eps(Float64)
  	return copy(p)
	else
  	return SnPoint(cos(t*len)*getValue(x) + sin(t*len)/len*getValue(ξ) )
	end
end
function log(M::Sphere,x::SnPoint,y::SnPoint)
  scp = dot( getValue(x), getValue(y) )
  ξvalue = getValue(y) - scp*getValue(x)
  ξvnorm = norm(ξvalue)
  if (ξvnorm > eps(Float64))
    value = ξvalue*acos(scp)/ξvnorm;
  else
    value = zeros(p.value)
  end
  return SnTVector(value)
end
manifoldDimension(x::SnPoint)::Integer = length( getValue(x) )-1
manifoldDimension(M::Sphere)::Integer = M.dimension
norm(M::Sphere, x::SnPoint, ξ::SnTVector) = norm( getValue(ξ) )
function parallelTransport(M::Sphere, x::SnPoint, y::SnPoint, ξ::SnTVector)
  ν = log(M,p,q);
	νL = norm(M,ν);
	if νL > 0
	  ν = ν/νL;
		return SnTVector( getValue(ξ) - dot(M,p,ν,ξ)*( getValue(ν) + getValue(log(M,q,p))/νL) );
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
