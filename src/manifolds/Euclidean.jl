#
#      Rn - The manifold of the n-dimensional (real valued) Euclidean space
#
# Manopt.jl, R. Bergmann, 2018-06-26
export Euclidean, RnPoint, RnTVector
import Base: exp, log, show
export distance, exp, log, norm, dot, manifoldDimension, show, getValue
# Types
# ---

doc"""
    Euclidean <: Manifold
The manifold $\mathcal M = \mathbb R^n$ of the $n$-dimensional Euclidean vector
space.
"""
struct Euclidean <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Euclidean(dimension::Int) = new("$dimension-dimensional Euclidean space",dimension,"R$dimension")
end

doc"""
    RnPoint <: MPoint
The point $x\in\mathbb M$ for $\mathbb M=\mathbb R^n$ represented by an
$n$-dimensional `Vector`.
"""
struct RnPoint <: MPoint
  value::Vector
  RnPoint(value::Vector) = new(value)
end
getValue(x::RnPoint) = x.value

doc"""
    RnTVector <: TVector
The point $\xi\in\mathbb M$ for $\mathbb M=\mathbb R^n$ represented by an
$n$-dimensional `Vector`.
"""
struct RnTVector <: TVector
  value::Vector
  RnTVector(value::Vector) = new(value)
end
getValue(ξ::RnTVector) = ξ.value

# Traits
# ---
# (a) Rn is a MatrixManifold
@traitimpl IsMatrixM{Euclidean}
@traitimpl IsMatrixP{RnPoint}
@traitimpl IsMatrixV{RnTVector}

# Functions
# ---
distance(M::Euclidean,x::RnPoint,y::RnPoint) = norm( getValue(x) - getValue(y) )
dot(M::Euclidean,x::RnPoint,ξ::RnTVector, ν::RnTVector) = dot( getValue(ξ) , getValue(ν) )
exp(M::Euclidean,x::RnPoint,ξ::RnTVector,t=1.0) = RnPoint(getValue(p) + t*getValue(ξ) )
log(M::Euclidean,x::RnPoint,y::RnPoint) = RnTVector( getValue(y) - getValue(x) )
manifoldDimension(x::RnPoint) = length( getValue(x) )
manifoldDimension(M::Euclidean) = M.dimension
norm(M::Euclidean,x::RnPoint, ξ::RnTVector) = norm(ξ.value)
parallelTransport(M::Euclidean, x::RnPoint, y::RnPoint, ξ::RnTVector) = ξ
#
#
# --- Display functions for the objects/types
show(io::IO, M::Euclidean) = print(io, "The Manifold $(M.name).");
show(io::IO, x::RnPoint) = print(io, "Rn($( getValue(x) ))");
show(io::IO, ξ::RnTVector) = print(io, "RnT($( getValue(ξ) )");
