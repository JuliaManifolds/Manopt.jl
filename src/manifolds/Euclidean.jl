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
space. We employ the notation $\langle\cdot,\cdot,\rangle$ for the inner product
and $\lVert\cdot\rVert_2$ for its induced norm.
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
doc"""
    distance(M,x,y)
Computes the Euclidean distance $\lVert x - y\rVert$
"""
distance(M::Euclidean,x::RnPoint,y::RnPoint) = norm( getValue(x) - getValue(y) )
doc"""
    dot(M,x,ξ,ν)
Computes the Euclidean inner product of `ξ` and `ν`, i.e.
$\langle\xi,\nu\rangle = \displaystyle\sum_{k=1}^n \xi_k\nu_k$.
"""
dot(M::Euclidean,x::RnPoint,ξ::RnTVector, ν::RnTVector) = dot( getValue(ξ) , getValue(ν) )
doc"""
    exp(M,x,ξ)
Computes the exponential map, i.e. $x+\xi$.
"""
exp(M::Euclidean,x::RnPoint,ξ::RnTVector,t=1.0) = RnPoint(getValue(p) + t*getValue(ξ) )
doc"""
    log(M,x,y)
Computes the logarithmic map, i.e. $y-x$.
"""
log(M::Euclidean,x::RnPoint,y::RnPoint) = RnTVector( getValue(y) - getValue(x) )
"""
    manifoldDimension(x)
Returns the manifold dimension, i.e. the length of the vector `x`.
"""
manifoldDimension(x::RnPoint) = length( getValue(x) )
"""
    manifoldDimension(M)
Returns the manifold dimension, i.e. the length of the vectors stored
in `M.dimension`.
"""
manifoldDimension(M::Euclidean) = M.dimension
doc"""
    norm(M,x,ξ)
Computes the length of the tangent vector `ξ` in the tangent
space $T_x\mathcal M$ of `x` on the Eclidean space `M`, i.e. $\lVert\xi\rVert$.
"""
norm(M::Euclidean,x::RnPoint, ξ::RnTVector) = norm(ξ.value)
"""
    parallelTransport(M,x,y,ξ)
Computes the parallel transport, which is in Eulidean space the identity.
"""
parallelTransport(M::Euclidean, x::RnPoint, y::RnPoint, ξ::RnTVector) = ξ
#
#
# --- Display functions for the objects/types
show(io::IO, M::Euclidean) = print(io, "The Manifold $(M.name).");
show(io::IO, x::RnPoint) = print(io, "Rn($( getValue(x) ))");
show(io::IO, ξ::RnTVector) = print(io, "RnT($( getValue(ξ) )");
