#
#      S1 - The manifold of the 1-dimensional sphere represented by angles
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show

export Circle, S1Point, S1TVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export show, getValue

export symRem
# Types
# ---
doc"""
    Circle <: Manifold
The one-dimensional manifold $\mathcal M = \mathbb S^1$ represented by angles.
Note that one can also use the $n$-dimensional sphere with $n=1$ to obtain the
same manifold represented by unit vectors in $\mathbb R^2$.
Its abbreviation is `S1`, since the abbreviation of Sn is in variables always `Sn`.
"""
struct Circle <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Circle() = new("1-Sphere as angles",1,"S1")
end
doc"""
    S1Point <: MPoint
a point $x\in\mathbb S^1$ represented by an angle `getValue(x)`$\in[-\pi,\pi)$.
"""
struct S1Point <: MPoint
  value::Float64
  S1Point(value::Float64) = new(value)
end
getValue(x::S1Point) = x.value
doc"""
    S1TVector <: TVector
a tangent vector $\xi\in\mathbb S^1$ represented by a real valiue
`getValue($\xi$)`$\in\mathbb R$.
"""
struct S1TVector <: TVector
  value::Float64
  S1TVector(value::Float64) = new(value)
end
getValue(ξ::S1TVector) = ξ.value
# Traits
# ---
#(a) S1 is a matrix manifold
@traitimpl IsMatrixM{Circle}
@traitimpl IsMatrixP{S1Point}
@traitimpl IsMatrixV{S1TVector}

# Functions
# ---
addNoise(M::Circle, x::S1Point,σ::Real) = S1Point(mod(getValue(x)-pi+σ*randn(),2*pi)+pi)
distance(M::Circle, x::S1Point,y::S1Point) = abs( symRem(getValue(y) - getValue(x)) )
dot(M::Circle, x::S1Point, ξ::S1TVector, ν::S1TVector) = getValue(ξ)*getValue(ν)
exp(M::Circle, x::S1Point,ξ::S1TVector,t::Float64=1.0) = S1Point( symRem(getValue(x) + t*getValue(ξ)) )
log(M::Circle, x::S1Point,y::S1Point)::S1TVector = S1TVector(symRem( getValue(y) - getValue(x) ))
manifoldDimension(x::S1Point) = 1
manifoldDimension(M::Circle) = 1
norm(M::Circle, x::S1Point, ξ::S1TVector)::Float64 = abs( getValue(ξ) )
parallelTransport(M::Circle, x::S1Point, y::S1Point, ξ::S1TVector) = ξ
# Display
# ---
show(io::IO, M::Circle) = print(io, "The Manifold S1 consisting of angles");
show(io::IO, x::S1Point) = print(io, "S1($( getValue(x) ))");
show(io::IO, ξ::S1TVector) = print(io, "S1T($( getValue(ξ) ))");
# little Helpers
# ---
"""
    symRem(x,y,[T=pi])
symmetric remainder with respect to the interall 2*`T`
"""
function symRem(x::Float64, T::Float64=Float64(pi))::Float64
  return (x+T)%(2*T) - T
end
