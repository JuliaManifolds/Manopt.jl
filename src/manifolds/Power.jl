#
#      Powermanifold – an array of points of _one_ manifold
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show

export Power, PowPoint, PowTVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export zeroTVector
export show, getValue
@doc doc"""
    Power{M<:Manifold} <: Manifold
a power manifold $\mathcal M = \mathcal N^m$, where $m$ can be an integer or an
integer vector. Its abbreviatio is `Pow`.
"""
struct Power{M<:Manifold} <: Manifold
  name::String
  manifold::M
  dims::Array{Int,1}
  dimension::Int
  abbreviation::String
  Power{M}(mv::M,dims::Array{Int,1}) where {M<:Manifold} = new(string("A Power Manifold of ",mv.name,"."),
    mv,dims,prod(dims)*manifoldDimension(mv),string("Pow(",m.abbreviation,",",repr(dims),")") )
end
@doc doc"""
    PowPoint <: MPoint
A point on the power manifold $\mathcal M = \mathcal N^m$ represented by a vector or array of [`MPoint`](@ref)s.
"""
struct PowPoint <: MPoint
  value::Array{T,N} where N where T<:MPoint
  PowPoint(v::Array{T,N} where N where T<:MPoint) = new(v)
end
getValue(x::PowPoint) = x.value;

@doc doc"""
    PowTVector <: TVector
A tangent vector on the power manifold $\mathcal M = \mathcal N^m$ represented by a vector of [`TVector`](@ref)s.
"""
struct PowTVector <: TVector
  value::Array{T,N} where N where T <: TVector
  PowTVector(value::Array{T,N} where N where T <: TVector) = new(value)
end
getValue(ξ::PowTVector) = ξ.value
# Functions
# ---
"""
    addNoise(M,x,δ)
computes a vectorized version of addNoise, and returns the noisy [`PowPoint`](@ref).
"""
addNoise(M::Power, x::PowPoint,σ) = PowPoint([addNoise.(M.manifold,p.value,σ)])
"""
    distance(M,x,y)
computes a vectorized version of distance, and the induced norm from the metric [`dot`](@ref).
"""
distance(M::Power, x::PowPoint, y::PowPoint) = sqrt(sum( distance.(M.manifold, getValue(x), getValue(y) ).^2 ))
"""
    dot(M,x,ξ,ν)
computes the inner product as sum of the component inner products on the [`Power`](@ref).
"""
dot(M::Power, x::PowPoint, ξ::PowTVector, ν::PowTVector) = sum(dot.(M.manifold,getValue(x), getValue(ξ), getValue(ν) ))
"""
    exp(M,x,ξ)
computes the product exponential map on the [`Power`](@ref) and returns the corresponding [`PowPoint`](@ref).
"""
exp(M::Power, x::PowPoint, ξ::PowTVector, t::Number=1.0) = PowPoint( exp.(M.manifold, getValue(p) , getValue(ξ) ))
"""
   log(M,x,y)
computes the product logarithmic map on the [`Power`](@ref) and returns the corresponding [`PowTVector`](@ref).
"""
log(M::Power, x::PowPoint, y::PowPoint)::PowTVector = PowTVector(log.(M.manifold, getValue(p), getValue(q) ))
"""
    manifoldDimension(x)
returns the (product of) dimension(s) of the [`Power`](@ref) the [`PowPoint`](@ref)`x` belongs to.
"""
manifoldDimension(p::PowPoint) = prod(manifoldDimension.( getValue(p) ) )
"""
    manifoldDimension(M)
returns the (product of) dimension(s) of the [`Power`](@ref)` M`.
"""
manifoldDimension(M::Power) = M.dims * manifoldDimension(M.manifold)
"""
    norm(M,x,ξ)
norm of the [`PowTVector`]` ξ` induced by the metric on the manifold components
of the [`Power`](@ref)` M`.
"""
norm(M::Power, x::PowPoint, ξ::PowTVector) = sqrt( dot(M,x,ξ,ξ) )
"""
    parallelTransport(M,x,ξ)
computes the product parallelTransport map on the [`Power`](@ref) and returns the corresponding [`PowTVector`](@ref).
"""
parallelTransport(M::Power, x::PowPoint, y::PowPoint, ξ::PowTVector) = PowTVector( parallelTransport.(M.manifold, getValue(x), getValue(y), getValue(ξ)) )
@doc doc"""
    typicalDistance(M)
returns the typical distance on the [`Power`](@ref)` Pow`, which is based on
the elementwise bae manifold.
"""
typicalDistance(M::Power) = sqrt( M.dims ) * typicalDistance(M.manifold);
@doc doc"""
    ξ = zeroTVector(M,x)
returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`PowPoint`](@ref) $x\in\mathcal M$ on the [`Power`](@ref)` M`.
"""
zeroTVector(M::Power, x::PowPoint) = PowTVector( zeroTVector.(M.manifold, getValue(x) )  )
#
#
# Display functions for the structs
show(io::IO, M::Power) = print(io,string("The Power Manifold of ",repr(M.manifold), " of size ",repr(M.dims),".") );
show(io::IO, p::PowPoint) = print(io,string("Pow[",join(repr.(p.value),", "),"]"));
show(io::IO, ξ::PowTVector) = print(io,String("ProdT[", join(repr.(ξ.value),", "),"]"));
