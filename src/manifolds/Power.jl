#
#      Powermanifold – an array of points of _one_ manifold
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show, getindex, setindex!, copy

export Power, PowPoint, PowTVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export zeroTVector
export show, getValue, setindex!, getindex,copy
@doc doc"""
    Power{M<:Manifold} <: Manifold
a power manifold $\mathcal M = \mathcal N^m$, where $m$ can be an integer or an
integer vector. Its abbreviatio is `Pow`.
"""
struct Power <: Manifold
  name::String
  manifold::M where {M <: Manifold}
  dims::Tuple{Int}
  dimension::Int
  abbreviation::String
  Power(m::mT,dims::Tuple{Int}) where {mT<:Manifold} = new(string("A Power Manifold of ",m.name,"."),
    m,dims,prod(dims)*manifoldDimension(m),string("Pow(",m.abbreviation,",",repr(dims),")") )
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
getindex(x::PowPoint, i) = getindex(x.value,i)
setindex!(x::PowPoint, p::P where {P <: MPoint},i) = setindex!(x.value,p,i)
copy(x::PowPoint) = PowPoint(copy(x.value))
@doc doc"""
    PowTVector <: TVector
A tangent vector on the power manifold $\mathcal M = \mathcal N^m$ represented by a vector of [`TVector`](@ref)s.
"""
struct PowTVector <: TVector
  value::Array{T,N} where N where T <: TVector
  PowTVector(value::Array{T,N} where N where T <: TVector) = new(value)
end
getValue(ξ::PowTVector) = ξ.value
getindex(ξ::PowTVector, i) = getindex(ξ.value,i)
setindex!(ξ::PowTVector, ν::T where {T <: TVector},i) = setindex!(ξ.value,ν,i)
copy(ξ::PowTVector) = PowTVector(copy(ξ.value))
# Functions
# ---
"""
    addNoise(M,x,δ)
computes a vectorized version of addNoise, and returns the noisy [`PowPoint`](@ref).
"""
addNoise(M::Power, x::PowPoint,σ) = PowPoint([addNoise.( Ref(M.manifold),x.value,Ref(σ) )])

"""
    distance(M,x,y)
computes a vectorized version of distance, and the induced norm from the metric [`dot`](@ref).
"""
distance(M::Power, x::PowPoint, y::PowPoint) = sqrt(sum( distance.( Ref(M.manifold), getValue(x), getValue(y) ).^2 ))

"""
    dot(M,x,ξ,ν)
computes the inner product as sum of the component inner products on the [`Power`](@ref)` manifold`.
"""
dot(M::Power, x::PowPoint, ξ::PowTVector, ν::PowTVector) = sum(dot.(Ref(M.manifold),getValue(x), getValue(ξ), getValue(ν) ))

"""
    exp(M,x,ξ)
computes the product exponential map on the [`Power`](@ref) and returns the corresponding [`PowPoint`](@ref).
"""
exp(M::Power, x::PowPoint, ξ::PowTVector, t::Number=1.0) = PowPoint( exp.(Ref(M.manifold), getValue(x) , getValue(ξ),t))

"""
   log(M,x,y)
computes the product logarithmic map on the [`Power`](@ref) and returns the corresponding [`PowTVector`](@ref).
"""
log(M::Power, x::PowPoint, y::PowPoint)::PowTVector = PowTVector(log.(Ref(M.manifold), getValue(x), getValue(y) ))

"""
    manifoldDimension(x)
returns the (product of) dimension(s) of the [`Power`](@ref) the [`PowPoint`](@ref)`x` belongs to.
"""
manifoldDimension(x::PowPoint) = prod(manifoldDimension.( getValue(x) ) )

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
parallelTransport(M::Power, x::PowPoint, y::PowPoint, ξ::PowTVector) = PowTVector( parallelTransport.(Ref(M.manifold), getValue(x), getValue(y), getValue(ξ)) )

"""
    typicalDistance(M)
returns the typical distance on the [`Power`](@ref)` Pow`, which is based on
the elementwise bae manifold.
"""
typicalDistance(M::Power) = sqrt( Float64(sum(M.dims)) ) * typicalDistance(M.manifold);
@doc doc"""
    ξ = zeroTVector(M,x)
returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`PowPoint`](@ref) $x\in\mathcal M$ on the [`Power`](@ref)` M`.
"""
zeroTVector(M::Power, x::PowPoint) = PowTVector( zeroTVector.(Ref(M.manifold), getValue(x) )  )
#
#
# Display functions for the structs
show(io::IO, M::Power) = print(io,string("The Power Manifold of ",repr(M.manifold), " of size ",repr(M.dims),".") );
show(io::IO, p::PowPoint) = print(io,string("Pow[",join(repr.(p.value),", "),"]"));
show(io::IO, ξ::PowTVector) = print(io,String("ProdT[", join(repr.(ξ.value),", "),"]"));
