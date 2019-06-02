#
#      Powermanifold – an array of points of _one_ manifold
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show, getindex, setindex!, copy, size, cat, hcat, vcat, repeat, ndims

export Power, PowPoint, PowTVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export zeroTVector
export typeofMPoint, typeofTVector, randomMPoint, randomTVector
export validateMPoint, validateTVector
export show, getValue, setindex!, getindex,copy, size, repeat, ndims
@doc doc"""
    Power{M<:Manifold} <: Manifold

A power manifold $\mathcal M = \mathcal N^n$, where $n$ can be an integer or an
integer vector.

# Abbreviation

`Pow`

# Constructors
    Power(M,n)

construct the power manifold $\mathcal M^n$ for a [`Manifold`](@ref) `M`
and a natural number `n`.

    Power(M,n)

construct the power manifold $\mathcal M^{n_1\times n_2\times\cdots\times n_d}$
for a [`Manifold`](@ref) `M` and a `Tuple` or `Array` `n` of natural numbers.

"""
struct Power{mT <: Manifold} <: Manifold
  name::String
  manifold::mT
  powerSize::NTuple{N,Int} where N
  abbreviation::String
  Power{mT}(M::mT, pSize::NTuple{N,Int}) where {mT <: Manifold,N} = new(string("The Power Manifold of ",repr(M),
    " to the power ",repr(pSize),"."), M, pSize, string("Pow(",M.abbreviation,"^",repr(pSize),")")
  )
end
Power(M::mT, n::Int ) where {mT <: Manifold} = Power{mT}(M, (n,) )
Power(M::mT, n::NTuple{N,<:Int}) where {mT <: Manifold,N} = Power{mT}(M, n)
@doc doc"""
    PowPoint <: MPoint

A point on the power manifold $\mathcal M = \mathcal N^n$ represented by
an array (of size `n`) of [`MPoint`](@ref)s.
"""
struct PowPoint{P,N} <: MPoint where {P <: MPoint,N}
  value::Array{P,N}
  PowPoint{P,N}(v) where { P <: MPoint,N} = new(v)
end
PowPoint(v::Array{P,N}) where {P <: MPoint,N} = PowPoint{P,N}(v)
getValue(x::PowPoint{P,N}) where {P <: MPoint, N} = x.value;
getindex(x::PowPoint, i...) = PowPoint(getindex( getValue(x) ,i...))
getindex(x::PowPoint, i::Union{Integer, CartesianIndex},
  I::Union{Integer, CartesianIndex}...) = getindex(getValue(x),i,I...)
setindex!(x::PowPoint, y::PowPoint, kv...) = setindex!(getValue(x),getValue(y),kv...)
setindex!(x::PowPoint, kv...) = setindex!(getValue(x),kv...)
@inline ndims(x::PowPoint{P,N}) where {P<:MPoint,N} = ndims(x.value)
function repeat(x::PowPoint{P,N}; inner=ntuple(y->1, Val(ndims(x))), outer=ntuple(y->1, Val(ndims(x))) ) where {P<:MPoint,N}
    b = repeat(x.value; inner=inner, outer=outer)
    return PowPoint(b)
end
repeat(x::PowPoint, counts::Integer...) = repeat(x; outer=counts)
size(x::PowPoint,k...) = size(getValue(x),k...)
copy(x::PowPoint) = PowPoint(copy(getValue(x)))
ndims(x::PowPoint) = ndims( getValue(x) )
@doc doc"""
    PowTVector <: TVector

A tangent vector on the power manifold $\mathcal M = \mathcal N^n$ represented
by an array (of size `n`) of [`TVector`](@ref)s.
"""
struct PowTVector{T,N} <: TVector where {T <: TVector, N}
  value::Array{T,N}
  PowTVector{T,N}(v) where {T <: TVector, N} = new(v)
end
PowTVector(v::Array{T,N}) where {T <: TVector, N} = PowTVector{T,N}(v)
getValue(ξ::PowTVector{T,N}) where {T <: TVector, N} = ξ.value
getindex(ξ::PowTVector,i...) = PowTVector(getindex(ξ.value,i...))
getindex(ξ::PowTVector, i::Union{Integer, CartesianIndex},
  I::Union{Integer, CartesianIndex}...) = getindex(getValue(ξ),i,I...)
setindex!(ξ::PowTVector, ν::T where {T <: TVector},i...) = setindex!(ξ.value,ν,i...)
function repeat(ξ::PowTVector{T,N}; inner=ntuple(t->1, Val(ndims(ξ))), outer=ntuple(t->1, Val(ndims(ξ))) ) where {T <: TVector,N}
    b = repeat(ξ.value; inner=inner, outer=outer)
    return PowTVector(b)
end
repeat(ξ::PowTVector, counts::Integer...) = repeat(ξ; outer=counts)
ndims(ξ::PowTVector) = ndims( getValue(ξ) )
size(ξ::PowTVector) = size(getValue(ξ))
copy(ξ::PowTVector) = PowTVector(copy(ξ.value))
# Functions
# ---
function adjointJacobiField(M::Power,x::PowPoint,y::PowPoint,t::Float64,η::PowTVector,β::Function=βDgx)::PowTVector
    return PowTVector( adjointJacobiField.(Ref(M.manifold), x.value, y.value, Ref(t), η.value ,Ref(β) ) )
end

"""
    distance(M,x,y)

compute a vectorized version of distance on the [`Power`] manifold `M` for two
[`PowPoint`](@ref) `x` and `y`.
"""
distance(M::Power, x::PowPoint, y::PowPoint) = sqrt(sum( abs.(distance.( Ref(M.manifold), x.value, y.value )).^2 ))

"""
    dot(M,x,ξ,ν)

compute the inner product as sum of the component inner products on the
[`Power`](@ref) manifold `M`.
"""
dot(M::Power, x::PowPoint, ξ::PowTVector, ν::PowTVector) = sum(dot.(Ref(M.manifold), x.value, ξ.value, ν.value ))

"""
    exp(M,x,ξ[, t=1.0])

compute the product exponential map on the [`Power`](@ref) manifold `M` and
return the corresponding [`PowPoint`](@ref).
"""
exp(M::Power, x::PowPoint, ξ::PowTVector, t::Float64=1.0) = PowPoint(
    exp.(Ref(M.manifold), x.value, ξ.value,t)
)

function jacobiField(M::Power,x::PowPoint,y::PowPoint,t::Float64,η::PowTVector,β::Function=βDgx)::PowTVector
    return PowTVector(
        jacobiField.(Ref(M.manifold), x.value, y.value, Ref(t), η.value, Ref(β))
    )
end

"""
    log(M,x,y)

compute the product logarithmic map on the [`Power`](@ref) manifold `M` and
return the corresponding [`PowTVector`](@ref).
"""
log(M::Power, x::PowPoint, y::PowPoint)::PowTVector = PowTVector(log.(Ref(M.manifold), getValue(x), getValue(y) ))

"""
    manifoldDimension(x)

return the (product of) dimension(s) of the [`Power`](@ref) the
[`PowPoint`](@ref)`x` belongs to.
"""
manifoldDimension(x::PowPoint) = sum(manifoldDimension.( getValue(x) ) )

"""
    manifoldDimension(M)

return the (product of) dimension(s) of the [`Power`](@ref) manifold `M`.
"""
manifoldDimension(M::Power) = prod(M.powerSize) * manifoldDimension(M.manifold)

"""
    norm(M,x,ξ)

compute the norm of the [`PowTVector`] `ξ` induced by the metric on the manifold
components of the [`Power`](@ref) manifold `M`.
"""
norm(M::Power, x::PowPoint, ξ::PowTVector) = sqrt( dot(M,x,ξ,ξ) )

"""
    parallelTransport(M,x,y,ξ)

compute the product parallelTransport map on the [`Power`](@ref) manifold `M`
from the [`PowPoint`](@ref) `x` to `y` of the [`PowTVector`](@ref) `ξ`.
"""
parallelTransport(M::Power, x::PowPoint, y::PowPoint, ξ::PowTVector) = 
    PowTVector( parallelTransport.(Ref(M.manifold), getValue(x), getValue(y), getValue(ξ)) )

@doc doc"""
    randomMPoint(M)

construct a random point on the [`Power`](@ref) manifold `M`, by creating
`n` points on the
[`Manifold`](@ref) `M.manifold` as corresponding [`PowPoint`](@ref).
Optional values are passed down.
"""
randomMPoint(M::Power,options...) = PowPoint( [randomMPoint(M.manifold,options...) for i in CartesianIndices(M.powerSize)] )

@doc doc"""
    randomTVector(M,x)

construct a random tangent vector on the [`Power`](@ref) manifold `M`, by creating
`n` tangent vectors on the
[`Manifold`](@ref) `M.manifold` at the enrties of the [`PowPoint`](@ref) `x`.
Optional values are passed down.
"""
randomTVector(M::Power,x::PowPoint,options...) = PowTVector(
    [randomTVector(M.manifold, getValue(x)[i] , options...)
        for i in CartesianIndices(getValue(x)) ]
)
randomTVector(M::Power,x::PowPoint,s::Symbol,options...) = PowTVector(
    [randomTVector(M.manifold, getValue(x)[i],s,options...)
        for i in CartesianIndices(getValue(x)) ]
)

@doc doc"""
    (Ξ,κ) = tangentONB(M,x,y)

compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi=\log_xy$
is the first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$, $\dot
g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

# See also
 [`jacobiField`](@ref), [`adjointJacobiField`](@ref).
"""
tangentONB(M::Power, x::PowPoint, y::PowPoint) = tangentONB(M,x,log(M,x,y))
function tangentONB(M::Power, x::PowPoint, ξ::PowTVector)
    A = collect(zip( tangentONB.(Ref(M.manifold),getValue(x), getValue(ξ) )... ) )
    κ = vcat( A[2]... )
    Ξ = [ zeroTVector(M,x) for i in Tuple(CartesianIndices(M.powerSize)) for j=1:manifoldDimension(M.manifold)]
    l = 1
    for i in Tuple(CartesianIndices(M.powerSize))
        for j=1:manifoldDimension(M.manifold)
            Ξ[l][i] = A[1][i][j]
            l=l+1
        end
    end
    return Ξ,κ
end
doc"""
    typeofTVector(P)

returns the type of the [`PowTVector`](@ref) that all tangent vectors of the
[`PowPoint`](@ref) `P` have.
"""
typeofTVector(::Type{PowPoint{P,N}}) where {P <: MPoint, N} = PowTVector{typeofTVector(P),N}
doc"""
    typeofMPoint(T)

return the type of the [`PowPoint`](@ref) that is the base point of the
[`PowTVector`](@ref) `T`.
"""
typeofMPoint(::Type{PowTVector{T,N}}) where {T <: TVector, N} = PowPoint{typeofMPoint(T),N}

"""
    typicalDistance(M)

returns the typical distance on the [`Power`](@ref) manifold `M`, which is
based on the elementwise manifold.
"""
typicalDistance(M::Power) = sqrt( prod(M.powerSize) ) * typicalDistance(M.manifold)

doc"""
    validateMPoint(M,x)

validate, that the [`PowPoint`](@ref) `x` is a point on the [`Power`](@ref)
manifold `M`, i.e. that the array dimensions are correct and that all elements
are valid points on the elements manifold.
"""
function validateMPoint(M::Power, x::PowPoint)
    if size(getValue(x)) ≠ M.powerSize
        throw( DomainError(
        " The power manifold point $x is not on $(M.name) since its array dimensions of the elements ($(size(getValue(x)))) does not fit the power ($(M.powerSize))."
        ))
    end
    validateMPoint.(Ref(M.manifold),getValue(x))
    return true
end

doc"""
    validateTVector(M,x,ξ)

validate, that the [`ProdTVector`](@ref) `ξ` is a valid tangent vector to the
[`ProdPoint`](@ref) `x` on the [`Product`](@ref) manifold `M`, i.e. that all
three array dimensions match and this validation holds elementwise.
"""
function validateTVector(M::Power, x::PowPoint, ξ::PowTVector)
    if (size(getValue(x)) ≠ size(getValue(ξ))) || (size(getValue(ξ)) ≠ M.powerSize)
        throw( DomainError(
        "The three dimensions of the $(M.name), the point x ($(size(getValue(x)))), and the tangent vector ($(size(getValue(ξ)))) don't match."
        ))
    end
    validateTVector.(Ref(M.manifold),getValue(x),getValue(ξ))
    return true
end

@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`PowPoint`](@ref) $x\in\mathcal M$ on the [`Power`](@ref) manifold `M`.
"""
zeroTVector(M::Power{Mt}, x::PowPoint{P,N}) where {Mt <: Manifold, P <: MPoint, N} = PowTVector{typeofTVector(P),N}( zeroTVector.(Ref(M.manifold), getValue(x) )  )
#
#
# Display functions for the structs
show(io::IO, M::Power) = print(io,string("The Power Manifold of ",repr(M.manifold), " of size ",repr(M.powerSize),".") );
show(io::IO, p::PowPoint) = print(io,string("Pow[",join(repr.(p.value),", "),"]"));
show(io::IO, ξ::PowTVector) = print(io,string("PowT[", join(repr.(ξ.value),", "),"]"));
