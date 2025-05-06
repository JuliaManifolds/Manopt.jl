module ManoptJuMPManifoldsExt

using Manopt
using Manifolds
using LinearAlgebra
using JuMP: JuMP
using ManifoldsBase
const MOI = JuMP.MOI

#
#
# Point and Tangent vector conversions
# Every representation has to implement
# * JuMP.vectorize(p::PointType, ::??)
# * JuMP.reshape_vector(vector::Vector, p::PointType)
# * (maybe available generically?) JuMP.reshape_set(manifold, ::PointType)

# Maybe we need a dummy type? Do we need one or two?
# Probably two We have to distinguish points and vectors somehow! Maybe something like this?
struct ManifoldPoint{P<:ManifoldsBase.AbstractManifoldPoint} <: JuMP.AbstractShape end
struct TangentSpaceVector{T<:ManifoldsBase.AbstractTangentVector} <: JuMP.AbstractShape end

# (a) generic definitions and errors
JuMP.reshape_set(M::AbstractManifold, ::ManifoldPoint) = M
JuMP.reshape_set(M::AbstractManifold, ::TangentSpaceVector) = M

function JuMP.vectorize(
    p::P, mp::ManifoldPoint
) where {P<:ManifoldsBase.AbstractManifoldPoint}
    throw(
        DomainError(
            p,
            """
            You are trying to call `JuMP.vectorize` $p of type $P,
            which seems not to be available.
            Since the provided shape $(typeof(mp)) is from the extension of
            `Manopt.jl` and `JuMP.jl` maybe consider opening an issue in `Manopt.jl`,
            where this extension is maintained.
            """,
        ),
    )
end
function JuMP.vectorize(
    X::T, tv::ManifoldPoint
) where {T<:ManifoldsBase.AbstractTangentVector}
    throw(
        DomainError(
            X,
            """
            You are trying to call `JuMP.vectorize` $X of type $T,
            which seems not to be available.
            Since the provided shape $(typeof(tv)) is from the extension of
            `Manopt.jl` and `JuMP.jl` maybe consider opening an issue in `Manopt.jl`,
            where this extension is maintained.
            """,
        ),
    )
end

function JuMP.reshape_vector(
    v::Vector, p::ManifoldPoint{P}
) where {P<:ManifoldsBase.AbstractManifoldPoint}
    throw(
        DomainError(
            p,
            """
            You are trying to call `reshape_vector` $v to a type $P,
            which seems not to be available.
            Since the provided shape $(typeof(p)) is from the extension of
            `Manopt.jl` and `JuMP.jl` maybe consider opening an issue in `Manopt.jl`,
            where this extension is maintained.
            """,
        ),
    )
end
function JuMP.reshape_vector(
    v::Vector, X::TangentSpaceVector{T}
) where {T<:ManifoldsBase.AbstractTangentVector}
    throw(
        DomainError(
            X,
            """
            You are trying to call `reshape_vector` $v to a type $T,
            which seems not to be available.
            Since the provided shape $(typeof(X)) is from the extension of
            `Manopt.jl` and `JuMP.jl` maybe consider opening an issue in `Manopt.jl`,
            where this extension is maintained.
            """,
        ),
    )
end

# ---
# A generic type for a few of the following types
#
@doc """
    ManifoldsBaseVectorShape{T<:Union{ManifoldsBase.AbstractManifoldPoint,ManifoldsBase.AbstractTangentVector}} <: JuMP.AbstractShape

An [`AbstractShape`](@extref `JuMP.AbstractShape`) to represent any type `T`that is either an
[`AbstractManifoldPoint`](@extref `ManifoldsBase.AbstractManifoldPoint`) or an [`ManifoldsBase.AbstractTangentVector`](@extref `ManifoldsBase.AbstractTangentVector`)
that is internally represented by a vector themselves.
"""
struct ManifoldsBaseVectorShape{
    T<:Union{ManifoldsBase.AbstractManifoldPoint,ManifoldsBase.AbstractTangentVector}
} <: JuMP.AbstractShape
    T::Type{T}
end

# (b) Specific points – the first of which all “fall back” to just `.value`
const _VectorValuePoints = Union{
    Manifolds.PoincareBallPoint,
    PoincareBallTangentVector,
    PoincareHalfSpacePoint,
    PoincareHalfSpaceTangentVector,
    HyperboloidPoint,
    HyperboloidTangentVector,
}
# from type -> vector
function JuMP.vectorize(
    point_or_vector::T, ::ManifoldsBaseVectorShape{T}
) where {T<:_VectorValuePoints}
    return point_or_vector.value
end
# from vector -> type
function JuMP.reshape_vector(
    vector::Vector, ::ManifoldsBaseVectorShape{T}
) where {T<:_VectorValuePoints}
    return T(vector)
end

# Todo – refactor _shape() but to what? What does _shape actually do? something like ManifoldsBase.allocate?

# SPD matrices (SPDPoint)

# Grassmann as Stiefel (StiefelPoint, StiefelTangentVector)

# Grassmann as Projection (ProjectorPoint, ProjectorTangentVector)

# Fixed Rank matrices (SVDMPoint, UMVTangentVector)

# Flag Manifold (OrthogonalPoint, OrthogonalTangentVector)

# Maybe complicated:
# ValidationManifold (ValidationMPoint)

# Maybe also (maybe also only if someone needs this)
# Tucker (TuckerPoint, TuckerTangentVector)

end # module
