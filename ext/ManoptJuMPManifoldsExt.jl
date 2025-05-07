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

function JuMP.vectorize(
    p::P, mp::ManifoldDataShape{M,P}
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
    X::T, tv::ManifoldDataShape{M,T}
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
    v::Vector, p::ManifoldDataShape{M,P}
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
    v::Vector, X::TangentSManifoldDataShape{M,T}
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

# (b) Specific points – the first of which all “fall back” to just `.value`
const _VectorValueData = Union{
    Manifolds.PoincareBallPoint,
    PoincareBallTangentVector,
    PoincareHalfSpacePoint,
    PoincareHalfSpaceTangentVector,
    HyperboloidPoint,
    HyperboloidTangentVector,
}
# from type -> vector
function JuMP.vectorize(
    point_or_vector::T, ::ManifoldDataShape{M,T}
) where {M,T<:_VectorValueData}
    return point_or_vector.value
end
# from vector -> type
function JuMP.reshape_vector(
    vector::AbstractVector, ::ManifoldDataShape{M,T}
) where {M,T<:_VectorValueData}
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
