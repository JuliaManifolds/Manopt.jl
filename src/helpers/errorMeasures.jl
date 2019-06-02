export meanSquaredError, meanAverageError
@doc doc"""
    meanSquaredError(M,x,y)
Computes the (mean) squared error between the two
[`MPoint`](@ref)s `x` and `y` on the ([`Power`](@ref)) manifold `M`.
"""
function meanSquaredError(M::mT,x::P,y::P) where {mT <: Manifold, P <: MPoint}
    return distance(M,x,y)^2
end
function meanSquaredError(M::Power,x::P,y::P) where {mT <: Manifold, P <: MPoint}
    return 1/prod(M.powerSize) * sum( distance.( Ref(M.manifold), getValue(x), getValue(y) ).^2 )
end
@doc doc"""
    meanSquaredError(M,x,y)
Computes the (mean) squared error between the two
[`MPoint`](@ref)s `x` and `y` on the ([`Power`](@ref)) manifold `M`.
"""
function meanAverageError(M::mT,x::P,y::P) where {mT <: Manifold, P <: MPoint}
    return distance(M,x,y)
end
function meanAverageError(M::Power,x::P,y::P) where {mT <: Manifold, P <: MPoint}
    return 1/prod(M.powerSize) * sum( distance.( Ref(M.manifold), getValue(x), getValue(y) ) )
end
