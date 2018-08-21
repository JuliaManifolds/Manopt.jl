#
#      MatrixManifold -- a matrix manifold,
#       i.e. the values/points on the manifold are matrices
#		this is here implemented as traits, such that every manifold can
#		“gain” this feature by setting @traitimlp IsMatrixM{}
#		for the manifold, its point and the tangent vector
#
# Manopt.jl, R. Bergmann, 2018-06-26
import LinearAlgebra: transpose
import Base: +, -, *, /
export transpose, IsMatrixM, IsMatrixP, IsMatrixV
export +,-,*,/
"""
    IsMatrixM{X}
An abstract Manifold to represent a manifold whose points are matrices.
For these manifolds the usual operators (+,-,*) are overloaded for points.
Furthermore, the `transpose` is also overloaded, though it returns the matrix,
since the dimensions mit be different for rectangular matrices.
"""
@traitdef IsMatrixM{X}
"""
    IsMatrixP{X}
An abstract Manifold Point belonging to a matrix manifold.
"""
@traitdef IsMatrixP{X}
"""
    IsMatrixV{X}
An abstract Manifold Point belonging to a matrix manifold.
"""
@traitdef IsMatrixV{X}

# for all that satisfy IsMatrixM -> introduce operators on points and points/TVecs
@traitfn +(x::T,y::T) where {T <: MPoint; IsMatrixP{T}} = T( getValue(x) + getValue(y) )
@traitfn -(x::T,y::T) where {T <: MPoint; IsMatrixP{T}} = T( getValue(x) - getValue(y) )
@traitfn *(x::T,y::T) where {T <: MPoint; IsMatrixP{T}} = transpose( getValue(x) )* getValue(y)
@traitfn *(ξ::T,ν::T) where {T <: TVector; IsMatrixP{T}} = transpose( getValue(ξ) ) * getValue(ν)
@traitfn *(x::T,y::S) where {T <: TVector, S <: MPoint; IsMatrixV{T},IsMatrixP{S}} = transpose( getValue(x) ) * getValue(y)
@traitfn *(x::S,y::T) where {T <: TVector, S <: MPoint; IsMatrixV{T},IsMatrixP{S}} = transpose( getValue(x) ) * getValue(y)
@traitfn transpose(x::T) where {T <: MPoint; IsMatrixP{T}} = transpose( getValue(x) )
@traitfn transpose(ξ::T) where {T <: TVector; IsMatrixV{T}} = transpose( getValue(x) )
