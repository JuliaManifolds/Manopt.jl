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
@traitfn +{T <: MPoint; IsMatrixP{T}}(x::T,y::T) = T( getValue(x) + getValue(y) )
@traitfn -{T <: MPoint; IsMatrixP{T}}(x::T,y::T) = T( getValue(x) - getValue(y) )
@traitfn *{T <: MPoint; IsMatrixP{T}}(x::T,y::T) = transpose( getValue(x) )* getValue(y)
@traitfn *{T <: TVector; IsMatrixP{T}}(ξ::T,ν::T) = transpose( getValue(ξ) ) * getValue(ν)
@traitfn *{T <: TVector, S <: MPoint; IsMatrixV{T},IsMatrixP{S}}(x::T,y::S) = transpose( getValue(x) ) * getValue(y)
@traitfn *{T <: TVector, S <: MPoint; IsMatrixV{T},IsMatrixP{S}}(x::S,y::T) = transpose( getValue(x) ) * getValue(y)
@traitfn transpose{T <: MPoint; IsMatrixP{T}}(x::T) = transpose( getValue(x) )
@traitfn transpose{T <: TVector; IsMatrixV{T}}(ξ::T) = transpose( getValue(x) )
