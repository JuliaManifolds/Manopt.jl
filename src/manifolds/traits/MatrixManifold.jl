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
export transpose, IsMatrixM, IsMatrixP, IsMatrixTV
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
    IsMatrixTV{X}
An abstract Manifold Point belonging to a matrix manifold.
"""
@traitdef IsMatrixTV{X}

# for all that satisfy IsMatrixM -> introduce operators on points and points/TVecs
@traitfn +(x::P, y::P) where {P <: MPoint; IsMatrixP{P}}  = P( getValue(x) + getValue(y) )
@traitfn -(x::P, y::P) where {P <: MPoint; IsMatrixP{P}}  = P( getValue(x) - getValue(y) )
@traitfn *(x::P, y::P) where {P <: MPoint; IsMatrixP{P}}  = P( getValue(x) * getValue(y) )
@traitfn transpose(x::P) where {P <: MPoint; IsMatrixP{P}} = P( Matrix(transpose( getValue(x) ) ) )
@traitfn *(ξ::T, ν::T) where {T <: TVector; IsMatrixTV{T}} = T( getValue(ξ) * getValue(ν) )
@traitfn transpose(ξ::T) where {T <: TVector; IsMatrixTV{T}} = T( Matrix( transpose( getValue(ξ) ) ) )
