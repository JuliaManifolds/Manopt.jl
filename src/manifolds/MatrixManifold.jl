#
#      MatrixManifold -- a matrix manifold,
#       i.e. the values/points on the manifold are matrices
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base.LinAlg: transpose
import Base: +, -, *, /
export transpose, IsMatrixM, IsMatrixP, IsMatrixV
export +,-,*,/
"""
	An abstract Manifold to represent a manifold whose points are matrices.
	For these manifolds the usual operators (+,-,*) are overloaded for points.
"""
# Introduces three Traits for Manifolds, Points and Vectors
@traitdef IsMatrixM{X}
@traitdef IsMatrixP{X}
@traitdef IsMatrixV{X}

# for all that satisfy IsMatrixM -> introduce operators on points and points/TVecs
@traitfn +{T <: MPoint; IsMatrixP{T}}(x::T,y::T) = T(x.value + y.value)
@traitfn -{T <: MPoint; IsMatrixP{T}}(x::T,y::T) = T(x.value - y.value)
@traitfn *{T <: MPoint; IsMatrixP{T}}(x::T,y::T) = transpose(x.value)*y.value
@traitfn *{T <: TVector; IsMatrixP{T}}(ξ::T,ν::T) = transpose(ξ.value)*ν.value
@traitfn *{T <: TVector, S <: MPoint; IsMatrixV{T},IsMatrixP{S}}(x::T,y::S) = transpose(x.value)*y.value
@traitfn *{T <: TVector, S <: MPoint; IsMatrixV{T},IsMatrixP{S}}(x::S,y::T) = transpose(x.value)*y.value

@traitfn transpose{T <: MPoint; IsMatrixP{T}}(x::T) = transpose(x.value)
@traitfn transpose{T <: TVector; IsMatrixV{T}}(ξ::T) = transpose(x.value)
