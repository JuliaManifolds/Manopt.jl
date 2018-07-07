#
#      MatrixManifold -- a matrix manifold,
#       i.e. the values/points on the manifold are matrices
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base.LinAlg: transpose
export transpose, IsMatrixM
"""
	An abstract Manifold to represent a manifold whose points are matrices.
	For these manifolds the usual operators (+,-,*) are overloaded for points.
"""
# Introduces three Traits for Manifolds, Points and Vectors
@traitdef IsMatrixM{X}
@traitdef IsMatrixP{X}
@traitdef IsMatrixV{X}

# for all that satisfy IsMatrixM -> introduce operators on points and points/TVecs
+{T <: MPoint; IsMatrixP{T}}(x::T,y::T)::MPoint = T(x.value + y.value)
-{T <: MPoint; IsMatrixP{T}}(x::T,y::T)::MPoint = T(x.value - y.value)
*{T <: MPoint; IsMatrixP{T}}(x::T,y::T)::Array = transpose(x.value)*y.value
*{T <: TVector; IsMatrixP{T}}(ξ::T,ν::T)::Array = transpose(ξ.value)*ν.value
*{T <: TVector, S <: MPoint; IsMatrixT{T},IsMatrixP{S}}(x::T,y::S)::Array = transpose(x.value)*y.value
*{T <: TVector, S <: MPoint; IsMatrixT{T},IsMatrixP{S}}(x::S,y::T)::Array = transpose(x.value)*y.value
function transpose{T <: MPoint; IsMatrixV{T}}(x::T)::Array
  return transpose(x.value)
end
function transpose{T <: TVector; IsMatrixV{T}}(ξ::T)::Array
  return transpose(x.value)
end
