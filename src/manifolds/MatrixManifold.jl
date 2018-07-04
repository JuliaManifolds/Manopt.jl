#
#      MatrixManifold -- a matrix manifold,
#       i.e. the values/points on the manifold are matrices
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show
import Base.LinAlg: transpose

export MatrixManifold, MatPoint, MatTVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export show, transpose
"""
	An abstract Manifold to represent a manifold whose points are matrices.
	For these manifolds the usual operators (+,-,*) are overloaded for points.
"""
abstract type MatrixManifold <: Manifold end
"""
	MatPoint - A point on a matrix manifold
	# Elements
	value  - (like Manifold) containg a representation of the manifold point
	decompostion – cache for a decomposition of value, e.g. into SVD
"""
abstract type MatPoint <: MPoint end
"""
	MatTVector
"""
abstract type MatTVector <: TVector end
#
#
# promote operations to the value field of MatPoint and MatTVector
+{T <: MatPoint}(x::T,y::T)::MatPoint = T(x.value + y.value)
-{T <: MatPoint}(x::T,y::T)::MatPoint = T(x.value - y.value)
*{T <: MatPoint}(x::T,y::T)::Array = transpose(x.value)*y.value
*{T <: MatTVector}(ξ::T,ν::T)::Array = transpose(ξ.value)*ν.value
*{T <: MatTVector, S <: MatPoint}(x::T,y::S)::Array = transpose(x.value)*y.value
*{T <: MatTVector, S <: MatPoint}(x::S,y::T)::Array = transpose(x.value)*y.value
function transpose{T <: MatPoint}(x::T)::Array
  return transpose(x.value)
end
function transpose{T <: MatTVector}(ξ::T)::Array
  return transpose(x.value)
end
