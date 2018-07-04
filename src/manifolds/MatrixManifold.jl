#
#      MatrixManifold -- a matrix manifold,
#       i.e. the values/points on the manifold are matrices
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show
import Base.LinAlg: transpose

export MatrixManifold, MMPoint, MMTVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export show, transpose
"""
	An abstract Manifold to represent a manifold whose points are matrices.
	For these manifolds the usual operators (+,-,*) are overloaded for points.
"""
abstract type MatrixManifold <: Manifold end
"""
	MMPoint - A point on a matrix manifold
	# Elements
	value  - (like Manifold) containg a representation of the manifold point
	decompostion – cache for a decomposition of value, e.g. into SVD
"""
abstract type MMPoint <: MPoint end
"""
	MMTVector
"""
abstract type MMTVector <: MTVector end
#
#
# promote operations to the value field of MMPoint and MMTVector
+{T <: MMPoint}(x::T,y::T)::MMPoint = T(x.value + y.value)
-{T <: MMPoint}(x::T,y::T)::MMPoint = T(x.value - y.value)
*{T <: MMPoint}(x::T,y::T)::Array = transpose(x.value)*y.value
*{T <: MMTVector}(ξ::T,ν::T)::Array = transpose(ξ.value)*ν.value
*{T <: MMTVector, S <: MMPoint}(x::T,y::S)::Array = transpose(x.value)*y.value
*{T <: MMTVector, S <: MMPoint}(x::S,y::T)::Array = transpose(x.value)*y.value
function transpose{T <: MMPoint}(x::T)::Array
  return transpose(x.value)
end
function transpose{T <: MMTVector}(ξ::T)::Array
  return transpose(x.value)
end
