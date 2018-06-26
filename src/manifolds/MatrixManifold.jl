#
#      MatrixManifold -- a matrix manifold,
#       i.e. the values/points on the manifold are matrices
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base.LinAlg: transpose

export MatrixManifold, MMPoint, MMTVector
export transpose

abstract type MatrixManifold <: Manifold end
"""
	MMPoint - A point on a matrix manifold
	# Elements
	value  - (like Manifold) containg a representation of the manifold point
	decompostion – cache for a decomposition of value, e.g. into SVD
"""
abstract type MMPoint <: MPoint end

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
