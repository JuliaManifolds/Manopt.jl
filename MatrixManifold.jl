#
#      MatrixManifold -- a matrix manifold,
#       i.e. the values/points on the manifold are matrices
#
import Base.LinAlg: transpose

export MatrixManifold MMPoint, MMTVector
export transpose

abstract MatrixManifold <: Manifold
abstract MMPoint <: MPoint
abstract MMTVector <: MTVector

+{T <: MMPoint}(x::T,y::T)::MMPoint = T(x.value + y.value)
-{T <: MMPoint}(x::T,y::T)::MMPoint = T(x.value - y.value)
*{T <: MMPoint}(x::T,y::T)::Array = transpose(x.value)*y.value
*{T <: MMTVector}(ξ::T,ν::T)::Array = transpose(ξ.value)*ν.value
*{T <: MMTVector, S <: MMPoint}(x::T,y::S)::Array = transpose(x.value)*y.value
*{T <: MMTVector, S <: MMPoint}(x::S,y::T)::Array = transpose(x.value)*y.value
function transpose{T <: MMPoint}(x::T)::Array = transpose(x.value)
function transpose{T <: MMTVector}(ξ::T)::Array = transpose(x.value)
