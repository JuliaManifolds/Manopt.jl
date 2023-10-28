Base.@deprecate_binding HeestenesStiefelCoefficient HestenesStiefelCoefficient
export HeestenesStiefelCoefficient
Base.@deprecate_binding SimpleCacheObjective SimpleManifoldCachedObjective
export SimpleCacheObjective
#
# Deprecated - even keeping old notation where the Hessian was last -> move upfront.
#
@deprecate truncated_conjugate_gradient_descent(
    M::AbstractManifold, F, gradF, x, Y, H::TH; kwargs...
) where {TH<:Function} truncated_conjugate_gradient_descent(M, F, gradF, H, x, Y; kwargs...)
