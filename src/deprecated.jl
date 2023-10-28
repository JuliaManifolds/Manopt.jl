Base.@deprecate_binding HeestenesStiefelCoefficient HestenesStiefelCoefficient
export HeestenesStiefelCoefficient
Base.@deprecate_binding SimpleCacheObjective SimpleManifoldCachedObjective
export SimpleCacheObjective
#
# Deprecated tCG calls
#
@deprecate truncated_conjugate_gradient_descent(
    M::AbstractManifold, F, gradF, x, Y, H::TH; kwargs...
) where {TH<:Function} truncated_conjugate_gradient_descent(M, F, gradF, H, x, Y; kwargs...)
@deprecate truncated_conjugate_gradient_descent!(
    M::AbstractManifold, F::TF, gradF::TG, x, Y, H::TH; kwargs...
) where {TF<:Function,TG<:Function,TH<:Function} truncated_conjugate_gradient_descent!(
    M, F, gradF, H, x, Y; kwargs...
)
