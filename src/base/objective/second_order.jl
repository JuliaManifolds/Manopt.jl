"""
    AbstractManifoldHessianObjective{E<:AbstractEvaluationType,F, G, H} <: AbstractManifoldFirstOrderObjective{E,Tuple{F,G}}

An abstract type for all objectives that provide a (full) Hessian, where
`T` is a [`AbstractEvaluationType`](@ref) for the gradient and Hessian functions.
"""
abstract type AbstractManifoldHessianObjective{F, G, H} <:
AbstractManifoldFirstOrderObjective{Tuple{F, G}} end

function _ensure_mutating_hessian(hess_f, p, evaluation::AbstractEvaluationType)
    return hess_f
end
function _ensure_mutating_hessian(hess_f, q::Number, evaluation::AllocatingEvaluation)
    return isnothing(hess_f) ? hess_f : (M, p, X) -> [hess_f(M, p[], X[])]
end
function _ensure_mutating_hessian(hess_f, q::Number, evaluation::InplaceEvaluation)
    return isnothing(hess_f) ? hess_f : (M, Y, p, X) -> (Y .= [hess_f(M, p[], X[])])
end
