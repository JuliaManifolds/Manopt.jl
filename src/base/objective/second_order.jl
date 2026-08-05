"""
    AbstractManifoldHessianObjective{F, G, H} <: AbstractManifoldFirstOrderObjective{Tuple{F,G}}

An abstract type for all objectives that provide a (full) Hessian.
"""
abstract type AbstractManifoldHessianObjective{F, G, H} <: AbstractManifoldFirstOrderObjective{F, G} end

function get_hessian(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X)
    return get_hessian(M, get_objective(admo, false), p, X)
end
function get_hessian!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, p, X
    )
    return get_hessian!(M, Y, get_objective(admo, false), p, X)
end
@doc """
    get_hessian_function(amgo::ManifoldHessianObjective)

return the function to evaluate (just) the Hessian ``$(_tex(:Hess)) f(p)``,
which has the form `(M, Y, p, X) -> Y` working in-place of `Y`.
"""
function get_hessian_function(
        mho::AbstractManifoldHessianObjective, recursive::Bool = false;
        evaluation::AbstractEvaluationType = AllocatingEvaluation()
    )
    if evaluation isa AllocatingEvaluation
        return mho.hessian! isa InplaceManifoldFunction ? mho.hessian!.f : (M, p, X) -> mho.hessian!(M, zero_vector(M, p), p, X)
    else
        return mho.hessian!
    end
end

function get_hessian_function(
        admo::AbstractDecoratedManifoldObjective, recursive::Bool = false; evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    )
    return get_hessian_function(get_objective(admo, recursive), recursive; evaluation = evaluation)
end
