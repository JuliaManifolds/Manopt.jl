
@doc raw"""
    X = get_gradient(M::ProductManifold, ago::ManifoldAlternatingGradientObjective, p)
    get_gradient!(M::ProductManifold, P::ManifoldAlternatingGradientObjective, X, p)

Evaluate all summands gradients at a point `p` on the `ProductManifold M` (in place of `X`)
"""
get_gradient(M::ProductManifold, ::ManifoldAlternatingGradientObjective, ::Any...)

function get_gradient(
    M::AbstractManifold,
    mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation,TC,<:AbstractVector},
    p,
) where {TC}
    return ProductRepr([gi(M, p) for gi in mago.gradient!!]...)
end

@doc raw"""
    X = get_gradient(M::AbstractManifold, p::ManifoldAlternatingGradientObjective, p, k)
    get_gradient!(M::AbstractManifold, p::ManifoldAlternatingGradientObjective, X, p, k)

Evaluate one of the component gradients ``\operatorname{grad}f_k``, ``k∈\{1,…,n\}``, at `x` (in place of `Y`).
"""
function get_gradient(
    M::ProductManifold,
    mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation,TC,<:Function},
    p,
    k,
) where {TC}
    return get_gradient(M, mago, p)[M, k]
end
function get_gradient!(
    M::AbstractManifold,
    X,
    mago::ManifoldAlternatingGradientObjective{InplaceEvaluation,TC,<:AbstractVector},
    p,
) where {TC}
    for (gi, Xi) in zip(mago.gradient!!, submanifold_components(M, X))
        gi(M, Xi, p)
    end
    return X
end

function get_gradient!(
    M::ProductManifold,
    X,
    mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation,TC,<:Function},
    p,
    k,
) where {TC}
    copyto!(M[k], X, mago.gradient!!(M, p)[M, k])
    return X
end

function alternating_gradient_descent(
    M::ProductManifold,
    f,
    grad_f::Union{TgF,AbstractVector{<:TgF}},
    p=rand(M);
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TgF}
    ago = ManifoldAlternatingGradientObjective(f, grad_f; evaluation=evaluation)
    return alternating_gradient_descent(M, ago, p; evaluation=evaluation, kwargs...)
end
function alternating_gradient_descent(
    M::ProductManifold, ago::ManifoldAlternatingGradientObjective, p; kwargs...
)
    q = copy(M, p)
    return alternating_gradient_descent!(M, ago, q; kwargs...)
end

function alternating_gradient_descent!(
    M::ProductManifold,
    f,
    grad_f::Union{TgF,AbstractVector{<:TgF}},
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TgF}
    agmo = ManifoldAlternatingGradientObjective(f, grad_f; evaluation=evaluation)
    return alternating_gradient_descent!(M, agmo, p; evaluation=evaluation, kwargs...)
end
function alternating_gradient_descent!(
    M::ProductManifold,
    agmo::ManifoldAlternatingGradientObjective,
    p;
    inner_iterations::Int=5,
    stopping_criterion::StoppingCriterion=StopAfterIteration(100) |
                                          StopWhenGradientNormLess(1e-9),
    stepsize::Stepsize=default_stepsize(M, AlternatingGradientDescentState),
    order_type::Symbol=:Linear,
    order=collect(1:length(M.manifolds)),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
)
    dagmo = decorate_objective!(M, agmo; kwargs...)
    dmp = DefaultManoptProblem(M, dagmo)
    agds = AlternatingGradientDescentState(
        M,
        p;
        inner_iterations=inner_iterations,
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        order_type=order_type,
        order=order,
        retraction_method=retraction_method,
    )
    agds = decorate_state!(agds; kwargs...)
    return get_solver_return(solve!(dmp, agds))
end
