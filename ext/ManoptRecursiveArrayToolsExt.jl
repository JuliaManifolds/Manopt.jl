module ManoptRecursiveArrayToolsExt
using Manopt
using ManifoldsBase
using ManifoldsBase: submanifold_components
import Manopt:
    max_stepsize,
    alternating_gradient_descent,
    alternating_gradient_descent!,
    get_gradient,
    get_gradient!,
    set_parameter!
using Manopt: _tex, ManifoldDefaultsFactory, _produce_type

using RecursiveArrayTools

@doc """
    X = get_gradient(M::ProductManifold, ago::ManifoldAlternatingGradientObjective, p)
    get_gradient!(M::ProductManifold, X, ago::ManifoldAlternatingGradientObjective, p)

Evaluate all summands gradients at a point `p` on the `ProductManifold M` (in place of `X`)
"""
get_gradient(M::ProductManifold, ::ManifoldAlternatingGradientObjective, ::Any...)

@doc """
    X = get_gradient(M::AbstractManifold, mago::ManifoldAlternatingGradientObjective, p, i)
    get_gradient!(M::AbstractManifold, X, mago::ManifoldAlternatingGradientObjective, p, i)

Evaluate one of the component gradients ``$(_tex(:grad)) f_i``, ``i∈ $(_tex(:set, "1,…,n"))``, at `p` (in place of `X`).
"""
get_gradient!(M::AbstractManifold, X, mago::ManifoldAlternatingGradientObjective, p)

function get_gradient!(
        M::AbstractManifold, X, mago::ManifoldAlternatingGradientObjective{F, <:AbstractVector}, p,
    ) where {F}
    for (gi, Xi) in zip(mago.gradient!, submanifold_components(M, X))
        gi(M, Xi, p)
    end
    return X
end

function alternating_gradient_descent(
        M::ProductManifold, f, grad_f::Union{TgF, AbstractVector{<:TgF}}, p = rand(M);
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    ) where {TgF}
    ago = ManifoldAlternatingGradientObjective(f, grad_f; evaluation = evaluation)
    return alternating_gradient_descent(M, ago, p; evaluation = evaluation, kwargs...)
end
function alternating_gradient_descent(
        M::ProductManifold, ago::ManifoldAlternatingGradientObjective, p; kwargs...
    )
    Manopt.keywords_accepted(alternating_gradient_descent; kwargs...)
    q = copy(M, p)
    return alternating_gradient_descent!(M, ago, q; kwargs...)
end

function alternating_gradient_descent!(
        M::ProductManifold, f, grad_f::Union{TgF, AbstractVector{<:TgF}}, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    ) where {TgF}
    agmo = ManifoldAlternatingGradientObjective(f, grad_f; evaluation = evaluation)
    return alternating_gradient_descent!(M, agmo, p; evaluation = evaluation, kwargs...)
end
function alternating_gradient_descent!(
        M::ProductManifold, agmo::ManifoldAlternatingGradientObjective, p;
        callbacks = Dict{Symbol, Function}(),
        inner_iterations::Int = 5,
        stopping_criterion::StoppingCriterion = StopAfterIteration(100) |
            StopWhenGradientNormLess(1.0e-9),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, AlternatingGradientDescentState; retraction_method = retraction_method
        ),
        order_type::Symbol = :Linear,
        order = collect(1:length(M.manifolds)),
        kwargs...,
    )
    Manopt.keywords_accepted(alternating_gradient_descent!; kwargs...)
    dagmo = Manopt.decorate_objective!(M, agmo; kwargs...)
    dmp = DefaultManoptProblem(M, dagmo)
    agds = AlternatingGradientDescentState(
        M;
        p = p,
        callbacks = Manopt.process_callbacks_arg(callbacks, AlternatingGradientDescentState),
        inner_iterations = inner_iterations,
        stopping_criterion = stopping_criterion,
        stepsize = _produce_type(stepsize, M),
        order_type = order_type,
        order = order,
        retraction_method = retraction_method,
    )
    agds = Manopt.decorate_state!(agds; kwargs...)
    Manopt.solve!(dmp, agds)
    return Manopt.get_solver_return(get_objective(dmp), agds)
end
end
