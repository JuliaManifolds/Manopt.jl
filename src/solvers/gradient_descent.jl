
@doc raw"""
    GradientDescentState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `p`:                  (`rand(M)` the current iterate
* `X`:                  (`zero_vector(M,p)`) the current gradient ``\operatorname{grad}f(p)``, initialised to zero vector.
* `stopping_criterion`: ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `stepsize`:           ([`default_stepsize`](@ref)`(M, GradientDescentState)`) a [`Stepsize`](@ref)
* `direction`:          ([`IdentityUpdateRule`](@ref)) a processor to compute the gradient
* `retraction_method`:  (`default_retraction_method(M, typeof(p))`) the retraction to use, defaults to
  the default set for your manifold.

# Constructor

    GradientDescentState(M, p=rand(M); X=zero_vector(M, p), kwargs...)

Generate gradient descent options, where `X` can be used to set the tangent vector to store
the gradient in a certain type. All other fields are keyword arguments.

# See also

[`gradient_descent`](@ref)
"""
mutable struct GradientDescentState{
    P,
    T,
    TStop<:StoppingCriterion,
    TStepsize<:Stepsize,
    TDirection<:DirectionUpdateRule,
    TRTM<:AbstractRetractionMethod,
} <: AbstractGradientSolverState
    p::P
    X::T
    direction::TDirection
    stepsize::TStepsize
    stop::TStop
    retraction_method::TRTM
    function GradientDescentState{P,T}(
        M::AbstractManifold,
        p::P,
        X::T,
        stop::StoppingCriterion=StopAfterIteration(100),
        step::Stepsize=default_stepsize(M, GradientDescentState),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
        direction::DirectionUpdateRule=IdentityUpdateRule(),
    ) where {P,T}
        s = new{P,T,typeof(stop),typeof(step),typeof(direction),typeof(retraction_method)}()
        s.direction = direction
        s.p = p
        s.retraction_method = retraction_method
        s.stepsize = step
        s.stop = stop
        s.X = X
        return s
    end
end

function GradientDescentState(
    M::AbstractManifold,
    p::P=rand(M);
    X::T=zero_vector(M, p),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-8),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stepsize::Stepsize=default_stepsize(
        M, GradientDescentState; retraction_method=retraction_method
    ),
    direction::DirectionUpdateRule=IdentityUpdateRule(),
) where {P,T}
    return GradientDescentState{P,T}(
        M, p, X, stopping_criterion, stepsize, retraction_method, direction
    )
end
function (r::IdentityUpdateRule)(mp::AbstractManoptProblem, s::GradientDescentState, i)
    return get_stepsize(mp, s, i), get_gradient!(mp, s.X, s.p)
end
function default_stepsize(
    M::AbstractManifold,
    ::Type{GradientDescentState};
    retraction_method=default_retraction_method(M),
)
    # take a default with a slightly defensive initial step size.
    return ArmijoLinesearch(M; retraction_method=retraction_method, initial_stepsize=1.0)
end
function get_message(gds::GradientDescentState)
    # for now only step size is quipped with messages
    return get_message(gds.stepsize)
end
function show(io::IO, gds::GradientDescentState)
    i = get_count(gds, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(gds.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Gradient Descent
    $Iter
    ## Parameters
    * retraction method: $(gds.retraction_method)

    ## Stepsize
    $(gds.stepsize)

    ## Stopping criterion

    $(status_summary(gds.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

doc_gd_iterate = raw"""
```math
p_{k+1} = \operatorname{retr}_{p_k}\bigl( s_k\operatorname{grad}f(p_k) \bigr),
\qquad k=0,1,…
```
"""
doc_gradient_descent = """
    gradient_descent(M, f, grad_f, p=rand(M); kwargs...)
    gradient_descent(M, gradient_objective, p=rand(M); kwargs...)
    gradient_descent!(M, f, grad_f, p; kwargs...)
    gradient_descent!(M, gradient_objective, p; kwargs...)

perform a gradient descent

$(doc_gd_iterate)

with different choices of the stepsize ``s_k`` available (see `stepsize` option below).

# Input

$_arg_M
$_arg_f
$_arg_grad_f
$_arg_p

$_arg_alt_mgo

# Keyword arguments

* `direction` specifies to perform a certain processing of the direction,
  for example [`Nesterov`](@ref), [`MomentumGradient`](@ref) or [`AverageGradient`](@ref)

  🏔️ [`IdentityUpdateRule`](@ref), which yields a classical gradient descent.

$_kw_evaluation
  🏔️ [`AllocatingEvaluation`](@ref)

$_kw_retraction_method
  🏔️ [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`

$_kw_stepsize
  🏔️ [`default_stepsize`](@ref)`(M, GradientDescentState)`)

$_kw_stopping_criterion
  🏔️ [`StopAfterIteration`](@ref)`(200) | `[`StopWhenGradientNormLess`](@ref)`(1e-8)`

* `X` specify a memory internally to store the gradient

  🏔️ [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.

If you provide the [`ManifoldGradientObjective`](@ref) directly, `evaluation` is ignored.

$_kw_others

If you provide the [`ManifoldGradientObjective`](@ref) directly, these decorations can still be specified

# Output

the obtained (approximate) minimizer ``p^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details
"""

"$(doc_gradient_descent)"
gradient_descent(M::AbstractManifold, args...; kwargs...)
function gradient_descent(M::AbstractManifold, f, grad_f; kwargs...)
    return gradient_descent(M, f, grad_f, rand(M); kwargs...)
end
function gradient_descent(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    return gradient_descent(M, mgo, p; kwargs...)
end
function gradient_descent(
    M::AbstractManifold,
    f,
    grad_f,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    # redefine initial point
    q = [p]
    f_(M, p) = f(M, p[])
    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    rs = gradient_descent(M, f_, grad_f_, q; evaluation=evaluation, kwargs...)
    #return just a number if the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function gradient_descent(
    M::AbstractManifold, mgo::O, p; kwargs...
) where {O<:Union{AbstractManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return gradient_descent!(M, mgo, q; kwargs...)
end

"$(doc_gradient_descent)"
gradient_descent!(M::AbstractManifold, args...; kwargs...)
function gradient_descent!(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    return gradient_descent!(M, mgo, p; kwargs...)
end
function gradient_descent!(
    M::AbstractManifold,
    mgo::O,
    p;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stepsize::Stepsize=default_stepsize(
        M, GradientDescentState; retraction_method=retraction_method
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-8),
    debug=if is_tutorial_mode()
        if (stepsize isa ConstantStepsize)
            [DebugWarnIfCostIncreases(), DebugWarnIfGradientNormTooLarge()]
        else
            [DebugWarnIfGradientNormTooLarge()]
        end
    else
        []
    end,
    direction=IdentityUpdateRule(),
    X=zero_vector(M, p),
    kwargs..., #collect rest
) where {O<:Union{AbstractManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    s = GradientDescentState(
        M,
        p;
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        direction=direction,
        retraction_method=retraction_method,
        X=X,
    )
    ds = decorate_state!(s; debug=debug, kwargs...)
    solve!(dmp, ds)
    return get_solver_return(get_objective(dmp), ds)
end
#
# Solver functions
#
function initialize_solver!(mp::AbstractManoptProblem, s::GradientDescentState)
    get_gradient!(mp, s.X, s.p)
    return s
end
function step_solver!(p::AbstractManoptProblem, s::GradientDescentState, i)
    step, s.X = s.direction(p, s, i)
    retract!(get_manifold(p), s.p, s.p, s.X, -step, s.retraction_method)
    return s
end
