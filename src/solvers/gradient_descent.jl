
@doc """
    GradientDescentState{P,T} <: AbstractGradientSolverState

Describes the state of a gradient based descent algorithm.

# Fields

* $_field_iterate
* $_field_gradient
* $_field_stop
* $_field_step
* `direction::`[`DirectionUpdateRule`](@ref) : a processor to handle the obtained gradient and compute a
  direction to “walk into”.
* $_field_retr

# Constructor

    GradientDescentState(M; kwargs...)

Initialize the gradient descent solver state, where

## Input

$_arg_M

## Keyword arguments

* `direction=`[`IdentityUpdateRule`](@ref)`()`
* $(_kw_p_default): $(_kw_p)
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(100)` $_kw_stop_note
* `stepsize=`[`default_stepsize`](@ref)`(M, GradientDescentState; retraction_method=retraction_method)`
* $_kw_retraction_method_default
* $_kw_X_default

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
end
function GradientDescentState(
    M::AbstractManifold;
    p::P=rand(M),
    X::T=zero_vector(M, p),
    stopping_criterion::SC=StopAfterIteration(200) | StopWhenGradientNormLess(1e-8),
    retraction_method::RTM=default_retraction_method(M, typeof(p)),
    stepsize::S=default_stepsize(
        M, GradientDescentState; retraction_method=retraction_method
    ),
    direction::D=IdentityUpdateRule(),
) where {
    P,
    T,
    SC<:StoppingCriterion,
    RTM<:AbstractRetractionMethod,
    S<:Stepsize,
    D<:DirectionUpdateRule,
}
    return GradientDescentState{P,T,SC,S,D,RTM}(
        p, X, direction, stepsize, stopping_criterion, retraction_method
    )
end
function (r::IdentityUpdateRule)(mp::AbstractManoptProblem, s::GradientDescentState, k)
    return get_stepsize(mp, s, k), get_gradient!(mp, s.X, s.p)
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

_doc_gd_iterate = raw"""
```math
p_{k+1} = \operatorname{retr}_{p_k}\bigl( s_k\operatorname{grad}f(p_k) \bigr),
\qquad k=0,1,…
```
where ``s_k > 0`` denotes a step size.
"""
_doc_gradient_descent = """
    gradient_descent(M, f, grad_f, p=rand(M); kwargs...)
    gradient_descent(M, gradient_objective, p=rand(M); kwargs...)
    gradient_descent!(M, f, grad_f, p; kwargs...)
    gradient_descent!(M, gradient_objective, p; kwargs...)

perform the gradient descent algorithm

$(_doc_gd_iterate)
The algorithm can be performed in-place of `p`.

# Input

$_arg_M
$_arg_f
$_arg_grad_f
$_arg_p

$_arg_alt_mgo

# Keyword arguments

* `direction=`[`IdentityUpdateRule`](@ref)`()`:
  specify to perform a certain processing of the direction, for example
  [`Nesterov`](@ref), [`MomentumGradient`](@ref) or [`AverageGradient`](@ref).

* $_kw_evaluation_default:
  $_kw_evaluation $_kw_evaluation_example

* $_kw_retraction_method_default:
  $_kw_retraction_method

* `stepsize=`[`default_stepsize`](@ref)`(M, GradientDescentState)`:
  $_kw_stepsize

* `stopping_criterion=`[`StopAfterIteration`](@ref)`(200)`$_sc_any[`StopWhenGradientNormLess`](@ref)`(1e-8)`:
  $_kw_stopping_criterion

* $_kw_X_default:
  $_kw_X, the evaluated gradient ``$_l_grad f`` evaluated at ``p^{(k)}``.

$_kw_others

If you provide the [`ManifoldGradientObjective`](@ref) directly, the `evaluation=` keyword is ignored.
The decorations are still applied to the objective.

$_doc_remark_tutorial_debug

$_doc_sec_output
"""

@doc "$(_doc_gradient_descent)"
gradient_descent(M::AbstractManifold, args...; kwargs...)

function gradient_descent(
    M::AbstractManifold,
    f,
    grad_f,
    p=rand(M);
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    mgo = ManifoldGradientObjective(f_, grad_f_; evaluation=evaluation)
    rs = gradient_descent(M, mgo, p_; kwargs...)
    return _ensure_matching_output(p, rs)
end
function gradient_descent(
    M::AbstractManifold, mgo::O, p; kwargs...
) where {O<:Union{AbstractManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return gradient_descent!(M, mgo, q; kwargs...)
end

"$(_doc_gradient_descent)"
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
    direction=Gradient(),
    X=zero_vector(M, p),
    kwargs..., #collect rest
) where {O<:Union{AbstractManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    s = GradientDescentState(
        M;
        p=p,
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        direction=_produce_rule(M, direction),
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
function step_solver!(p::AbstractManoptProblem, s::GradientDescentState, k)
    step, s.X = s.direction(p, s, k)
    retract!(get_manifold(p), s.p, s.p, s.X, -step, s.retraction_method)
    return s
end
