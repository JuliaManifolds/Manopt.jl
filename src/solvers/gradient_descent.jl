
@doc raw"""
    GradientDescentState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `p – (`rand(M)` the current iterate
* `X` – (`zero_vector(M,p)`) the current gradient ``\operatorname{grad}f(p)``, initialised to zero vector.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `stepsize` – ([`default_stepsize`](@ref)`(M, GradientDescentState)`) a [`Stepsize`](@ref)
* `direction` - ([`IdentityUpdateRule`](@ref)) a processor to compute the gradient
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) the retraction to use, defaults to
  the default set for your manifold.

# Constructor

    GradientDescentState(M, p=rand(M); X=zero_vector(M, p), kwargs...)

Generate gradient descent options, where `X` can be used to set the tangent vector to store
the gradient in a certain type; it will be initialised accordingly at a later stage.
All following fields are keyword arguments.

# See also

[`gradient_descent`](@ref)
"""
mutable struct GradientDescentState{
    P,T,TStop<:StoppingCriterion,TStepsize<:Stepsize,TRTM<:AbstractRetractionMethod
} <: AbstractGradientSolverState
    p::P
    X::T
    direction::DirectionUpdateRule
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
        o = new{P,T,typeof(stop),typeof(step),typeof(retraction_method)}()
        o.direction = direction
        o.p = p
        o.retraction_method = retraction_method
        o.stepsize = step
        o.stop = stop
        o.X = X
        return o
    end
end
function GradientDescentState(
    M::AbstractManifold,
    p::P=rand(M);
    X::T=zero_vector(M, p),
    stopping_criterion::StoppingCriterion=StopAfterIteration(100),
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

@doc raw"""
    gradient_descent(M, f, grad_f, p; kwargs...)

perform a gradient descent

```math
p_{k+1} = \operatorname{retr}_{p_k}\bigl( s_k\operatorname{grad}f(p_k) \bigr),
\qquad k=0,1,…
```

with different choices of the stepsize ``s_k`` available (see `stepsize` option below).

# Input
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f: \mathcal M→ℝ`` to find a minimizer ``p^*`` for
* `grad_f` – the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of f
  - as a function `(M, p) -> X` or a function `(M, X, p) -> X`
* `p` – an initial value `p` ``= p_0 ∈ \mathcal M``

# Optional
* `direction` – [`IdentityUpdateRule`](@ref) perform a processing of the direction, e.g.
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `grad_f(M, p)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `grad_f!(M, X, p)`.
* `retraction_method` – (`default_retraction_method(M)`) a retraction to use
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
  functor.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

All other keyword arguments are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details
"""
function gradient_descent(
    M::AbstractManifold, F::TF, gradF::TDF, x; kwargs...
) where {TF,TDF}
    x_res = copy(M, x)
    return gradient_descent!(M, F, gradF, x_res; kwargs...)
end
@doc raw"""
    gradient_descent!(M, F, gradF, x)

perform a gradient_descent

```math
x_{k+1} = \operatorname{retr}_{x_k}\bigl( s_k\operatorname{grad}f(x_k) \bigr)
```

in place of `x` with different choices of ``s_k`` available.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF` – the gradient ``\operatorname{grad}F:\mathcal M→ T\mathcal M`` of F
* `x` – an initial value ``x ∈ \mathcal M``

For more options, especially [`Stepsize`](@ref)s for ``s_k``, see [`gradient_descent`](@ref)
"""
function gradient_descent!(
    M::AbstractManifold,
    F::TF,
    gradF::TDF,
    p;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stepsize::Stepsize=default_stepsize(
        M, GradientDescentState; retraction_method=retraction_method
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-9),
    debug=stepsize isa ConstantStepsize ? [DebugWarnIfCostIncreases()] : [],
    direction=IdentityUpdateRule(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs..., #collect rest
) where {TF,TDF}
    mgo = ManifoldGradientObjective(F, gradF; evaluation=evaluation)
    dmgo = decorate_objective!(M, mgo; kwargs...)
    mp = DefaultManoptProblem(M, dmgo)
    s = GradientDescentState(
        M,
        p;
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        direction=direction,
        retraction_method=retraction_method,
    )
    s = decorate_state!(s; debug=debug, kwargs...)
    return get_solver_return(solve!(mp, s))
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
