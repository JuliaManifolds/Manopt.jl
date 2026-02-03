@doc """
    GradientDescentState{P,T} <: AbstractGradientSolverState

Describes the state of a gradient based descent algorithm.

# Fields

$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:X; add_properties = [:as_Gradient]))
$(_fields(:stopping_criterion; name = "stop"))
$(_fields(:stepsize))
* `direction::`[`DirectionUpdateRule`](@ref) : a processor to handle the obtained gradient and compute a
  direction to “walk into”.
$(_fields(:retraction_method))

# Constructor

    GradientDescentState(M::AbstractManifold; kwargs...)

Initialize the gradient descent solver state, where

## Input

$(_args(:M))

## Keyword arguments

* `direction=`[`IdentityUpdateRule`](@ref)`()`
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(100)"))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`GradientDescentState`](@ref)`; retraction_method=retraction_method)"))
$(_kwargs(:retraction_method))
$(_kwargs(:X; add_properties = [:as_Memory]))

# See also

[`gradient_descent`](@ref)
"""
mutable struct GradientDescentState{
        P,
        T,
        TStop <: StoppingCriterion,
        TStepsize <: Stepsize,
        TDirection <: DirectionUpdateRule,
        TRTM <: AbstractRetractionMethod,
    } <: AbstractGradientSolverState
    p::P
    X::T
    direction::TDirection
    stepsize::TStepsize
    stop::TStop
    retraction_method::TRTM
end
function GradientDescentState(
        M::AbstractManifold = ManifoldsBase.DefaultManifold();
        p::P = rand(M),
        X::T = zero_vector(M, p),
        stopping_criterion::SC = StopAfterIteration(200) | StopWhenGradientNormLess(1.0e-8),
        retraction_method::RTM = default_retraction_method(M, typeof(p)),
        stepsize::S = default_stepsize(
            M, GradientDescentState; retraction_method = retraction_method
        ),
        direction::D = IdentityUpdateRule(),
        kwargs..., # ignore rest
    ) where {
        P,
        T,
        SC <: StoppingCriterion,
        RTM <: AbstractRetractionMethod,
        S <: Stepsize,
        D <: DirectionUpdateRule,
    }
    return GradientDescentState{P, T, SC, S, D, RTM}(
        p, X, direction, stepsize, stopping_criterion, retraction_method
    )
end
function (r::IdentityUpdateRule)(
        mp::AbstractManoptProblem, s::AbstractGradientSolverState, k
    )
    get_gradient!(mp, s.X, s.p)
    return get_stepsize(mp, s, k; gradient = s.X), s.X
end

function default_stepsize(
        M::AbstractManifold,
        ::Type{GradientDescentState};
        retraction_method = default_retraction_method(M),
    )
    # take a default with a slightly defensive initial step size.
    return ArmijoLinesearchStepsize(
        M; retraction_method = retraction_method, initial_stepsize = 1.0
    )
end
function get_message(gds::GradientDescentState)
    # for now only step size is quipped with messages
    return get_message(gds.stepsize)
end

function Base.show(io::IO, gds::GradientDescentState)
    return print(
        io,
        "GradientDescentState(; direction=$(repr(gds.direction)), p=$(repr(gds.p)), stepsize=$(repr(gds.stepsize)), stopping_criterion=$(repr(gds.stop)), retraction_method=$(repr(gds.retraction_method)), X=$(repr(gds.X)))"
    )
end

function status_summary(gds::GradientDescentState; inline = false)
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
    $(status_summary(gds.stop; inline = true))

    This indicates convergence: $Conv"""
    return s
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

$(_args([:M, :f, :grad_f, :p]))

$(_note(:GradientObjective))

# Keyword arguments


$(_kwargs(:differential))
* `direction=`[`IdentityUpdateRule`](@ref)`()`:
  specify to perform a certain processing of the direction, for example
  [`Nesterov`](@ref), [`MomentumGradient`](@ref) or [`AverageGradient`](@ref).
$(_kwargs(:evaluation; add_properties = [:GradientExample]))
$(_kwargs(:retraction_method))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`GradientDescentState`](@ref)`; retraction_method=retraction_method)"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-8)"))
$(_kwargs(:X; add_properties = [:as_Gradient]))

$(_note(:OtherKeywords))

If you provide the [`ManifoldFirstOrderObjective`](@ref) directly, the `evaluation=` keyword is ignored.
The decorations are still applied to the objective.

$(_note(:TutorialMode))

$(_note(:OutputSection))
"""

@doc "$(_doc_gradient_descent)"
gradient_descent(M::AbstractManifold, args...; kwargs...)

function gradient_descent(
        M::AbstractManifold,
        f,
        grad_f,
        p = rand(M);
        differential = nothing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    mgo = ManifoldGradientObjective(
        f_, grad_f_; evaluation = evaluation, differential = differential
    )
    rs = gradient_descent(M, mgo, p_; kwargs...)
    return _ensure_matching_output(p, rs)
end
function gradient_descent(
        M::AbstractManifold, mgo::O, p = rand(M); kwargs...
    ) where {O <: Union{AbstractManifoldFirstOrderObjective, AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return gradient_descent!(M, mgo, q; kwargs...)
end
calls_with_kwargs(::typeof(gradient_descent)) = (gradient_descent!,)

"$(_doc_gradient_descent)"
gradient_descent!(M::AbstractManifold, args...; kwargs...)

function gradient_descent!(
        M::AbstractManifold, f, grad_f, p;
        differential = nothing, evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    keywords_accepted(gradient_descent; kwargs...)
    mgo = ManifoldGradientObjective(
        f, grad_f; differential = differential, evaluation = evaluation
    )
    return gradient_descent!(M, mgo, p; kwargs...)
end
function gradient_descent!(
        M::AbstractManifold,
        mgo::O,
        p;
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, GradientDescentState; retraction_method = retraction_method
        ),
        stopping_criterion::StoppingCriterion = StopAfterIteration(200) |
            StopWhenGradientNormLess(1.0e-8),
        debug = if is_tutorial_mode()
            if (stepsize isa ManifoldDefaultsFactory{Manopt.ConstantStepsize})
                # If you pass the step size (internal) directly, this is considered expert mode
                [DebugWarnIfCostIncreases(), DebugWarnIfGradientNormTooLarge()]
            else
                [DebugWarnIfGradientNormTooLarge()]
            end
        else
            []
        end,
        direction = Gradient(),
        X = zero_vector(M, p),
        kwargs..., #collect rest
    ) where {O <: Union{AbstractManifoldFirstOrderObjective, AbstractDecoratedManifoldObjective}}
    # all explicit others others from above are anyways accepted here, so we only have to pass kwargs in
    keywords_accepted(gradient_descent!; kwargs...)
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    s = GradientDescentState(
        M;
        p = p,
        stopping_criterion = stopping_criterion,
        stepsize = _produce_type(stepsize, M),
        direction = _produce_type(direction, M),
        retraction_method = retraction_method,
        X = X,
    )
    ds = decorate_state!(s; debug = debug, kwargs...)
    solve!(dmp, ds)
    return get_solver_return(get_objective(dmp), ds)
end
calls_with_kwargs(::typeof(gradient_descent!)) = (decorate_objective!, decorate_state!)
#
# Solver functions
#
function initialize_solver!(mp::AbstractManoptProblem, s::GradientDescentState)
    get_gradient!(mp, s.X, s.p)
    return s
end
function step_solver!(p::AbstractManoptProblem, s::GradientDescentState, k)
    step, s.X = s.direction(p, s, k)
    ManifoldsBase.retract_fused!(get_manifold(p), s.p, s.p, s.X, -step, s.retraction_method)
    return s
end
