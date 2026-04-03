"""
    StochasticGradientDescentState <: AbstractGradientDescentSolverState

Store the following fields for a default stochastic gradient descent algorithm,
see also [`ManifoldStochasticGradientObjective`](@ref) and [`stochastic_gradient_descent`](@ref).

# Fields

$(_fields(:p; add_properties = [:as_Iterate]))
* `direction`:  a direction update to use
$(_fields(:stopping_criterion; name = "stop"))
$(_fields(:stepsize))
* `evaluation_order`: specify whether to use a randomly permuted sequence (`:FixedRandom`:),
  a per cycle permuted sequence (`:Linear`) or the default, a `:Random` sequence.
* `order`: stores the current permutation
$(_fields(:retraction_method))

# Constructor

    StochasticGradientDescentState(M::AbstractManifold; kwargs...)

Create a `StochasticGradientDescentState` with start point `p`.

# Keyword arguments

* `direction=`[`StochasticGradientRule`](@ref)`(M, `$(_link(:zero_vector))`)`
* `order_type=:RandomOrder``
* `order=Int[]`: specify how to store the order of indices for the next epoche
$(_kwargs(:retraction_method))
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)"))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`StochasticGradientDescentState`](@ref)`)"))
$(_kwargs(:X; add_properties = [:as_Memory]))

"""
mutable struct StochasticGradientDescentState{
        P, T, D <: DirectionUpdateRule, SC <: StoppingCriterion, S <: Stepsize, RM <: AbstractRetractionMethod, V <: Vector{<:Int},
    } <: AbstractGradientSolverState
    p::P
    X::T
    direction::D
    stop::SC
    stepsize::S
    order_type::Symbol
    order::V
    retraction_method::RM
    k::Int # current iterate
    function StochasticGradientDescentState(;
            direction::D, p::P, X::T, stopping_criterion::SC, stepsize::S,
            order_type::Symbol, order::V, retraction_method::RM, k = 0
        ) where {
            P, T, D <: DirectionUpdateRule, SC <: StoppingCriterion, S <: Stepsize, RM <: AbstractRetractionMethod, V <: Vector{<:Int},
        }
        return new{P, T, D, SC, S, RM, V}(
            p, X, direction, stopping_criterion, stepsize, order_type, order, retraction_method, k
        )
    end
end

function StochasticGradientDescentState(
        M::AbstractManifold;
        p::P = rand(M),
        X::T = zero_vector(M, p),
        direction::D = StochasticGradientRule(M; X = copy(M, p, X)),
        order_type::Symbol = :RandomOrder,
        order::Vector{<:Int} = Int[],
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        stopping_criterion::SC = StopAfterIteration(1000),
        stepsize::S = default_stepsize(M, StochasticGradientDescentState),
    ) where {
        P, T, D <: DirectionUpdateRule, RM <: AbstractRetractionMethod, SC <: StoppingCriterion, S <: Stepsize,
    }
    return StochasticGradientDescentState(;
        p = p, X = X, direction = direction, stopping_criterion = stopping_criterion,
        stepsize = stepsize, order_type = order_type, order = order, retraction_method = retraction_method, k = 0,
    )
end
function Base.show(io::IO, sgds::StochasticGradientDescentState)
    print(io, "StochasticGradientDescentState(; ")
    print(io, "direction = "); print(io, sgds.direction); print(io, ", ")
    print(io, "order = "); print(io, sgds.order); print(io, ", ")
    print(io, "order_type = :$(sgds.order_type), ")
    print(io, "p = $(sgds.p), ")
    print(io, "retraction_method = "); print(io, sgds.retraction_method); print(io, ", ")
    print(io, "stepsize = "); print(io, sgds.stepsize); print(io, ", ")
    print(io, "stopping_crierion = "); print(io, status_summary(sgds.stop; context = :short)); print(io, ", ")
    print(io, "X = "); print(io, sgds.X)
    return print(io, ")")
end
function status_summary(sgds::StochasticGradientDescentState; context::Symbol = :default)
    (context === :short) && return repr(sgds)
    i = get_count(sgds, :Iterations)
    conv_inl = (i > 0) ? (indicates_convergence(sgds.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the stochastic gradient descent algorithm$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(sgds.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Stochastic Gradient Descent
    $Iter
    ## Parameters
    * direction: $(status_summary(sgds.direction; context = :inline))
    * order: $(sgds.order_type)
    * retraction method: $(sgds.retraction_method)

    ## Stepsize
    $(sgds.stepsize)

    ## Stopping criterion
    $(status_summary(sgds.stop))
    This indicates convergence: $Conv"""
    return s
end
"""
    StochasticGradientRule<: AbstractGradientGroupDirectionRule

Create a functor `(problem, state k) -> (s,X)` to evaluate the stochatsic gradient,
that is chose a random index from the `state` and use the internal field for
evaluation of the gradient in-place.

The default gradient processor, which just evaluates the (stochastic) gradient or a subset thereof.

# Fields

$(_fields(:X))

# Constructor

    StochasticGradientRule(M::AbstractManifold; p=rand(M), X=zero_vector(M, p))

Initialize the stochastic gradient processor with tangent vector type of `X`,
where both `M` and `p` are just help variables.

# See also
[`stochastic_gradient_descent`](@ref), [`StochasticGradient`])@ref)
"""
struct StochasticGradientRule{T} <: AbstractGradientGroupDirectionRule
    X::T
end
function StochasticGradientRule(
        M::AbstractManifold; p = rand(M), X::T = zero_vector(M, p)
    ) where {T}
    return StochasticGradientRule{T}(X)
end
function (sg::StochasticGradientRule)(
        apm::AbstractManoptProblem, sgds::StochasticGradientDescentState, k
    )
    # for each new epoch choose new order if at random order
    ((sgds.k == 1) && (sgds.order_type == :Random)) && shuffle!(sgds.order)
    # the gradient to choose, either from the order or completely random
    j = sgds.order_type == :Random ? rand(1:length(sgds.order)) : sgds.order[sgds.k]
    return sgds.stepsize(apm, sgds, k), get_gradient!(apm, sg.X, sgds.p, j)
end
function Base.show(io::IO, sg::StochasticGradientRule)
    return print(io, "StochasticGradientRule($(sg.X)")
end
function status_summary(sg::StochasticGradientRule; context::Symbol = :default)
    (context === :short) && return repr(sg)
    return "A stochastic gradient processor"
end
@doc """
    StochasticGradient(; kwargs...)
    StochasticGradient(M::AbstractManifold; kwargs...)

# Keyword arguments

$(_kwargs(:X; name = "initial_gradient"))
$(_kwargs(:p; add_properties = [:as_Initial]))

$(_note(:ManifoldDefaultFactory, "StochasticGradientRule"))
"""
function StochasticGradient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.StochasticGradientRule, args...; kwargs...)
end

"""
    default_stepsize(M::AbstractManifold, ::Type{StochasticGradientDescentState})

Deinfe the default step size computed for the [`StochasticGradientDescentState`](@ref),
which is [`ConstantStepsize`](@ref)`M`.
"""
function default_stepsize(M::AbstractManifold, ::Type{StochasticGradientDescentState})
    return ConstantStepsize(M)
end

_doc_SGD = """
    stochastic_gradient_descent(M, grad_f, p=rand(M); kwargs...)
    stochastic_gradient_descent(M, msgo; kwargs...)
    stochastic_gradient_descent!(M, grad_f, p; kwargs...)
    stochastic_gradient_descent!(M, msgo, p; kwargs...)

perform a stochastic gradient descent. This can be performed in-place of `p`.

# Input

$(_args(:M))
* `grad_f`: a gradient function, that either returns a vector of the gradients
  or is a vector of gradient functions
$(_args(:p))

alternatively to the gradient you can provide an [`ManifoldStochasticGradientObjective`](@ref) `msgo`,
then using the `cost=` keyword does not have any effect since if so, the cost is already within the objective.

# Keyword arguments

* `cost=missing`: you can provide a cost function for example to track the function value
* `direction=`[`StochasticGradient`](@ref)`(`$(_link(:zero_vector))`)` add a post-processor to
  the direction obtained from evaluating the sub-gradient.
$(_kwargs(:evaluation))
* `evaluation_order=:Random`: specify whether to use a randomly permuted sequence (`:FixedRandom`:,
  a per cycle permuted sequence (`:Linear`) or the default `:Random` one.
* `order_type=:RandomOrder`: a type of ordering of gradient evaluations.
  Possible values are `:RandomOrder`, a `:FixedPermutation`, `:LinearOrder`
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)"))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`StochasticGradientDescentState`](@ref)`)"))
* `order=[1:n]`: the initial permutation, where `n` is the number of gradients in `gradF`.
$(_kwargs(:retraction_method))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_SGD)"
stochastic_gradient_descent(M::AbstractManifold, args...; kwargs...)
function stochastic_gradient_descent(M::AbstractManifold, grad_f; kwargs...)
    return stochastic_gradient_descent(M, grad_f, rand(M); kwargs...)
end
function stochastic_gradient_descent(
        M::AbstractManifold, grad_f, p;
        cost = Missing(), evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    )
    p_ = _ensure_mutating_variable(p)
    cost_ = _ensure_mutating_cost(cost, p)
    if p isa Number
        if grad_f isa Function
            n = grad_f(M, p) isa Number
            grad_f_ = (M, p) -> [[X] for X in (n ? [grad_f(M, p[])] : grad_f(M, p[]))]
        else
            grad_f_ = [_ensure_mutating_gradient(f, p, evaluation) for f in grad_f]
        end
    else
        grad_f_ = grad_f
    end
    msgo = ManifoldStochasticGradientObjective(grad_f_; cost = cost_, evaluation = evaluation)
    rs = stochastic_gradient_descent(M, msgo, p_; evaluation = evaluation, kwargs...)
    return _ensure_matching_output(p, rs)
end
function stochastic_gradient_descent(
        M::AbstractManifold, msgo::O, p; kwargs...
    ) where {O <: Union{ManifoldStochasticGradientObjective, AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    keywords_accepted(stochastic_gradient_descent; kwargs...)
    return stochastic_gradient_descent!(M, msgo, q; kwargs...)
end
calls_with_kwargs(::typeof(stochastic_gradient_descent)) = (stochastic_gradient_descent!,)

@doc "$(_doc_SGD)"
stochastic_gradient_descent!(::AbstractManifold, args...; kwargs...)
function stochastic_gradient_descent!(
        M::AbstractManifold, grad_f, p;
        cost = Missing(), evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    )
    msgo = ManifoldStochasticGradientObjective(grad_f; cost = cost, evaluation = evaluation)
    return stochastic_gradient_descent!(M, msgo, p; evaluation = evaluation, kwargs...)
end
function stochastic_gradient_descent!(
        M::AbstractManifold, msgo::O, p;
        direction::Union{<:DirectionUpdateRule, ManifoldDefaultsFactory} = StochasticGradient(;
            p = p
        ),
        stopping_criterion::StoppingCriterion = StopAfterIteration(10000) | StopWhenGradientNormLess(1.0e-9),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, StochasticGradientDescentState
        ),
        order = collect(1:length(get_gradients(M, msgo, p))),
        order_type::Symbol = :Random,
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        kwargs...,
    ) where {O <: Union{ManifoldStochasticGradientObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(stochastic_gradient_descent!; kwargs...)
    dmsgo = decorate_objective!(M, msgo; kwargs...)
    mp = DefaultManoptProblem(M, dmsgo)
    sgds = StochasticGradientDescentState(
        M; p = p, X = zero_vector(M, p),
        direction = _produce_type(direction, M, p), stepsize = _produce_type(stepsize, M, p),
        order_type = order_type, order = order,
        stopping_criterion = stopping_criterion, retraction_method = retraction_method,
    )
    dsgds = decorate_state!(sgds; kwargs...)
    solve!(mp, dsgds)
    return get_solver_return(get_objective(mp), dsgds)
end
calls_with_kwargs(::typeof(stochastic_gradient_descent!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(::AbstractManoptProblem, s::StochasticGradientDescentState)
    s.k = 1
    (s.order_type == :FixedRandom) && (shuffle!(s.order))
    return s
end
function step_solver!(mp::AbstractManoptProblem, s::StochasticGradientDescentState, iter)
    step, s.X = s.direction(mp, s, iter)
    retract!(get_manifold(mp), s.p, s.p, -step * s.X)
    s.k = ((s.k) % length(s.order)) + 1
    return s
end
