"""
    StochasticGradientDescentState <: AbstractGradientDescentSolverState

Store the following fields for a default stochastic gradient descent algorithm,
see also [`ManifoldStochasticGradientObjective`](@ref) and [`stochastic_gradient_descent`](@ref).

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:p; add_properties = [:as_Iterate]))
* `direction`:  a direction update to use
$(_fields(:stopping_criterion; name = "stop"))
$(_fields(:stepsize))
* `order_type`: specify whether to use a fixed randomly permuted sequence (`:FixedRandom`),
  the sequence as given in `order` (`:Linear`), or the default `:Random` one,
  which chooses a random gradient in every step.
* `order`: stores the current permutation
$(_fields(:retraction_method))

# Constructor

    StochasticGradientDescentState(M::AbstractManifold; kwargs...)

Create a `StochasticGradientDescentState` with start point `p`.

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
* `direction=`[`StochasticGradientRule`](@ref)`(M, `$(_link(:zero_vector))`)`
* `order_type=:Random`
* `order=Int[]`: specify how to store the order of indices for the next epoche
$(_kwargs(:retraction_method))
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)"))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`StochasticGradientDescentState`](@ref)`)"))
$(_kwargs(:X; add_properties = [:as_Memory]))
"""
mutable struct StochasticGradientDescentState{
        P, T, C <: AbstractDict{Symbol}, D <: DirectionUpdateRule, SC <: StoppingCriterion, S <: Stepsize, RM <: AbstractRetractionMethod, V <: Vector{<:Int},
    } <: AbstractGradientSolverState
    callbacks::C
    direction::D
    k::Int # current iterate
    order::V
    order_type::Symbol
    p::P
    retraction_method::RM
    stepsize::S
    stop::SC
    X::T
    function StochasticGradientDescentState(;
            callbacks::C = Dict{Symbol, Function}(), direction::D, p::P, X::T, stopping_criterion::SC, stepsize::S,
            order_type::Symbol, order::V, retraction_method::RM, k = 0
        ) where {
            P, T, C <: AbstractDict{Symbol}, D <: DirectionUpdateRule, SC <: StoppingCriterion, S <: Stepsize, RM <: AbstractRetractionMethod, V <: Vector{<:Int},
        }
        return new{P, T, C, D, SC, S, RM, V}(
            callbacks, direction, k, order, order_type, p, retraction_method, stepsize, stopping_criterion, X
        )
    end
end

function StochasticGradientDescentState(
        M::AbstractManifold;
        callbacks::C = Dict{Symbol, Function}(),
        p::P = rand(M),
        X::T = zero_vector(M, p),
        direction::D = StochasticGradientRule(M; X = copy(M, p, X)),
        order_type::Symbol = :Random,
        order::Vector{<:Int} = Int[],
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        stopping_criterion::SC = StopAfterIteration(1000),
        stepsize::S = default_stepsize(M, StochasticGradientDescentState),
    ) where {
        P, T, C <: AbstractDict{Symbol}, D <: DirectionUpdateRule, RM <: AbstractRetractionMethod, SC <: StoppingCriterion, S <: Stepsize,
    }
    (order_type in (:Random, :FixedRandom, :Linear)) || throw(
        DomainError(order_type, "The order type has to be one of :Random, :FixedRandom, or :Linear.")
    )
    return StochasticGradientDescentState(;
        callbacks = callbacks, p = p, X = X, direction = direction, stopping_criterion = stopping_criterion,
        stepsize = stepsize, order_type = order_type, order = order, retraction_method = retraction_method, k = 0,
    )
end
get_callbacks(sgds::StochasticGradientDescentState) = sgds.callbacks
provided_callbacks(::Type{StochasticGradientDescentState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:Direction])
function Base.show(io::IO, sgds::StochasticGradientDescentState)
    print(io, "StochasticGradientDescentState(; ")
    print(io, "callbacks = ", sgds.callbacks, ", ")
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
    conv_inl = (i > 0) ? (has_converged(sgds.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the stochastic gradient descent algorithm$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(sgds.stop) ? "Yes" : "No"
    as = _callbacks_summary(sgds)
    s = """
    # Solver state for `Manopt.jl`s Stochastic Gradient Descent
    $Iter
    ## Parameters$(as)
    * direction: $(status_summary(sgds.direction; context = :inline))
    * order: $(sgds.order_type)
    * retraction method: $(sgds.retraction_method)

    ## Stepsize
    $(_in_str(status_summary(sgds.stepsize; context = context); indent = 0, headers = 1))

    ## Stopping criterion
    $(_in_str(status_summary(sgds.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv"""
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
[`stochastic_gradient_descent`](@ref), [`StochasticGradient`](@ref)
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
    return print(io, "StochasticGradientRule($(sg.X))")
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

$(_note(:ManifoldDefaultsFactory, "StochasticGradientRule"))
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

$(_kwargs(:callbacks; add_properties = [:process_note]))
* `cost=missing`: you can provide a cost function for example to track the function value
* `direction=`[`StochasticGradient`](@ref)`(`$(_link(:zero_vector))`)` add a post-processor to
  the direction obtained from evaluating the sub-gradient.
$(_kwargs(:evaluation))
* `order_type=:Linear`: whether to use a randomly permuted sequence (`:FixedRandom`),
  a per cycle permuted sequence (`:Random`, default) or the default `:Linear` one.
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(10000)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1.0e-9)"))
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
    p_ = maybe_wrap_variable(p)
    msgo = ManifoldStochasticGradientObjective(grad_f; cost = cost, evaluation = evaluation, p = p)
    rs = stochastic_gradient_descent(M, msgo, p_; evaluation = evaluation, kwargs...)
    return maybe_unwrap_variable(p, rs)
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
        callbacks = Dict{Symbol, Function}(),
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
        M; callbacks = process_callbacks_arg(callbacks, StochasticGradientDescentState),
        p = p, X = zero_vector(M, p),
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
    initialize_stepsize!(s.stepsize)
    return s
end
function step_solver!(mp::AbstractManoptProblem, s::StochasticGradientDescentState, iter)
    step, s.X = s.direction(mp, s, iter)
    callback(:Direction, mp, s, iter)
    retract!(get_manifold(mp), s.p, s.p, -step * s.X, s.retraction_method)
    s.k = ((s.k) % length(s.order)) + 1
    return s
end
