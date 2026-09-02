"""
    AlternatingGradientRule <: AbstractGradientGroupDirectionRule

The direction processor to alternate the gradient directions.

Create a functor `(problem, state k) -> (s,X)` to evaluate the alternating gradient,
that is alternating between the components of the gradient and has an field for
partial evaluation of the gradient in-place.

# Fields

$(_fields(:X))

# Constructor

    AlternatingGradientRule(M::AbstractManifold; p=rand(M), X=zero_vector(M, p))

Initialize the alternating gradient processor with tangent vector type of `X`,
where both `M` and `p` are just help variables.

# See also
[`alternating_gradient_descent`](@ref), [`AlternatingGradient`](@ref)
"""
struct AlternatingGradientRule{T} <: AbstractGradientGroupDirectionRule
    X::T
end
function AlternatingGradientRule(
        M::AbstractManifold; p = rand(M), X::T = zero_vector(M, p)
    ) where {T}
    return AlternatingGradientRule{T}(X)
end
function Base.show(io::IO, ag::AlternatingGradientRule)
    return print(io, "AlternatingGradientRule($(ag.X))")
end
function status_summary(ag::AlternatingGradientRule; context::Symbol = :default)
    (context === :short) && return repr(ag)
    return "A alternating gradient processor"
end
"""
    AlternatingGradientDescentState <: AbstractGradientDescentSolverState

Store the fields for an alternating gradient descent algorithm,
see also [`alternating_gradient_descent`](@ref).

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
* `direction::`[`DirectionUpdateRule`](@ref)
* `order_type::Symbol`: whether to use a randomly permuted sequence (`:FixedRandom`),
  a per cycle newly permuted sequence (`:Random`) or the default `:Linear` evaluation order.
* `inner_iterations`: how many gradient steps to take in a component before alternating to the next
* `order`: the current permutation
$(_fields([:retraction_method, :stepsize]))
$(_fields(:stopping_criterion; name = "stop"))
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:X; add_properties = [:as_Gradient]))
* `k`, `i`: internal counters for the outer and inner iterations, respectively.

# Constructors

    AlternatingGradientDescentState(M::AbstractManifold; kwargs...)

# Keyword arguments
* `inner_iterations=5`
$(_kwargs(:p))
$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `order_type::Symbol=:Linear`
* `order::Vector{<:Int}=Int[]`
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)"))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, AlternatingGradientDescentState)"))
$(_kwargs(:X))

Generate the options for point `p` and where `inner_iterations`, `order_type`, `order`,
`retraction_method`, `stopping_criterion`, and `stepsize` are keyword arguments.

For internal use, there also exists a constructor solely having the fields as keyword arguments,
but then all of them are mandatory.
"""
mutable struct AlternatingGradientDescentState{
        P, T, C <: AbstractDict{Symbol}, D <: DirectionUpdateRule,
        TStop <: StoppingCriterion, TStep <: Stepsize, RM <: AbstractRetractionMethod,
    } <: AbstractGradientSolverState
    callbacks::C
    p::P
    X::T
    direction::D
    stop::TStop
    stepsize::TStep
    order_type::Symbol
    order::Vector{<:Int}
    retraction_method::RM
    k::Int # current component
    i::Int # inner iterate
    inner_iterations::Int
    function AlternatingGradientDescentState(
            M::AbstractManifold;
            p::P = rand(M), X::T = zero_vector(M, p),
            callbacks::C = Dict{Symbol, Function}(),
            inner_iterations::Int = 5,
            order_type::Symbol = :Linear, order::Vector{<:Int} = Int[],
            retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
            stopping_criterion::StoppingCriterion = StopAfterIteration(1000),
            stepsize::Stepsize = default_stepsize(
                M, AlternatingGradientDescentState; retraction_method = retraction_method
            ),
        ) where {P, T, C <: AbstractDict{Symbol}}
        (order_type in (:Linear, :FixedRandom, :Random)) || throw(
            DomainError(order_type, "The order type has to be one of :Linear, :FixedRandom, or :Random.")
        )
        return AlternatingGradientDescentState(;
            callbacks = callbacks,
            p = p, X = X, direction = _produce_type(AlternatingGradient(; p = p, X = X), M),
            inner_iterations = inner_iterations,
            order_type = order_type, order = order,
            retraction_method = retraction_method, stopping_criterion = stopping_criterion,
            stepsize = stepsize,
        )
    end
    function AlternatingGradientDescentState(;
            callbacks::CA, p::P, X::T, direction::D, inner_iterations::Int, order_type::Symbol, order::Vector{<:Int},
            retraction_method::RTM, stopping_criterion::SC, stepsize::S, k::Int = 0, i::Int = 0
        ) where {P, T, CA <: AbstractDict{Symbol}, RTM <: AbstractRetractionMethod, SC <: StoppingCriterion, S <: Stepsize, D <: AlternatingGradientRule}
        return new{P, T, CA, D, SC, S, RTM}(
            callbacks, p, X, direction, stopping_criterion, stepsize,
            order_type, order, retraction_method, k, i, inner_iterations,
        )
    end
end
function Base.show(io::IO, agds::AlternatingGradientDescentState)
    print(io, "AlternatingGradientDescentState(; ")
    print(io, "callbacks = ", agds.callbacks, ", ")
    print(io, "p = $(agds.p), ")
    print(io, "X = $(agds.X), ")
    print(io, "direction = $(agds.direction), ")
    print(io, "inner_iterations = $(agds.inner_iterations), ")
    print(io, "order_type = :$(agds.order_type), ")
    print(io, "order = $(agds.order), ")
    print(io, "retraction_method = $(agds.retraction_method), ")
    print(io, "stepsize = $(agds.stepsize), ")
    print(io, "stopping_criterion = $(status_summary(agds.stop, context = :short)), ")
    return print(io, "i = $(agds.i), k = $(agds.k))")
end
function status_summary(agds::AlternatingGradientDescentState; context::Symbol = :default)
    (context === :short) && return repr(agds)
    i = get_count(agds, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(agds.stop) ? "Yes" : "No"
    conv_inl = (i > 0) ? (has_converged(agds.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the alternating gradient descent solver$(conv_inl)"
    as = _callbacks_summary(agds)
    s = """
    # Solver state for `Manopt.jl`s Alternating Gradient Descent Solver
    $Iter
    ## Parameters$(as)
    * order: :$(agds.order_type)
    * retraction method: $(agds.retraction_method)
    * direction: $(status_summary(agds.direction; context = :inline))

    ## Stepsize
    $(agds.stepsize)

    ## Stopping criterion
    $(_in_str(status_summary(agds.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv"""
    return s
end
function get_message(agds::AlternatingGradientDescentState)
    # for now only step size is quipped with messages
    return get_message(agds.stepsize)
end
get_callbacks(agds::AlternatingGradientDescentState) = agds.callbacks
provided_callbacks(::Type{<:AlternatingGradientDescentState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:Stepsize])

function (ag::AlternatingGradientRule)(
        amp::AbstractManoptProblem, agds::AlternatingGradientDescentState, k
    )
    M = get_manifold(amp)
    # at begin of inner iterations reset internal vector to zero
    (k == 1) && zero_vector!(M, ag.X, agds.p)
    # update order(k)th component in-place
    get_gradient!(amp, ag.X[M, agds.order[agds.k]], agds.p, agds.order[agds.k])
    return agds.stepsize(amp, agds, k; gradient = ag.X), ag.X # return current full gradient
end

@doc """
    AlternatingGradient(; kwargs...)
    AlternatingGradient(M::AbstractManifold; kwargs...)

Specify that a gradient based method should only update parts of the gradient
in order to do a alternating gradient descent.

# Keyword arguments

$(_kwargs(:X))
$(_kwargs(:p; add_properties = [:as_Initial]))

$(_note(:ManifoldDefaultsFactory, "AlternatingGradientRule"))
"""
function AlternatingGradient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.AlternatingGradientRule, args...; kwargs...)
end

# update Armijo to work on the kth gradient only.
function (a::ArmijoLinesearchStepsize)(
        amp::AbstractManoptProblem, agds::AlternatingGradientDescentState, ::Int, η;
        kwargs...
    )
    reset_messages!(a.messages)
    M = get_manifold(amp)
    X = zero_vector(M, agds.p)
    get_gradient!(amp, X[M, agds.order[agds.k]], agds.p, agds.order[agds.k])
    a.last_stepsize = linesearch_backtrack!(
        M,
        a.candidate_point,
        (M, p) -> get_cost(amp, p),
        agds.p,
        a.last_stepsize,
        a.sufficient_decrease,
        a.contraction_factor,
        -X;
        gradient = X,
        retraction_method = a.retraction_method,
        report_messages_in = a.messages,
    )
    return a.last_stepsize
end

function default_stepsize(
        M::AbstractManifold,
        ::Type{AlternatingGradientDescentState};
        retraction_method = default_retraction_method(M),
    )
    return ArmijoLinesearchStepsize(M; retraction_method = retraction_method)
end

# the line search works on the product manifold, the update on a single component
_component_retraction(r::AbstractRetractionMethod, ::Int) = r
_component_retraction(r::ProductRetraction, j::Int) = r.retractions[j]

function alternating_gradient_descent end
function alternating_gradient_descent! end

_doc_AGD = """
    alternating_gradient_descent(M::ProductManifold, f, grad_f, p=rand(M))
    alternating_gradient_descent(M::ProductManifold, ago::ManifoldAlternatingGradientObjective, p)
    alternating_gradient_descent!(M::ProductManifold, f, grad_f, p)
    alternating_gradient_descent!(M::ProductManifold, ago::ManifoldAlternatingGradientObjective, p)

perform an alternating gradient descent. This can be done in-place of the start point `p`

# Input

$(_args([:M, :f]))
* `grad_f`: a gradient, that can be of two cases
  * is a single function returning an `ArrayPartition` from [`RecursiveArrayTools.jl`](https://docs.sciml.ai/RecursiveArrayTools/stable/array_types/) or
  * is a vector functions each returning a component part of the whole gradient
$(_args(:p))

# Keyword arguments

$(_kwargs(:evaluation))
* `order_type=:Linear`: whether to use a randomly permuted sequence (`:FixedRandom`),
  a per cycle permuted sequence (`:Random`, default) or the default `:Linear` one.
* `inner_iterations=5`:  how many gradient steps to take in a component before alternating to the next
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)"))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`AlternatingGradientDescentState`](@ref)`; retraction_method=retraction_method)"))
* `order=[1:n]`:         the initial permutation, where `n` is the number of gradients in `gradF`.
$(_kwargs(:retraction_method))

# Output

usually the obtained (approximate) minimizer, see [`get_solver_return`](@ref) for details

!!! note

    The input of each of the (component) gradients is still the whole vector `X`,
    just that all other then the `i`th input component are assumed to be fixed and just
    the `i`th components gradient is computed / returned.
"""

@doc "$(_doc_AGD)"
alternating_gradient_descent(::AbstractManifold, args...; kwargs...)
calls_with_kwargs(::typeof(alternating_gradient_descent)) = (alternating_gradient_descent!,)

@doc "$(_doc_AGD)"
alternating_gradient_descent!(M::AbstractManifold, args...; kwargs...)
calls_with_kwargs(::typeof(alternating_gradient_descent!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(
        amp::AbstractManoptProblem, agds::AlternatingGradientDescentState
    )
    agds.k = 1
    agds.i = 1
    get_gradient!(amp, agds.X, agds.p)
    (agds.order_type == :FixedRandom || agds.order_type == :Random) &&
        (shuffle!(agds.order))
    initialize_stepsize!(agds.stepsize)
    return agds
end
function step_solver!(amp::AbstractManoptProblem, agds::AlternatingGradientDescentState, k)
    M = get_manifold(amp)
    step, agds.X = agds.direction(amp, agds, k)
    callback(:Stepsize, amp, agds, k)
    j = agds.order[agds.k]
    retract!(
        M[j], agds.p[M, j], agds.p[M, j], -step * agds.X[M, j],
        _component_retraction(agds.retraction_method, j),
    )
    agds.i += 1
    if agds.i > agds.inner_iterations
        agds.k = ((agds.k) % length(agds.order)) + 1
        (agds.order_type == :Random) && (shuffle!(agds.order))
        agds.i = 1
    end
    return agds
end
