"""
    AlternatingGradientDescentState <: AbstractGradientDescentSolverState

Store the fields for an alternating gradient descent algorithm,
see also [`alternating_gradient_descent`](@ref).

# Fields

* `direction::`[`DirectionUpdateRule`](@ref)
* `evaluation_order::Symbol`: whether to use a randomly permuted sequence (`:FixedRandom`),
  a per cycle newly permuted sequence (`:Random`) or the default `:Linear` evaluation order.
* `inner_iterations`: how many gradient steps to take in a component before alternating to the next
* `order`: the current permutation
$(_fields([:retraction_method, :stepsize]))
$(_fields(:stopping_criterion; name = "stop"))
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:X; add_properties = [:as_Gradient]))
* `k`, ì`:              internal counters for the outer and inner iterations, respectively.

# Constructors

    AlternatingGradientDescentState(M::AbstractManifold; kwargs...)

# Keyword arguments
* `inner_iterations=5`
$(_kwargs(:p))
* `order_type::Symbol=:Linear`
* `order::Vector{<:Int}=Int[]`
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)"))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, AlternatingGradientDescentState)"))
$(_kwargs(:X))

Generate the options for point `p` and where `inner_iterations`, `order_type`, `order`,
`retraction_method`, `stopping_criterion`, and `stepsize`` are keyword arguments
"""
mutable struct AlternatingGradientDescentState{
        P,
        T,
        D <: DirectionUpdateRule,
        TStop <: StoppingCriterion,
        TStep <: Stepsize,
        RM <: AbstractRetractionMethod,
    } <: AbstractGradientSolverState
    p::P
    X::T
    direction::D
    stop::TStop
    stepsize::TStep
    order_type::Symbol
    order::Vector{<:Int}
    retraction_method::RM
    k::Int # current iterate
    i::Int # inner iterate
    inner_iterations::Int
end
function AlternatingGradientDescentState(
        M::AbstractManifold;
        p::P = rand(M),
        X::T = zero_vector(M, p),
        inner_iterations::Int = 5,
        order_type::Symbol = :Linear,
        order::Vector{<:Int} = Int[],
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stopping_criterion::StoppingCriterion = StopAfterIteration(1000),
        stepsize::Stepsize = default_stepsize(M, AlternatingGradientDescentState),
    ) where {P, T}
    return AlternatingGradientDescentState{
        P,
        T,
        AlternatingGradientRule,
        typeof(stopping_criterion),
        typeof(stepsize),
        typeof(retraction_method),
    }(
        p,
        X,
        _produce_type(AlternatingGradient(; p = p, X = X), M),
        stopping_criterion,
        stepsize,
        order_type,
        order,
        retraction_method,
        0,
        0,
        inner_iterations,
    )
end
function show(io::IO, agds::AlternatingGradientDescentState)
    Iter = (agds.i > 0) ? "After $(agds.i) iterations\n" : ""
    Conv = indicates_convergence(agds.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Alternating Gradient Descent Solver
    $Iter
    ## Parameters
    * order: :$(agds.order_type)
    * retraction method: $(agds.retraction_method)


    ## Stepsize
    $(agds.stepsize)

    ## Stopping criterion

    $(status_summary(agds.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
function get_message(agds::AlternatingGradientDescentState)
    # for now only step size is quipped with messages
    return get_message(agds.stepsize)
end

"""
    AlternatingGradientRule <: AbstractGradientGroupDirectionRule

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
[`alternating_gradient_descent`](@ref), [`AlternatingGradient`])@ref)
"""
struct AlternatingGradientRule{T} <: AbstractGradientGroupDirectionRule
    X::T
end
function AlternatingGradientRule(
        M::AbstractManifold; p = rand(M), X::T = zero_vector(M, p)
    ) where {T}
    return AlternatingGradientRule{T}(X)
end

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

$(_kwargs(:X, name = "initial_gradient"))
$(_kwargs(:p; add_properties = [:as_Initial]))

$(_note(:ManifoldDefaultFactory, "AlternatingGradientRule"))
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

function default_stepsize(M::AbstractManifold, ::Type{AlternatingGradientDescentState})
    return ArmijoLinesearchStepsize(M)
end

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
* `evaluation_order=:Linear`: whether to use a randomly permuted sequence (`:FixedRandom`),
  a per cycle permuted sequence (`:Random`) or the default `:Linear` one.
* `inner_iterations=5`:  how many gradient steps to take in a component before alternating to the next
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)`)"))
$(_kwargs(:stepsize; default = "`[`ArmijoLinesearch`](@ref)`()"))
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
    return agds
end
function step_solver!(amp::AbstractManoptProblem, agds::AlternatingGradientDescentState, k)
    M = get_manifold(amp)
    step, agds.X = agds.direction(amp, agds, k)
    j = agds.order[agds.k]
    retract!(M[j], agds.p[M, j], agds.p[M, j], -step * agds.X[M, j])
    agds.i += 1
    if agds.i > agds.inner_iterations
        agds.k = ((agds.k) % length(agds.order)) + 1
        (agds.order_type == :Random) && (shuffle!(agds.order))
        agds.i = 1
    end
    return agds
end
