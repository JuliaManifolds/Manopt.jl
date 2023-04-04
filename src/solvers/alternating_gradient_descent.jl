"""
    AlternatingGradientDescentState <: AbstractGradientDescentSolverState

Store the fields for an alternating gradient descent algorithm,
see also [`alternating_gradient_descent`](@ref).

# Fields
* `direction` (`AlternatingGradient(zero_vector(M, x))` a [`DirectionUpdateRule`](@ref)
* `evaluation_order` – (`:Linear`) – whether
* `inner_iterations`– (`5`) how many gradient steps to take in a component before alternating to the next
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle newly permuted sequence (`:Random`) or the default `:Linear` evaluation order.
* `order` the current permutation
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) a `retraction(M,x,ξ)` to use.
* `stepsize` ([`ConstantStepsize`](@ref)`(M)`) a [`Stepsize`](@ref)
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `p` the current iterate
* `X` (`zero_vector(M,p)`) the current gradient tangent vector
* `k`, ì` internal counters for the outer and inner iterations, respectively.

# Constructors

    AlternatingGradientDescentState(M, p; kwargs...)

Generate the options for point `p` and and where `inner_iterations`, `order_type`, `order`,
`retraction_method`, `stopping_criterion`, and `stepsize`` are keyword arguments
"""
mutable struct AlternatingGradientDescentState{
    P,
    T,
    D<:DirectionUpdateRule,
    TStop<:StoppingCriterion,
    TStep<:Stepsize,
    RM<:AbstractRetractionMethod,
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
    M::AbstractManifold,
    p::P;
    X::T=zero_vector(M, p),
    inner_iterations::Int=5,
    order_type::Symbol=:Linear,
    order::Vector{<:Int}=Int[],
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopAfterIteration(1000),
    stepsize::Stepsize=default_stepsize(M, AlternatingGradientDescentState),
) where {P,T}
    return AlternatingGradientDescentState{
        P,
        T,
        AlternatingGradient,
        typeof(stopping_criterion),
        typeof(stepsize),
        typeof(retraction_method),
    }(
        p,
        X,
        AlternatingGradient(zero_vector(M, p)),
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

    ## Stopping Criterion
    $(status_summary(agds.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
function get_message(agds::AlternatingGradientDescentState)
    # for now only step size is quipped with messages
    return get_message(agds.stepsize)
end
function get_message_type(agds::AlternatingGradientDescentState)
    # for now only step size is quipped with messages
    return get_message_type(agds.stepsize)
end

"""
    AlternatingGradient <: DirectionUpdateRule

The default gradient processor, which just evaluates the (alternating) gradient on one of
the components
"""
struct AlternatingGradient{T} <: AbstractGradientGroupProcessor
    dir::T
end

function (ag::AlternatingGradient)(
    amp::AbstractManoptProblem, agds::AlternatingGradientDescentState, i
)
    M = get_manifold(amp)
    # at begin of inner iterations reset internal vector to zero
    (i == 1) && zero_vector!(M, ag.dir, agds.p)
    # update order(k)th component inplace
    get_gradient!(amp, ag.dir[M, agds.order[agds.k]], agds.p, agds.order[agds.k])
    return agds.stepsize(amp, agds, i), ag.dir # return urrent full gradient
end

# update Armijo to work on the kth gradient only.
function (a::ArmijoLinesearch)(
    amp::AbstractManoptProblem, agds::AlternatingGradientDescentState, ::Int
)
    M = get_manifold(amp)
    X = zero_vector(M, agds.p)
    get_gradient!(amp, X[M, agds.order[agds.k]], agds.p, agds.order[agds.k])
    (a.last_stepsize, a.message) = linesearch_backtrack(
        M,
        p -> get_cost(amp, p),
        agds.p,
        X,
        a.last_stepsize,
        a.sufficient_decrease,
        a.contraction_factor,
        a.retraction_method,
    )
    return a.last_stepsize
end

function default_stepsize(M::AbstractManifold, ::Type{AlternatingGradientDescentState})
    return ArmijoLinesearch(M)
end

@doc raw"""
    alternating_gradient_descent(M::ProductManifold, f, grad_f, p)

perform an alternating gradient descent

# Input

* `M` – the product manifold ``\mathcal M = \mathcal M_1 × \mathcal M_2 × ⋯ ×\mathcal M_n``
* `f` – the objective function (cost) defined on `M`.
* `grad_f` – a gradient, that can be of two cases
  * is a single function returning a `ProductRepr` or
  * is a vector functions each returning a component part of the whole gradient
* `p` – an initial value ``p_0 ∈ \mathcal M``

# Optional
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient(s) works by
   allocation (default) form `gradF(M, x)` or [`InplaceEvaluation`](@ref) in place, i.e.
   is of the form `gradF!(M, X, x)` (elementwise).
* `evaluation_order` – (`:Linear`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Random`) or the default `:Linear` one.
* `inner_iterations`– (`5`) how many gradient steps to take in a component before alternating to the next
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ArmijoLinesearch`](@ref)`()`) a [`Stepsize`](@ref)
* `order` - (`[1:n]`) the initial permutation, where `n` is the number of gradients in `gradF`.
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) a `retraction(M, p, X)` to use.

# Output

usually the obtained (approximate) minimizer, see [`get_solver_return`](@ref) for details

!!! note

    This Problem requires the `ProductManifold` from `Manifolds.jl`, so `Manifolds.jl` to be loaded.

!!! note

    The input of each of the (component) gradients is still the whole vector `x`,
    just that all other then the `i`th input component are assumed to be fixed and just
    the `i`th components gradient is computed / returned.

"""
function alternating_gradient_descent(
    M::ProductManifold, f, grad_f::Union{TgF,AbstractVector{<:TgF}}, p; kwargs...
) where {TgF}
    q = copy(M, p)
    return alternating_gradient_descent!(M, f, grad_f, q; kwargs...)
end
@doc raw"""
    alternating_gradient_descent!(M::ProductManifold, f, grad_f, p)

perform a alternating gradient descent in place of `p`.

# Input

* `M` a product manifold ``\mathcal M``
* `f` – the objective functioN (cost)
* `grad_f` – a gradient function, that either returns a vector of the subgradients
  or is a vector of gradients
* `p` – an initial value ``p_0 ∈ \mathcal M``

for all optional parameters, see [`alternating_gradient_descent`](@ref).
"""
function alternating_gradient_descent!(
    M::ProductManifold,
    f,
    grad_f::Union{TgF,AbstractVector{<:TgF}},
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inner_iterations::Int=5,
    stopping_criterion::StoppingCriterion=StopAfterIteration(100) |
                                          StopWhenGradientNormLess(1e-9),
    stepsize::Stepsize=default_stepsize(M, AlternatingGradientDescentState),
    order_type::Symbol=:Linear,
    order=collect(1:(grad_f isa Function ? length(grad_f(M, p)) : length(grad_f))),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
) where {TgF}
    agmo = ManifoldAlternatingGradientObjective(f, grad_f; evaluation=evaluation)
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
function step_solver!(amp::AbstractManoptProblem, agds::AlternatingGradientDescentState, i)
    M = get_manifold(amp)
    step, agds.X = agds.direction(amp, agds, i)
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
