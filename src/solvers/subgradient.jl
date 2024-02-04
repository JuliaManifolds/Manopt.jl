"""
    SubGradientMethodState <: AbstractManoptSolverState

stores option values for a [`subgradient_method`](@ref) solver

# Fields

* `retraction_method`: the retraction to use within
* `stepsize`:          ([`ConstantStepsize`](@ref)`(M)`) a [`Stepsize`](@ref)
* `stop`:              ([`StopAfterIteration`](@ref)`(5000)``)a [`StoppingCriterion`](@ref)
* `p`:                 (initial or current) value the algorithm is at
* `p_star`:            optimal value (initialized to a copy of `p`.)
* `X`:                 (`zero_vector(M, p)`) the current element from the possible
  subgradients at `p` that was last evaluated.

# Constructor

    SubGradientMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields besides `p_star` which obtains the same type as `p`.
You can use `X=` to specify the type of tangent vector to use
"""
mutable struct SubGradientMethodState{
    TR<:AbstractRetractionMethod,TS<:Stepsize,TSC<:StoppingCriterion,P,T
} <: AbstractManoptSolverState where {P,T}
    p::P
    p_star::P
    retraction_method::TR
    stepsize::TS
    stop::TSC
    X::T
    function SubGradientMethodState(
        M::TM,
        p::P;
        stopping_criterion::SC=StopAfterIteration(5000),
        stepsize::S=default_stepsize(M, SubGradientMethodState),
        X::T=zero_vector(M, p),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
    ) where {
        TM<:AbstractManifold,
        P,
        T,
        SC<:StoppingCriterion,
        S<:Stepsize,
        TR<:AbstractRetractionMethod,
    }
        return new{TR,S,SC,P,T}(
            p, copy(M, p), retraction_method, stepsize, stopping_criterion, X
        )
    end
end
function show(io::IO, sgms::SubGradientMethodState)
    i = get_count(sgms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(sgms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Subgradient Method
    $Iter
    ## Parameters
    * retraction method: $(sgms.retraction_method)

    ## Stepsize
    $(sgms.stepsize)

    ## Stopping criterion

    $(status_summary(sgms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
get_iterate(sgs::SubGradientMethodState) = sgs.p
get_subgradient(sgs::SubGradientMethodState) = sgs.X
function set_iterate!(sgs::SubGradientMethodState, M, p)
    copyto!(M, sgs.p, p)
    return sgs
end
function default_stepsize(M::AbstractManifold, ::Type{SubGradientMethodState})
    return ConstantStepsize(M)
end

@doc raw"""
    subgradient_method(M, f, ∂f, p; kwargs...)
    subgradient_method(M; sgo, p; kwargs...)

perform a subgradient method ``p_{k+1} = \mathrm{retr}(p_k, s_k∂f(p_k))``,

where ``\mathrm{retr}`` is a retraction, ``s_k`` is a step size, usually the
[`ConstantStepsize`](@ref) but also be specified.
Though the subgradient might be set valued,
the argument `∂f` should always return _one_ element from the subgradient, but
not necessarily deterministic.

# Input

* `M`:  a manifold ``\mathcal M``
* `f`:  a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`: the (sub)gradient ``∂ f: \mathcal M→ T\mathcal M`` of f
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p`:  an initial value ``p_0=p ∈ \mathcal M``

alternatively to `f` and `∂f` a [`ManifoldSubgradientObjective`](@ref) `sgo` can be provided.

# Optional

* `evaluation`:         ([`AllocatingEvaluation`](@ref)) specify whether the subgradient
  works by allocation (default) form `∂f(M, y)` or [`InplaceEvaluation`](@ref) in place
  of the form `∂f!(M, X, x)`.
* `retraction`:         (`default_retraction_method(M, typeof(p))`) a retraction to use.
* `stepsize`:           ([`ConstantStepsize`](@ref)`(M)`) specify a [`Stepsize`](@ref)
* `stopping_criterion`: ([`StopAfterIteration`](@ref)`(5000)`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.

and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
subgradient_method(::AbstractManifold, args...; kwargs...)
function subgradient_method(M::AbstractManifold, f, ∂f; kwargs...)
    return subgradient_method(M, f, ∂f, rand(M); kwargs...)
end
function subgradient_method(
    M::AbstractManifold,
    f,
    ∂f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    sgo = ManifoldSubgradientObjective(f, ∂f; evaluation=evaluation)
    return subgradient_method(M, sgo, p; evaluation=evaluation, kwargs...)
end
function subgradient_method(
    M::AbstractManifold,
    f,
    ∂f,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    q = [p]
    f_(M, p) = f(M, p[])
    ∂f_ = _to_mutating_gradient(∂f, evaluation)
    rs = subgradient_method(M, f_, ∂f_, q; evaluation=evaluation, kwargs...)
    #return just a number if  the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function subgradient_method(
    M::AbstractManifold, sgo::O, p; kwargs...
) where {O<:Union{ManifoldSubgradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return subgradient_method!(M, sgo, q; kwargs...)
end

@doc raw"""
    subgradient_method!(M, f, ∂f, p)
    subgradient_method!(M, sgo, p)

perform a subgradient method ``p_{k+1} = \mathrm{retr}(p_k, s_k∂f(p_k))``,

# Input

* `M`:  a manifold ``\mathcal M``
* `f`:  a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`: the (sub)gradient ``∂f: \mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p`:  an initial value ``p_0=p ∈ \mathcal M``

alternatively to `f` and `∂f` a [`ManifoldSubgradientObjective`](@ref) `sgo` can be provided.

for more details and all optional parameters, see [`subgradient_method`](@ref).
"""
subgradient_method!(M::AbstractManifold, args...; kwargs...)
function subgradient_method!(
    M::AbstractManifold,
    f,
    ∂f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    sgo = ManifoldSubgradientObjective(f, ∂f; evaluation=evaluation)
    return subgradient_method!(M, sgo, p; evaluation=evaluation, kwargs...)
end
function subgradient_method!(
    M::AbstractManifold,
    sgo::O,
    p;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stepsize::Stepsize=default_stepsize(M, SubGradientMethodState),
    stopping_criterion::StoppingCriterion=StopAfterIteration(5000),
    kwargs...,
) where {O<:Union{ManifoldSubgradientObjective,AbstractDecoratedManifoldObjective}}
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    sgs = SubGradientMethodState(
        M,
        p;
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        retraction_method=retraction_method,
    )
    dsgs = decorate_state!(sgs; kwargs...)
    solve!(mp, dsgs)
    return get_solver_return(get_objective(mp), dsgs)
end
function initialize_solver!(mp::AbstractManoptProblem, sgs::SubGradientMethodState)
    M = get_manifold(mp)
    copyto!(M, sgs.p_star, sgs.p)
    sgs.X = zero_vector(M, sgs.p)
    return sgs
end
function step_solver!(mp::AbstractManoptProblem, sgs::SubGradientMethodState, i)
    get_subgradient!(mp, sgs.X, sgs.p)
    step = get_stepsize(mp, sgs, i)
    M = get_manifold(mp)
    retract!(M, sgs.p, sgs.p, -step * sgs.X, sgs.retraction_method)
    (get_cost(mp, sgs.p) < get_cost(mp, sgs.p_star)) && copyto!(M, sgs.p_star, sgs.p)
    return sgs
end
get_solver_result(sgs::SubGradientMethodState) = sgs.p_star
function (cs::ConstantStepsize)(
    amp::AbstractManoptProblem, sgs::SubGradientMethodState, ::Any, args...; kwargs...
)
    s = cs.length
    if cs.type == :absolute
        ns = norm(get_manifold(amp), get_iterate(sgs), get_subgradient(sgs))
        if ns > eps(eltype(s))
            s /= ns
        end
    end
    return s
end
function (s::DecreasingStepsize)(
    amp::AbstractManoptProblem, sgs::SubGradientMethodState, i::Int, args...; kwargs...
)
    ds = (s.length - i * s.subtrahend) * (s.factor^i) / ((i + s.shift)^(s.exponent))
    if s.type == :absolute
        ns = norm(get_manifold(amp), get_iterate(sgs), get_subgradient(sgs))
        if ns > eps(eltype(ds))
            ds /= ns
        end
    end
    return ds
end
