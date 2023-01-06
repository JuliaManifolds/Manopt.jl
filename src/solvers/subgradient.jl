"""
    SubGradientMethodState <: AbstractManoptSolverState

stories option values for a [`subgradient_method`](@ref) solver

# Fields
* `retraction_method` – the retration to use within
* `stepsize` – ([`ConstantStepsize`](@ref)`(M)`) a [`Stepsize`](@ref)
* `stop` – ([`StopAfterIteration`](@ref)`(5000)``)a [`StoppingCriterion`](@ref)
* `p` – (initial or current) value the algorithm is at
* `p_star` – optimal value (initialized to a copy of `p`.)
* `X` (`zero_vector(M, p)`) the current element from the possible subgradients at
   `p` that was last evaluated.

# Constructor

SubGradientMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields above besides `p_star` which obtains the same type as `p`.
You can use e.g. `X=` to specify the type of tangent vector to use

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
        retraction_method::TR=default_retraction_method(M),
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
    subgradient_method(M, f, ∂f, p)

perform a subgradient method ``p_{k+1} = \mathrm{retr}(p_k, s_k∂f(p_k))``,

where ``\mathrm{retr}`` is a retraction, ``s_k`` is a step size, usually the
[`ConstantStepsize`](@ref) but also be specified.
Though the subgradient might be set valued,
the argument `∂f` should always return _one_ element from the subgradient, but
not necessarily deterministic.

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`– the (sub)gradient ``\partial f: \mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subgradient.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

# Optional

* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by
   allocation (default) form `∂F(M, y)` or [`InplaceEvaluation`](@ref) in place, i.e. is
   of the form `∂F!(M, X, x)`.
* `stepsize` – ([`ConstantStepsize`](@ref)`(M)`) specify a [`Stepsize`](@ref)
* `retraction` – (`default_retraction_method(M)`) a `retraction(M,x,ξ)` to use.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(5000)`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
...
and the ones that are passed to [`decorate_state`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
function subgradient_method(
    M::AbstractManifold, f::TF, ∂f::TdF, p; kwargs...
) where {TF,TdF}
    p_star = copy(M, p)
    return subgradient_method!(M, f, ∂f, p_star; kwargs...)
end
@doc raw"""
    subgradient_method!(M, f, ∂f, x)

perform a subgradient method ``p_{k+1} = \mathrm{retr}(p_k, s_k∂f(p_k))``,

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`– the (sub)gradient ``\partial f: \mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subgradient.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

for more details and all optional parameters, see [`subgradient_method`](@ref).
"""
function subgradient_method!(
    M::AbstractManifold,
    f::TF,
    ∂f!!::TdF,
    p;
    retraction_method::TRetr=default_retraction_method(M),
    stepsize::Stepsize=default_stepsize(M, SubGradientMethodState),
    stopping_criterion::StoppingCriterion=StopAfterIteration(5000),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    mp = DefaultManoptProblem(M, sgo)
    sgs = SubGradientMethodState(
        M,
        p;
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        retraction_method=retraction_method,
    )
    sgs = decorate_state(sgs; kwargs...)
    return get_solver_return(solve!(mp, sgs))
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
