
@doc raw"""
    DifferenceOfConvexProximalState{Type} <: Options
A struct to store the current state of the algorithm as well as the form.
It comes in two forms, depending on the realisation of the `subproblem`.
# Fields
* `inverse_retraction_method` – (`default_inverse_retraction_method(M)`) an inverse retraction method to use within Frank Wolfe.
* `retraction_method` – (`default_retraction_method(M)`) a type of retraction
* `p`, `q`, `r`  – the current iterate, the gradient step and the prox, respetively
  their type is set by intializing `p`
* `stepsize` – ([`ArmijoLinesearch`](@ref)`(M)`) a [`Stepsize`](@ref) function
* `stop` – ([`StopWhenChangeLess`](@ref)`(1e-8)`) a [`StoppingCriterion`](@ref)
* `X`, `Y` – (`zero_vector(M,p)`) the current gradient and descent direction, respectively
  their common type is set by the keyword `X`
# Constructor
    DifferenceOfConvexProximalState(M, p; kwargs...)
## Keyword arguments
* `X`, `retraction_method`, `inverse_retraction_method`, `stepsize` for the fields above
* `stoppping_criterion` for the [`StoppingCriterion`](@ref)
"""
mutable struct DifferenceOfConvexProximalState{
    P,
    T,
    S<:Stepsize,
    SC<:StoppingCriterion,
    RTR<:AbstractRetractionMethod,
    ITR<:AbstractInverseRetractionMethod,
    Tλ,
} <: AbstractManoptSolverState
    λ::Tλ
    p::P
    q::P
    r::P
    X::T
    retraction_method::RTR
    inverse_retraction_method::ITR
    stepsize::S
    stop::SC
    function DifferenceOfConvexProximalState(
        M::AbstractManifold,
        p::P;
        X::T=zero_vector(M, p),
        stepsize::S=ArmijoLinesearch(M),
        stopping_criterion::SC=StopWhenChangeLess(1e-8),
        inverse_retraction_method::I=default_inverse_retraction_method(M),
        retraction_method::R=default_retraction_method(M),
        λ::Fλ=i -> 1,
    ) where {
        P,
        T,
        S<:Stepsize,
        SC<:StoppingCriterion,
        I<:AbstractInverseRetractionMethod,
        R<:AbstractRetractionMethod,
        Fλ,
    }
        return new{P,T,S,SC,R,I,Fλ}(
            λ,
            p,
            copy(M, p),
            copy(M, p),
            X,
            retraction_method,
            inverse_retraction_method,
            stepsize,
            stopping_criterion,
        )
    end
end
get_iterate(dcps::DifferenceOfConvexProximalState) = dcps.p
function set_iterate!(dcps::DifferenceOfConvexProximalState, p)
    dcps.p = p
    return dcps
end
get_gradient(dcps::DifferenceOfConvexProximalState) = dcps.X

#
# Prox approach
#
@doc raw"""
    difference_of_convex_proximal_point(M, prox_g, grad_h, p; kwargs...)

Compute the difference of convex algorithm to minimize

```math
    \operatorname*{arg\,min}_{p∈\mathcal M} g(p) - h(p)
```

where you have to provide the proximal map of `g` and the gradient of `h`.
Optionally, the cost can also be provided, e.g. for debug or recordings.
This algorithm performs the following steps given a start point `p`= ``p^{(0)}``.
Then repeat for ``k=0,1,\ldots``

    1. ``X^{(k)}  ∈ \operatorname{grad} h(p^{(k)})``
2. ``q^{(k)} = \operatorname{retr}_{p^{(k)}}(λ_kX^{(k)})``
3. ``r^{(k)} = \operatorname{prox}_{λ_kg}(q^{(k)})``
4. ``X^{(k)} = \operatorname{retr}^{-1}_{p^{(k)}}(r^{(k)})``
5. Compute a stepsize ``s_k`` and
6. set ``p^{(k+1)} = \operatorname{retr}_{p^{(k)}(s_kX^{(k)})``.

until the `stopping_criterion` is fulfilled.
See [^AlmeidaNetoOliveiraSouza2020] for more details,
where we slightly changed step 4-6, sine here we get the classical proximal point
method for DC functions for ``s_k = 1`` we obtain the classical proximal method for

# Optional parameters

* `λ` – ( `i -> 1/2` ) a function returning the sequence of prox parameters λi
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `gradF(M, x)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
* `cost` (`nothing`) provide the cost `f`, e.g. for debug reasonscost to be used within the default `sub_problem`
  use this if you have a more efficient version than using `g` from above.
* `inverse_retraction_method` - (`default_inverse_retraction_method(M)`) an inverse retraction method to use (see step 4).
* `retraction_method` – (`default_retraction_method(M)`) a retraction to use (see step 2)
* `stepsize` – ([`ArmijoLinesearch`](@ref)`(M)`) specify a [`Stepsize`](@ref)
  functor.
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(200)` a [`StoppingCriterion`](@ref) for the algorithm.
...all others are passed on to decorate the inner [`DifferenceOfConvexOptions`](@ref).

# Output
the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details

[^AlmeidaNetoOliveiraSouza2020]:
    > Almeida, Y. T., de Cruz Neto, J. X. Oliveira, P. R., de Oliveira Souza, J. C.:
    > _A modified proximal point method for DC functions on Hadamard manifolds_,
    Computational Optimization and Applications (76), 2020, pp. 649–673.
    > doi: [10.1007/s10589-020-00173-3](https://doi.org/10.1007/s10589-020-00173-3)
"""
function difference_of_convex_proximal_point(
    M::AbstractManifold, prox_g, grad_h, p; kwargs...
)
    q = copy(M, p)
    difference_of_convex_proximal_point!(M, grad_g, grad_h, q; kwargs...)
    return q
end

@doc raw"""
    difference_of_convex_proximal_point!(M, prox_g, grad_h, p; cost=nothing, kwargs...)

Compute the difference of convex algorithm to minimize

```math
    \operatorname*{arg\,min}_{p∈\mathcal M} g(p) - h(p)
```

where you have to provide the proximal map of `g` and the gradient of `h`.

The compuation is done inplace of `p`.

For all further details, especially the keyword arguments, see [`difference_of_convex_proximal_point`](@ref).
"""
function difference_of_convex_proximal_point!(
    M,
    prox_g,
    grad_h,
    p;
    X=zero_vector(M, p),
    λ=i -> 1 / 2,
    evaluation=AllocatingEvaluation(),
    cost=nothing,
    inverse_retraction_method=default_inverse_retraction_method(M),
    retraction_method=default_retraction_method(M),
    stepsize=ArmijoLinesearch(M),
    stopping_criterion=StopAfterIteration,
    kwargs...,
)
    mdcpo = ManifoldDifferenceOfConvexProximalObjective(
        prox_g, grad_h; cost=cost, evaluation=evaluation
    )
    dmdcpo = decorate_objective!(M, mdcpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmdcpo)
    dcps = DifferenceOfConvexProximalState(
        M,
        p;
        X=X,
        stepsize=stepsize,
        stopping_criterion=stopping_criterion,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
    )
    ddcps = decorate_options(dcps; kwargs...)
    return get_solver_return(solve!(dmp, ddcps))
end
function initialize_solver!(
    amp::AbstractManoptProblem, dcps::DifferenceOfConvexProximalState
)
    return dcps
end
function step_solver!(amp::AbstractManoptProblem, dcps::DifferenceOfConvexProximalState, i)
    M = get_manifold(amp)
    # each line is one step in the documented solver steps. Note that we can reuse dcps.X
    get_gradient!(amp, dcps.X, dcps.p)
    retract!(M, dcps.q, dcps.p, dpcs.λ(i) * dcps.X, dcps.retraction_method)
    get_proximal_map!(amp, dcps.r, dpcs.λ(i), dcps.q)
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    s = dcps.stepsize(amp, dcps, i)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    return dcps
end
