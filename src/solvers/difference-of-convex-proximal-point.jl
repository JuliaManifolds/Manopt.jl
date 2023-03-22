
@doc raw"""
    DifferenceOfConvexProximalState{Type} <: Options
A struct to store the current state of the algorithm as well as the form.
It comes in two forms, depending on the realisation of the `subproblem`.
# Fields
* `inverse_retraction_method` – (`default_inverse_retraction_method(M)`) an inverse retraction method to use within Frank Wolfe.
* `retraction_method` – (`default_retraction_method(M)`) a type of retraction
* `p`, `q`, `r`  – the current iterate, the gradient step and the prox, respetively
  their type is set by intializing `p`
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref) function to run the modified algorithm (experimental)
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
    Pr,
    St,
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
    sub_problem::Pr
    sub_state::St
    X::T
    retraction_method::RTR
    inverse_retraction_method::ITR
    stepsize::S
    stop::SC
    function DifferenceOfConvexProximalState(
        M::AbstractManifold,
        p::P,
        sub_problem::Pr,
        sub_state::St;
        X::T=zero_vector(M, p),
        stepsize::S=ConstantStepsize(M),
        stopping_criterion::SC=StopWhenChangeLess(1e-8),
        inverse_retraction_method::I=default_inverse_retraction_method(M),
        retraction_method::R=default_retraction_method(M),
        λ::Fλ=i -> 1,
    ) where {
        P,
        T,
        Pr,
        St,
        S<:Stepsize,
        SC<:StoppingCriterion,
        I<:AbstractInverseRetractionMethod,
        R<:AbstractRetractionMethod,
        Fλ,
    }
        return new{P,T,Pr,St,S,SC,R,I,Fλ}(
            λ,
            p,
            copy(M, p),
            copy(M, p),
            sub_problem,
            sub_state,
            X,
            retraction_method,
            inverse_retraction_method,
            stepsize,
            stopping_criterion,
        )
    end
end
get_iterate(dcps::DifferenceOfConvexProximalState) = dcps.p
function set_iterate!(dcps::DifferenceOfConvexProximalState, M, p)
    copyto!(M, dcps.p, p)
    return dcps
end
get_gradient(dcs::DifferenceOfConvexProximalState) = dcs.X
function set_gradient!(dcps::DifferenceOfConvexProximalState, M, p, X)
    copyto!(M, dcps.X, p, X)
    return dcps
end

#
# Prox approach
#
@doc raw"""
    difference_of_convex_proximal_point(M, prox_g, grad_h, p; kwargs...)
    difference_of_convex_proximal_point(M, grad_h, p; kwargs...)

Compute the difference of convex proximal point algorithm [^SouzaOliveira2015] to minimize

```math
    \operatorname*{arg\,min}_{p∈\mathcal M} g(p) - h(p)
```

where you have to provide the (sub) gradient ``∂h`` of ``h`` and either
* the proximal map ``\operatorname{prox}_{\lambda g}`` of `g` as a function `prox_g(M, λ, p)` or  `prox_g(M, q, λ, p)`
* the functions `g` and `grad_g` to compute the proximal map using a sub solver
* your own sub-solver, see optional keywods below


This algorithm performs the following steps given a start point `p`= ``p^{(0)}``.
Then repeat for ``k=0,1,\ldots``

1. ``X^{(k)}  ∈ \operatorname{grad} h(p^{(k)})``
2. ``q^{(k)} = \operatorname{retr}_{p^{(k)}}(λ_kX^{(k)})``
3. ``r^{(k)} = \operatorname{prox}_{λ_kg}(q^{(k)})``
4. ``X^{(k)} = \operatorname{retr}^{-1}_{p^{(k)}}(r^{(k)})``
5. Compute a stepsize ``s_k`` and
6. set ``p^{(k+1)} = \operatorname{retr}_{p^{(k)}(s_kX^{(k)})``.

until the `stopping_criterion` is fulfilled.
See [^AlmeidaNetoOliveiraSouza2020] for more details on the modified variant,
where we slightly changed step 4-6, sine here we get the classical proximal point
method for DC functions for ``s_k = 1`` and we can employ linesearches similar to other solvers.

# Optional parameters

* `λ` – ( `i -> 1/2` ) a function returning the sequence of prox parameters λi
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `gradF(M, x)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
* `cost` (`nothing`) provide the cost `f`, e.g. for debug reasonscost to be used within the default `sub_problem`
  use this if you have a more efficient version than using `g` from above.
* `gradient` (`nothing`) – specify ``\operatorname{grad} f``, for debug / analysis or enhancing the `stopping_criterion`
* `g` (`nothing`) – specify the function `g`.
* `grad_g` (`nothing`) – specify the gradient of `g`. If both `g`and `grad_g` are specified, a subsolver is automatically set up.
* `inverse_retraction_method` - (`default_inverse_retraction_method(M)`) an inverse retraction method to use (see step 4).
* `retraction_method` – (`default_retraction_method(M)`) a retraction to use (see step 2)
* `stepsize` – ([`ConstantStepsize`](@ref)`(M)`) specify a [`Stepsize`](@ref) to run the modified algorithm (experimental.)
  functor.
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenChangeLess`](@ref)`(1e-8)`)
    a [`StoppingCriterion`](@ref) for the algorithm – includes a [`StopWhenGradientNormLess`](@ref)`(1e-8)`, when a `gradient` is provided.

While there are several parameters for a sub solver, the easiest is to provide the function `g` and `grad_g`,
such that together with the mandatory function `g` a default cost and gradient can be generated and passed to
a default subsolver. Hence the easiest example call looks like

```
difference_of_convex_proximal_point(M, grad_h, p0; g=g, grad_g=grad_g)
```

# Optional parameters for the sub problem

* `sub_cost` ([`ProximalDCCost`](@ref)`(g, copy(M, p), λ(1))`) cost to be used within the default `sub_problem`
  that is initialized as soon as `g` is provided.
* `sub_grad` ([`ProximalDCGrad`](@ref)`(grad_g, copy(M, p), λ(1); evaluation=evaluation)`
  gradient to be used within the default `sub_problem`, that is initialized as soon as `grad_g` is provided.
  This is generated by default when `grad_g` is provided. You can specify your own by overwriting this keyword.
* `sub_hess` – (a fininte difference approximation by default) specify a Hessian of the subproblem, e.g. to run a trust region algorithm.
  set this to nothing to just use a [`ManifoldGradientObjective`](@ref) in the `subobjective=`
* `sub_kwargs` (`[]`) pass keyword arguments to the `sub_state`, in form of a `Dict(:kwname=>value)`,
  unless you set the `sub_state` directly.
* `sub_objective` (a gradient or hessian objetive based on the last 3 keywords)
  provide the objective used within `sub_problem` (if that is not specified by the user)
* `sub_problem` ([`DefaultManoptProblem`](@ref)`(M, sub_objective)` specify a manopt problem for the sub-solver runs.
  You can also provide a function for a closed form solution. Then `evaluation=` is taken into account for the form of this function.
* `sub_state` ([`TrustRegionState`](@ref) by default, if the hessian is deavtivated (`nothing`) [`GradientDescentState`](@ref); decorated with `sub_kwargs`)
  choose the solver by specifying a solver state to solve the `sub_problem`
* `sub_stopping_criterion` ([`StopAfterIteration`](@ref)`(300) | `[`StopWhenStepsizeLess`](@ref)`(1e-9) | `[`StopWhenGradientNormLess`](@ref)`(1e-9)`)
  a stopping crierion used withing the default `sub_state=`
* `sub_stepsize` ([`ArmijoLinesearch`](@ref)`(M)`) specify a step size used within the `sub_state`

...all others are passed on to decorate the inner [`DifferenceOfConvexProximalState`](@ref).

# Output
the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details

[^SouzaOliveira2015]:
    > de Oliveira Souza, J. C., Oliveira, P. R.:
    > _A proximal point algorithm for {DC} fuctions on Hadamard manifolds_,
    > Journal of Global Optimization (64), 4, 2015, pp. 797–810,
    > doi: [10.1007/s10898-015-0282-7](https://doi.org/10.1007/s10898-015-0282-7).
[^AlmeidaNetoOliveiraSouza2020]:
    > Almeida, Y. T., de Cruz Neto, J. X. Oliveira, P. R., de Oliveira Souza, J. C.:
    > _A modified proximal point method for DC functions on Hadamard manifolds_,
    > Computational Optimization and Applications (76), 2020, pp. 649–673.
    > doi: [10.1007/s10589-020-00173-3](https://doi.org/10.1007/s10589-020-00173-3)
"""
function difference_of_convex_proximal_point(M::AbstractManifold, grad_h, p; kwargs...)
    q = copy(M, p)
    difference_of_convex_proximal_point!(M, grad_h, q; kwargs...)
    return q
end
function difference_of_convex_proximal_point(
    M::AbstractManifold, prox_g, grad_h, p; kwargs...
)
    q = copy(M, p)
    difference_of_convex_proximal_point!(M, prox_g, grad_h, q; kwargs...)
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
    grad_h,
    p;
    g=nothing,
    grad_g=nothing,
    X=zero_vector(M, p),
    λ=i -> 1 / 2,
    evaluation=AllocatingEvaluation(),
    cost=nothing,
    gradient=nothing,
    inverse_retraction_method=default_inverse_retraction_method(M),
    retraction_method=default_retraction_method(M),
    stepsize=ConstantStepsize(M),
    stopping_criterion=if isnothing(gradient)
        StopAfterIteration(300) | StopWhenChangeLess(1e-9)
    else
        StopAfterIteration(300) | StopWhenChangeLess(1e-9) | StopWhenGradientNormLess(1e-9)
    end,
    sub_cost=isnothing(g) ? nothing : ProximalDCCost(g, copy(M, p), λ(1)),
    sub_grad=if isnothing(grad_g)
        nothing
    else
        ProximalDCGrad(grad_g, copy(M, p), λ(1); evaluation=evaluation)
    end,
    sub_hess=ApproxHessianFiniteDifference(M, copy(M, p), sub_grad; evaluation=evaluation),
    sub_kwargs=[],
    sub_stopping_criterion=StopAfterIteration(300) | StopWhenGradientNormLess(1e-8),
    sub_stepsize=ArmijoLinesearch(M),
    sub_objective=if isnothing(sub_cost) || isnothing(sub_grad)
        nothing
    else
        if isnothing(sub_hess)
            ManifoldGradientObjective(sub_cost, sub_grad; evaluation=evaluation)
        else
            ManifoldHessianObjective(sub_cost, sub_grad, sub_hess; evaluation=evaluation)
        end
    end,
    sub_problem::Union{AbstractManoptProblem,Function,Nothing}=if isnothing(sub_objective)
        nothing
    else
        DefaultManoptProblem(M, sub_objective)
    end,
    sub_state::AbstractManoptSolverState=decorate_state!(
        if isnothing(sub_hess)
            GradientDescentState(
                M,
                copy(M, p);
                stepsize=sub_stepsize,
                stopping_criterion=sub_stopping_criterion,
            )
        else
            TrustRegionsState(M, copy(M, p); stopping_criterion=sub_stopping_criterion)
        end,
        sub_kwargs...,
    ),
    kwargs...,
)
    # Check whether either the right defaults were provided or a sub_problen.
    if isnothing(sub_problem)
        error(
            """
            The `sub_problem` is not correctly intialized.
            Please provide _either_
            * a `sub_problem=` to be solved
            * a `sub_objective` to automatically generate the sub problem,
            * `sub_cost=` and `sub_grad=` to automatically generate the sub objective _or_
            * `g=` and `grad_g=` keywords to automatically generate the sub cost and gradient.
            """,
        )
    end
    mdcpo = ManifoldDifferenceOfConvexProximalObjective(
        grad_h; cost=cost, gradient=gradient, evaluation=evaluation
    )
    dmdcpo = decorate_objective!(M, mdcpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmdcpo)
    dcps = DifferenceOfConvexProximalState(
        M,
        p,
        sub_problem,
        sub_state;
        X=X,
        stepsize=stepsize,
        stopping_criterion=stopping_criterion,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        λ=λ,
    )
    ddcps = decorate_state!(dcps; kwargs...)
    return get_solver_return(solve!(dmp, ddcps))
end
function difference_of_convex_proximal_point!(
    M,
    prox_g,
    grad_h,
    p;
    X=zero_vector(M, p),
    λ=i -> 1 / 2,
    evaluation=AllocatingEvaluation(),
    cost=nothing,
    gradient=nothing,
    inverse_retraction_method=default_inverse_retraction_method(M),
    retraction_method=default_retraction_method(M),
    stepsize=ConstantStepsize(M),
    stopping_criterion=if isnothing(gradient)
        StopAfterIteration(300) | StopWhenChangeLess(1e-9)
    else
        StopAfterIteration(300) | StopWhenChangeLess(1e-9) | StopWhenGradientNormLess(1e-9)
    end,
    kwargs...,
)
    mdcpo = ManifoldDifferenceOfConvexProximalObjective(
        grad_h; cost=cost, gradient=gradient, evaluation=evaluation
    )
    dmdcpo = decorate_objective!(M, mdcpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmdcpo)
    dcps = DifferenceOfConvexProximalState(
        M,
        p,
        prox_g,
        evaluation;
        X=X,
        stepsize=stepsize,
        stopping_criterion=stopping_criterion,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        λ=λ,
    )
    ddcps = decorate_state!(dcps; kwargs...)
    return get_solver_return(solve!(dmp, ddcps))
end

function initialize_solver!(::AbstractManoptProblem, dcps::DifferenceOfConvexProximalState)
    return dcps
end
#=
    Varant I: Allocating closed form of the prox
=#
function step_solver!(
    amp::AbstractManoptProblem,
    dcps::DifferenceOfConvexProximalState{P,T,<:Function,AllocatingEvaluation},
    i,
) where {P,T}
    M = get_manifold(amp)
    # each line is one step in the documented solver steps. Note that we can reuse dcps.X
    get_subtrahend_gradient!(amp, dcps.X, dcps.p)
    retract!(M, dcps.q, dcps.p, dcps.λ(i) * dcps.X, dcps.retraction_method)
    copyto!(M, dcps.r, dcps.sub_problem(M, dcps.λ(i), dcps.q))
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    s = dcps.stepsize(amp, dcps, i)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    return dcps
end

#=
    Varant II: In-Place closed form of the prox
=#

function step_solver!(
    amp::AbstractManoptProblem,
    dcps::DifferenceOfConvexProximalState{P,T,<:Function,InplaceEvaluation},
    i,
) where {P,T}
    M = get_manifold(amp)
    # each line is one step in the documented solver steps. Note that we can reuse dcps.X
    get_subtrahend_gradient!(amp, dcps.X, dcps.p)
    retract!(M, dcps.q, dcps.p, dcps.λ(i) * dcps.X, dcps.retraction_method)
    dcps.sub_problem(M, dcps.r, dcps.λ(i), dcps.q)
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    s = dcps.stepsize(amp, dcps, i)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    return dcps
end
#=
    Varant III: Subsolver variant of the prox
=#
function step_solver!(
    amp::AbstractManoptProblem,
    dcps::DifferenceOfConvexProximalState{
        P,T,<:AbstractManoptProblem,<:AbstractManoptSolverState
    },
    i,
) where {P,T}
    M = get_manifold(amp)
    # Evaluate gradient of h into X
    get_subtrahend_gradient!(amp, dcps.X, dcps.p)
    # do a step in that direction
    retract!(M, dcps.q, dcps.p, dcps.λ(i) * dcps.X, dcps.retraction_method)
    # use this point (q) for the prox
    set_manopt_parameter!(dcps.sub_problem, :Cost, :p, dcps.q)
    set_manopt_parameter!(dcps.sub_problem, :Cost, :λ, dcps.λ(i))
    set_manopt_parameter!(dcps.sub_problem, :Gradient, :p, dcps.q)
    set_manopt_parameter!(dcps.sub_problem, :Gradient, :λ, dcps.λ(i))
    set_iterate!(dcps.sub_state, M, copy(M, dcps.q))
    solve!(dcps.sub_problem, dcps.sub_state)
    copyto!(M, dcps.r, get_solver_result(dcps.sub_state))
    # use that direction
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    # to determine a step size
    s = dcps.stepsize(amp, dcps, i)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    if !isnothing(get_gradient_function(get_objective(amp)))
        get_gradient!(amp, dcps.X, dcps.p)
    end
    return dcps
end
