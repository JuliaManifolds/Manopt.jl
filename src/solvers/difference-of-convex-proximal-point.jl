
@doc """
    DifferenceOfConvexProximalState{P, T, Pr, St, S<:Stepsize, SC<:StoppingCriterion, RTR<:AbstractRetractionMethod, ITR<:AbstractInverseRetractionMethod}
        <: AbstractSubProblemSolverState

A struct to store the current state of the algorithm as well as the form.
It comes in two forms, depending on the realisation of the `subproblem`.

# Fields

* $(_field_inv_retr)
* $(_field_retr)
* `p`, `q`, `r`:               the current iterate, the gradient step and the prox, respectively
  their type is set by initializing `p`
* $(_field_step)
* $(_field_stop)
* `X`, `Y`: the current gradient and descent direction, respectively
  their common type is set by the keyword `X`
* `sub_problem`:               an [`AbstractManoptProblem`](@ref) problem or a function `(M, p, X) -> q` or `(M, q, p, X)` for the a closed form solution of the sub problem
* `sub_state`:                 an [`AbstractManoptSolverState`](@ref) for the subsolver or an [`AbstractEvaluationType`](@ref) in case the sub problem is provided as a function

# Constructor

    DifferenceOfConvexProximalState(M::AbstractManifold, sub_problem, sub_state; kwargs...)

construct an difference of convex proximal point state

    DifferenceOfConvexProximalState(M::AbstractManifold, sub_problem;
        evaluation=AllocatingEvaluation(), kwargs...
)

construct an difference of convex proximal point state, where `sub_problem` is a closed form solution with `evaluation` as type of evaluation.

## Input

$_arg_M
$_arg_sub_problem
$_arg_sub_state

# Keyword arguments

* $(_kw_inverse_retraction_method_default): $(_kw_inverse_retraction_method)
* $(_kw_p_default): $(_kw_p)
* $(_kw_retraction_method_default): $(_kw_retraction_method)
* `stepsize=`[`ConstantStepsize`](@ref)`(M)`: $(_kw_stepsize)
* `stopping_criterion=`[StopWhenChangeLess`](@ref)`(1e-8)`: $(_kw_stopping_criterion)
* $(_kw_X_default): $(_kw_X)
"""
mutable struct DifferenceOfConvexProximalState{
    P,
    T,
    Pr,
    St<:AbstractManoptSolverState,
    S<:Stepsize,
    SC<:StoppingCriterion,
    RTR<:AbstractRetractionMethod,
    ITR<:AbstractInverseRetractionMethod,
    Tλ,
} <: AbstractSubProblemSolverState
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
        sub_problem::Pr,
        sub_state::St;
        p::P=rand(M),
        X::T=zero_vector(M, p),
        stepsize::S=ConstantStepsize(M),
        stopping_criterion::SC=StopWhenChangeLess(M, 1e-8),
        inverse_retraction_method::I=default_inverse_retraction_method(M),
        retraction_method::R=default_retraction_method(M, typeof(p)),
        λ::Fλ=i -> 1,
    ) where {
        P,
        T,
        Pr<:Union{AbstractManoptProblem,F} where {F},
        S<:Stepsize,
        St<:AbstractManoptSolverState,
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
function DifferenceOfConvexProximalState(
    M::AbstractManifold, sub_problem; evaluation::E=AllocatingEvaluation(), kwargs...
) where {E<:AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation=evaluation)
    return DifferenceOfConvexProximalState(M, sub_problem, cfs; kwargs...)
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
function get_message(dcs::DifferenceOfConvexProximalState)
    # for now only the sub solver might have messages
    return get_message(dcs.sub_state)
end
function show(io::IO, dcps::DifferenceOfConvexProximalState)
    i = get_count(dcps, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(dcps.stop) ? "Yes" : "No"
    sub = repr(dcps.sub_state)
    sub = replace(sub, "\n" => "\n    | ")
    s = """
    # Solver state for `Manopt.jl`s Difference of Convex Proximal Point Algorithm
    $Iter
    ## Parameters
    * retraction method:         $(dcps.retraction_method)
    * inverse retraction method: $(dcps.inverse_retraction_method)
    * sub solver state:
        | $(sub)

    ## Stepsize
    $(dcps.stepsize)

    ## Stopping criterion

    $(status_summary(dcps.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
#
# Prox approach
#
_doc_DCPPA = """
    difference_of_convex_proximal_point(M, grad_h, p=rand(M); kwargs...)
    difference_of_convex_proximal_point(M, mdcpo, p=rand(M); kwargs...)
    difference_of_convex_proximal_point!(M, grad_h, p; kwargs...)
    difference_of_convex_proximal_point!(M, mdcpo, p; kwargs...)

Compute the difference of convex proximal point algorithm [SouzaOliveira:2015](@cite) to minimize

```math
    $(_l_argmin)_{p∈$(_math(:M))} g(p) - h(p)
```

where you have to provide the subgradient ``∂h`` of ``h`` and either
* the proximal map ``$(_l_prox)_{λg}`` of `g` as a function `prox_g(M, λ, p)` or  `prox_g(M, q, λ, p)`
* the functions `g` and `grad_g` to compute the proximal map using a sub solver
* your own sub-solver, specified by `sub_problem=`and `sub_state=`

This algorithm performs the following steps given a start point `p`= ``p^{(0)}``.
Then repeat for ``k=0,1,…``

1. ``X^{(k)}  ∈ $(_l_grad) h(p^{(k)})``
2. ``q^{(k)} = $(_l_retr)_{p^{(k)}}(λ_kX^{(k)})``
3. ``r^{(k)} = $(_l_prox)_{λ_kg}(q^{(k)})``
4. ``X^{(k)} = $(_l_retr)^{-1}_{p^{(k)}}(r^{(k)})``
5. Compute a stepsize ``s_k`` and
6. set ``p^{(k+1)} = $(_l_retr)_{p^{(k)}}(s_kX^{(k)})``.

until the `stopping_criterion` is fulfilled.

See [AlmeidaNetoOliveiraSouza:2020](@cite) for more details on the modified variant,
where steps 4-6 are slightly changed, since here the classical proximal point method for
DC functions is obtained for ``s_k = 1`` and one can hence employ usual line search method.


# Keyword arguments

* `λ`:                          ( `k -> 1/2` ) a function returning the sequence of prox parameters ``λ_k``
* `cost=nothing`: provide the cost `f`, for debug reasons / analysis
* $(_kw_evaluation_default): $(_kw_evaluation)
* `gradient=nothing`: specify ``$(_l_grad) f``, for debug / analysis
   or enhancing the `stopping_criterion`
* `prox_g=nothing`: specify a proximal map for the sub problem _or_ both of the following
* `g=nothing`: specify the function `g`.
* `grad_g=nothing`: specify the gradient of `g`. If both `g`and `grad_g` are specified, a subsolver is automatically set up.
* $(_kw_inverse_retraction_method_default); $(_kw_inverse_retraction_method)
* $(_kw_retraction_method_default); $(_kw_retraction_method)
* `stepsize=`[`ConstantStepsize`](@ref)`(M)`): $(_kw_stepsize)
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(200)`$(_sc_any)[`StopWhenChangeLess`](@ref)`(1e-8)`):
  $(_kw_stopping_criterion)
  A [`StopWhenGradientNormLess`](@ref)`(1e-8)` is added with $(_sc_any), when a `gradient` is provided.
* `sub_cost=`[`ProximalDCCost`](@ref)`(g, copy(M, p), λ(1))`):
  cost to be used within the default `sub_problem` that is initialized as soon as `g` is provided.
  $(_kw_used_in("sub_objective"))
* `sub_grad=`[`ProximalDCGrad`](@ref)`(grad_g, copy(M, p), λ(1); evaluation=evaluation)`:
  gradient to be used within the default `sub_problem`, that is initialized as soon as `grad_g` is provided.
  $(_kw_used_in("sub_objective"))
* `sub_hess`:              (a finite difference approximation using `sub_grad` by default):
   specify a Hessian of the `sub_cost`, which the default solver, see `sub_state=` needs.
* $(_kw_sub_kwargs_default): $(_kw_sub_kwargs)
* `sub_objective`:         a gradient or Hessian objective based on `sub_cost=`, `sub_grad=`, and `sub_hess`if provided
   the objective used within `sub_problem`.
  $(_kw_used_in("sub_problem"))
* `sub_problem=`[`DefaultManoptProblem`](@ref)`(M, sub_objective)`:
  specify a manopt problem or a function for the sub-solver runs.
  You can also provide a function for a closed form solution. Then `evaluation=` is taken into account for the form of this function.
* `sub_state`([`GradientDescentState`](@ref) or [`TrustRegionsState`](@ref) if `sub_hessian`):
  the subsolver to be used when solving the sub problem.
  By default this is also decorated using the `sub_kwargs`.
  if the `sub_problem` if a function (a closed form solution), this is set to `evaluation`
  and can be changed to the evaluation type of the closed form solution accordingly.
* `sub_stopping_criterion`: ([`StopAfterIteration`](@ref)`(300)`$(_sc_any)`[`StopWhenGradientNormLess`](@ref)`(1e-8)`:
  a stopping criterion used withing the default `sub_state=`
  $(_kw_used_in("sub_state"))

$(_kw_others)

$(_doc_sec_output)
"""

@doc "$(_doc_DCPPA)"
difference_of_convex_proximal_point(M::AbstractManifold, args...; kwargs...)
function difference_of_convex_proximal_point(
    M::AbstractManifold,
    grad_h,
    p=rand(M);
    cost=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    gradient=nothing,
    g=nothing,
    grad_g=nothing,
    prox_g=nothing,
    kwargs...,
)
    p_ = _ensure_mutating_variable(p)
    cost_ = _ensure_mutating_cost(cost, p)
    grad_h_ = _ensure_mutating_gradient(grad_h, p, evaluation)
    g_ = _ensure_mutating_cost(g, p)
    gradient_ = _ensure_mutating_gradient(gradient, p, evaluation)
    grad_g_ = _ensure_mutating_gradient(grad_g, p, evaluation)
    prox_g_ = _ensure_mutating_prox(prox_g, p, evaluation)

    mdcpo = ManifoldDifferenceOfConvexProximalObjective(
        grad_h_; cost=cost_, gradient=gradient_, evaluation=evaluation
    )
    rs = difference_of_convex_proximal_point(
        M,
        mdcpo,
        p_;
        cost=cost_,
        evaluation=evaluation,
        gradient=gradient_,
        g=g_,
        grad_g=grad_g_,
        prox_g=prox_g_,
        kwargs...,
    )
    return _ensure_matching_output(p, rs)
end

function difference_of_convex_proximal_point(
    M::AbstractManifold, mdcpo::O, p; kwargs...
) where {
    O<:Union{ManifoldDifferenceOfConvexProximalObjective,AbstractDecoratedManifoldObjective}
}
    q = copy(M, p)
    return difference_of_convex_proximal_point!(M, mdcpo, q; kwargs...)
end

@doc "$(_doc_DCPPA)"
difference_of_convex_proximal_point!(M::AbstractManifold, args...; kwargs...)
function difference_of_convex_proximal_point!(
    M::AbstractManifold,
    grad_h,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    cost=nothing,
    gradient=nothing,
    kwargs...,
)
    mdcpo = ManifoldDifferenceOfConvexProximalObjective(
        grad_h; cost=cost, gradient=gradient, evaluation=evaluation
    )
    return difference_of_convex_proximal_point!(
        M, mdcpo, p; evaluation=evaluation, kwargs...
    )
end
function difference_of_convex_proximal_point!(
    M::AbstractManifold,
    mdcpo::O,
    p;
    g=nothing,
    grad_g=nothing,
    prox_g=nothing,
    X=zero_vector(M, p),
    λ=i -> 1 / 2,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method=default_inverse_retraction_method(M),
    objective_type=:Riemannian,
    retraction_method=default_retraction_method(M, typeof(p)),
    stepsize=ConstantStepsize(M),
    stopping_criterion=if isnothing(get_gradient_function(mdcpo))
        StopAfterIteration(300) | StopWhenChangeLess(M, 1e-9)
    else
        StopAfterIteration(300) |
        StopWhenChangeLess(M, 1e-9) |
        StopWhenGradientNormLess(1e-9)
    end,
    sub_cost=isnothing(g) ? nothing : ProximalDCCost(g, copy(M, p), λ(1)),
    sub_grad=if isnothing(grad_g)
        nothing
    else
        ProximalDCGrad(grad_g, copy(M, p), λ(1); evaluation=evaluation)
    end,
    sub_hess=ApproxHessianFiniteDifference(M, copy(M, p), sub_grad; evaluation=evaluation),
    sub_kwargs=(;),
    sub_stopping_criterion=StopAfterIteration(300) | StopWhenGradientNormLess(1e-8),
    sub_objective=if isnothing(sub_cost) || isnothing(sub_grad)
        nothing
    else
        decorate_objective!(
            M,
            if isnothing(sub_hess)
                ManifoldGradientObjective(sub_cost, sub_grad; evaluation=evaluation)
            else
                ManifoldHessianObjective(
                    sub_cost, sub_grad, sub_hess; evaluation=evaluation
                )
            end;
            objective_type=objective_type,
            sub_kwargs...,
        )
    end,
    sub_problem::Union{AbstractManoptProblem,Function,Nothing}=if !isnothing(prox_g)
        prox_g # closed form solution
    else
        if isnothing(sub_objective)
            nothing
        else
            DefaultManoptProblem(M, sub_objective)
        end
    end,
    sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState,Nothing}=if !isnothing(
        prox_g
    )
        maybe_wrap_evaluation_type(evaluation)
    elseif isnothing(sub_objective)
        nothing
    else
        decorate_state!(
            if isnothing(sub_hess)
                GradientDescentState(
                    M;
                    p=copy(M, p),
                    stopping_criterion=sub_stopping_criterion,
                    sub_kwargs...,
                )
            else
                TrustRegionsState(
                    M,
                    DefaultManoptProblem(
                        TangentSpace(M, copy(M, p)),
                        TrustRegionModelObjective(sub_objective),
                    ),
                    TruncatedConjugateGradientState(TangentSpace(M, p); sub_kwargs...);
                    p=copy(M, p),
                )
            end;
            sub_kwargs...,
        )
    end,
    kwargs...,
) where {
    O<:Union{ManifoldDifferenceOfConvexProximalObjective,AbstractDecoratedManifoldObjective}
}
    # Check whether either the right defaults were provided or a `sub_problem`.
    if isnothing(sub_problem)
        error(
            """
            The `sub_problem` is not correctly initialized. Provie _one of_ the following setups
            * `prox_g` as a closed form solution,
            * `g=` and `grad_g=` keywords to automatically generate the sub cost and gradient,
            * provide individual `sub_cost=` and `sub_grad=` to automatically generate the sub objective,
            * provide a `sub_objective`, _or_
            * provide a `sub_problem=` (consider maybe specifying `sub_state=` to specify the solver)
            """,
        )
    end
    dmdcpo = decorate_objective!(M, mdcpo; objective_type=objective_type, kwargs...)
    dmp = DefaultManoptProblem(M, dmdcpo)
    dcps = DifferenceOfConvexProximalState(
        M,
        sub_problem,
        maybe_wrap_evaluation_type(sub_state);
        p=p,
        X=X,
        stepsize=stepsize,
        stopping_criterion=stopping_criterion,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        λ=λ,
    )
    ddcps = decorate_state!(dcps; kwargs...)
    solve!(dmp, ddcps)
    return get_solver_return(get_objective(dmp), ddcps)
end

function initialize_solver!(::AbstractManoptProblem, dcps::DifferenceOfConvexProximalState)
    return dcps
end
#=
    Variant I: allocating closed form of the prox
=#
function step_solver!(
    amp::AbstractManoptProblem,
    dcps::DifferenceOfConvexProximalState{
        P,T,<:Function,ClosedFormSubSolverState{AllocatingEvaluation}
    },
    k,
) where {P,T}
    M = get_manifold(amp)
    # each line is one step in the documented solver steps. Note the reuse of `dcps.X`
    get_subtrahend_gradient!(amp, dcps.X, dcps.p)
    retract!(M, dcps.q, dcps.p, dcps.λ(k) * dcps.X, dcps.retraction_method)
    copyto!(M, dcps.r, dcps.sub_problem(M, dcps.λ(k), dcps.q))
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    s = dcps.stepsize(amp, dcps, k)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    return dcps
end

#=
    Variant II: in-place closed form of the prox
=#
function step_solver!(
    amp::AbstractManoptProblem,
    dcps::DifferenceOfConvexProximalState{
        P,T,<:Function,ClosedFormSubSolverState{InplaceEvaluation}
    },
    k,
) where {P,T}
    M = get_manifold(amp)
    # each line is one step in the documented solver steps. Note the reuse of `dcps.X`
    get_subtrahend_gradient!(amp, dcps.X, dcps.p)
    retract!(M, dcps.q, dcps.p, dcps.λ(k) * dcps.X, dcps.retraction_method)
    dcps.sub_problem(M, dcps.r, dcps.λ(k), dcps.q)
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    s = dcps.stepsize(amp, dcps, k)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    return dcps
end
#=
    Variant III: subsolver variant of the prox
=#
function step_solver!(
    amp::AbstractManoptProblem,
    dcps::DifferenceOfConvexProximalState{
        P,T,<:AbstractManoptProblem,<:AbstractManoptSolverState
    },
    k,
) where {P,T}
    M = get_manifold(amp)
    # Evaluate gradient of h into X
    get_subtrahend_gradient!(amp, dcps.X, dcps.p)
    # do a step in that direction
    retract!(M, dcps.q, dcps.p, dcps.λ(k) * dcps.X, dcps.retraction_method)
    # use this point (q) for the proximal map
    set_parameter!(dcps.sub_problem, :Objective, :Cost, :p, dcps.q)
    set_parameter!(dcps.sub_problem, :Objective, :Cost, :λ, dcps.λ(k))
    set_parameter!(dcps.sub_problem, :Objective, :Gradient, :p, dcps.q)
    set_parameter!(dcps.sub_problem, :Objective, :Gradient, :λ, dcps.λ(k))
    set_iterate!(dcps.sub_state, M, copy(M, dcps.q))
    solve!(dcps.sub_problem, dcps.sub_state)
    copyto!(M, dcps.r, get_solver_result(dcps.sub_state))
    # use that direction
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    # to determine a step size
    s = dcps.stepsize(amp, dcps, k)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    if !isnothing(get_gradient_function(get_objective(amp)))
        get_gradient!(amp, dcps.X, dcps.p)
    end
    return dcps
end
