@doc raw"""
    ExactPenaltyMethodState{P,T} <: AbstractManoptSolverState

Describes the exact penalty method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `p` – a set point on a manifold as starting point
* `sub_problem` – an [`AbstractManoptProblem`](@ref) problem for the subsolver
* `sub_state` – an [`AbstractManoptSolverState`](@ref) for the subsolver
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `u` – (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_min` – (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ` – (`1.0`) the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min), `[`StopWhenChangeLess`](@ref)`(min_stepsize)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.


# Constructor

    ExactPenaltyMethodState(M::AbstractManifold, p, sub_problem, sub_state; kwargs...)

construct an exact penalty options with the fields and defaults as above, where the
manifold `M` is used for defaults in the keyword
arguments.

# See also

[`exact_penalty_method`](@ref)
"""
mutable struct ExactPenaltyMethodState{P,Pr,Op,TStopping<:StoppingCriterion} <:
               AbstractSubProblemSolverState
    p::P
    sub_problem::Pr
    sub_state::Op
    ϵ::Real
    ϵ_min::Real
    u::Real
    u_min::Real
    ρ::Real
    θ_ρ::Real
    θ_u::Real
    θ_ϵ::Real
    stop::TStopping
    function ExactPenaltyMethodState(
        ::AbstractManifold,
        p::P,
        sub_problem::Pr,
        sub_state::Op;
        ϵ::Real=1e-3,
        ϵ_min::Real=1e-6,
        ϵ_exponent=1 / 100,
        θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
        u::Real=1e-1,
        u_min::Real=1e-6,
        u_exponent=1 / 100,
        θ_u=(u_min / u)^(u_exponent),
        ρ::Real=1.0,
        θ_ρ::Real=0.3,
        stopping_criterion::StoppingCriterion=StopWhenAny(
            StopAfterIteration(300),
            StopWhenAll(StopWhenSmallerOrEqual(:ϵ, ϵ_min), StopWhenChangeLess(1e-10)),
        ),
    ) where {P,Pr<:AbstractManoptProblem,Op<:AbstractManoptSolverState}
        epms = new{P,Pr,Op,typeof(stopping_criterion)}()
        epms.p = p
        epms.sub_problem = sub_problem
        epms.sub_state = sub_state
        epms.ϵ = ϵ
        epms.ϵ_min = ϵ_min
        epms.u = u
        epms.u_min = u_min
        epms.ρ = ρ
        epms.θ_ρ = θ_ρ
        epms.θ_u = θ_u
        epms.θ_ϵ = θ_ϵ
        epms.stop = stopping_criterion
        return epms
    end
end
get_iterate(epms::ExactPenaltyMethodState) = epms.p
function get_message(epms::ExactPenaltyMethodState)
    # for now only the sub solver might have messages
    return get_message(epms.sub_state)
end
function set_iterate!(epms::ExactPenaltyMethodState, M, p)
    epms.p = p
    return epms
end
function show(io::IO, epms::ExactPenaltyMethodState)
    i = get_count(epms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(epms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Exact Penalty Method
    $Iter
    ## Parameters
    * ϵ: $(epms.ϵ) (ϵ_min: $(epms.ϵ_min), θ_ϵ: $(epms.θ_ϵ))
    * u: $(epms.u) (ϵ_min: $(epms.u_min), θ_u: $(epms.θ_u))
    * ρ: $(epms.ρ) (θ_ρ: $(epms.θ_ρ))

    ## Stopping Criterion
    $(status_summary(epms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    exact_penalty_method(M, F, gradF, p=rand(M); kwargs...)
    exact_penalty_method(M, cmo::ConstrainedManifoldObjective, p=rand(M); kwargs...)

perform the exact penalty method (EPM)[^LiuBoumal2020].
The aim of the EPM is to find a solution of the constrained optimisation task

```math
\begin{aligned}
\min_{p ∈\mathcal{M}} &f(p)\\
\text{subject to } &g_i(p)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(p)=0 \quad  \text{ for } j=1,…,n,
\end{aligned}
```

where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^n`` are twice continuously differentiable functions from `M` to ℝ.
For that a weighted ``L_1``-penalty term for the violation of the constraints is added to the objective

```math
f(x) + ρ (\sum_{i=1}^m \max\left\{0, g_i(x)\right\} + \sum_{j=1}^n \vert h_j(x)\vert),
```

where ``ρ>0`` is the penalty parameter.
Since this is non-smooth, a [`SmoothingTechnique`](@ref) with parameter `u` is applied,
see the [`ExactPenaltyCost`](@ref).

In every step ``k`` of the exact penalty method, the smoothed objective is then minimized over all
``x ∈\mathcal{M}``.
Then, the accuracy tolerance ``ϵ`` and the smoothing parameter ``u`` are updated by setting

```math
ϵ^{(k)}=\max\{ϵ_{\min}, θ_ϵ ϵ^{(k-1)}\},
```

where ``ϵ_{\min}`` is the lowest value ``ϵ`` is allowed to become and ``θ_ϵ ∈ (0,1)`` is constant scaling factor, and

```math
u^{(k)} = \max \{u_{\min}, \theta_u u^{(k-1)} \},
```

where ``u_{\min}`` is the lowest value ``u`` is allowed to become and ``θ_u ∈ (0,1)`` is constant scaling factor.

Last, we update the penalty parameter ``ρ`` according to

```math
ρ^{(k)} = \begin{cases}
ρ^{(k-1)}/θ_ρ,  & \text{if } \displaystyle \max_{j \in \mathcal{E},i \in \mathcal{I}} \Bigl\{ \vert h_j(x^{(k)}) \vert, g_i(x^{(k)})\Bigr\} \geq u^{(k-1)} \Bigr) ,\\
ρ^{(k-1)}, & \text{else,}
\end{cases}
```

where ``θ_ρ \in (0,1)`` is a constant scaling factor.


[^LiuBoumal2020]:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > In: Applied Mathematics & Optimization, vol 82, 949–981 (2020),
    > doi [10.1007/s00245-019-09564-3](https://doi.org/10.1007/s00245-019-09564-3),
    > arXiv: [1901.10000](https://arxiv.org/abs/1901.10000).
    > Matlab source: [https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints](https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints)

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `grad_f` – the gradient of the cost function

# Optional (if not called with the [`ConstrainedManifoldObjective`](@ref) `cmo`)

* `g` – (`nothing`) the inequality constraints
* `h` – (`nothing`) the equality constraints
* `grad_g` – (`nothing`) the gradient of the inequality constraints
* `grad_h` – (`nothing`) the gradient of the equality constraints

Note that one of the pairs (`g`, `grad_g`) or (`h`, `grad_h`) has to be proviede.
Otherwise the problem is not constrained and you can also call e.g. [`quasi_Newton`](@ref)

# Optional

* `smoothing` – ([`LogarithmicSumOfExponentials`](@ref)) [`SmoothingTechnique`](@ref) to use
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_exponent` – (`1/100`) exponent of the ϵ update factor;
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `u` – (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_exponent` – (`1/100`) exponent of the u update factor;
* `u_min` – (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ` – (`1.0`) the penalty parameter
* `min_stepsize` – (`1e-10`) the minimal step size
* `sub_cost` – ([`ExactPenaltyCost`](@ref)`(problem, ρ, u; smoothing=smoothing)`) use this exact penality cost, expecially with the same numbers `ρ,u` as in the options for the sub problem
* `sub_grad` – ([`ExactPenaltyGrad`](@ref)`(problem, ρ, u; smoothing=smoothing)`) use this exact penality gradient, expecially with the same numbers `ρ,u` as in the options for the sub problem
* `sub_kwargs` – keyword arguments to decorate the sub options, e.g. with debug.
* `sub_stopping_criterion` – ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenGradientNormLess`](@ref)`(ϵ) | `[`StopWhenStepsizeLess`](@ref)`(1e-10)`) specify a stopping criterion for the subsolver.
* `sub_problem` – ([`DefaultManoptProblem`](@ref)`(M, `[`ManifoldGradientObjective`](@ref)`(sub_cost, sub_grad; evaluation=evaluation)` – ` problem for the subsolver
* `sub_state` – ([`QuasiNewtonState`](@ref)) using [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) with [`InverseBFGS`](@ref) and `sub_stopping_criterion` as a stopping criterion. See also `sub_kwargs`.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(300)` | ([`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min)` & [`StopWhenChangeLess`](@ref)`(1e-10)`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
exact_penalty_method(M::AbstractManifold, args...; kwargs...)
function exact_penalty_method(M::AbstractManifold, f, grad_f; kwargs...)
    return exact_penalty_method(M, f, grad_f, rand(M); kwargs...)
end
function exact_penalty_method(
    M::AbstractManifold,
    f::TF,
    grad_f::TGF,
    p;
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TGF}
    cmo = ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h; evaluation=evaluation
    )
    return exact_penalty_method(M, cmo, p; evaluation=evaluation, kwargs...)
end
function exact_penalty_method(
    M::AbstractManifold,
    f,
    grad_f,
    p::Number;
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    q = [p]
    f_(M, p) = f(M, p[])
    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    g_ = isnothing(g) ? nothing : (M, p) -> g(M, p[])
    grad_g_ = isnothing(grad_g) ? nothing : _to_mutating_gradient(grad_g, evaluation)
    h_ = isnothing(h) ? nothing : (M, p) -> h(M, p[])
    grad_h_ = isnothing(grad_h) ? nothing : _to_mutating_gradient(grad_h, evaluation)
    cmo = ConstrainedManifoldObjective(
        f_, grad_f_, g_, grad_g_, h_, grad_h_; evaluation=evaluation
    )
    rs = exact_penalty_method(M, cmo, q; evaluation=evaluation, kwargs...)
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function exact_penalty_method(
    M::AbstractManifold, cmo::ConstrainedManifoldObjective, p=rand(M); kwargs...
)
    q = copy(M, p)
    return exact_penalty_method!(M, cmo, q; kwargs...)
end

@doc raw"""
    exact_penalty_method!(M, f, grad_f, p; kwargs...)
    exact_penalty_method!(M, cmo::ConstrainedManifoldObjective, p; kwargs...)

perform the exact penalty method (EPM)[^LiuBoumal2020] in place of `p`.

For all options, see [`exact_penalty_method`](@ref).
"""
exact_penalty_method!(M::AbstractManifold, args...; kwargs...)
function exact_penalty_method!(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    cmo = ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h; evaluation=evaluation
    )
    return exact_penalty_method!(M, cmo, p; evaluation=evaluation, kwargs...)
end
function exact_penalty_method!(
    M::AbstractManifold,
    cmo::ConstrainedManifoldObjective,
    p;
    evaluation=AllocatingEvaluation(),
    ϵ::Real=1e-3,
    ϵ_min::Real=1e-6,
    ϵ_exponent=1 / 100,
    θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
    u::Real=1e-1,
    u_min::Real=1e-6,
    u_exponent=1 / 100,
    θ_u=(u_min / u)^(u_exponent),
    ρ::Real=1.0,
    θ_ρ::Real=0.3,
    smoothing=LogarithmicSumOfExponentials(),
    sub_cost=ExactPenaltyCost(cmo, ρ, u; smoothing=smoothing),
    sub_grad=ExactPenaltyGrad(cmo, ρ, u; smoothing=smoothing),
    sub_problem::AbstractManoptProblem=DefaultManoptProblem(
        M, ManifoldGradientObjective(sub_cost, sub_grad; evaluation=evaluation)
    ),
    sub_kwargs=[],
    sub_stopping_criterion=StopAfterIteration(300) |
                           StopWhenGradientNormLess(ϵ) |
                           StopWhenStepsizeLess(1e-8),
    sub_state::AbstractManoptSolverState=decorate_state!(
        QuasiNewtonState(
            M,
            copy(M, p);
            initial_vector=zero_vector(M, p),
            direction_update=QuasiNewtonLimitedMemoryDirectionUpdate(
                M, copy(M, p), InverseBFGS(), 30
            ),
            stopping_criterion=sub_stopping_criterion,
            stepsize=default_stepsize(M, QuasiNewtonState),
        ),
        sub_kwargs...,
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
        StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(1e-10)
    ),
    kwargs...,
)
    emps = ExactPenaltyMethodState(
        M,
        p,
        sub_problem,
        sub_state;
        ϵ=ϵ,
        ϵ_min=ϵ_min,
        u=u,
        u_min=u_min,
        ρ=ρ,
        θ_ρ=θ_ρ,
        θ_ϵ=θ_ϵ,
        θ_u=θ_u,
        stopping_criterion=stopping_criterion,
    )
    deco_o = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, deco_o)
    epms = decorate_state!(emps; kwargs...)
    return get_solver_return(solve!(dmp, epms))
end
#
# Solver functions
#
function initialize_solver!(::AbstractManoptProblem, epms::ExactPenaltyMethodState)
    return epms
end
function step_solver!(
    amp::AbstractManoptProblem, epms::ExactPenaltyMethodState{P,<:AbstractManoptProblem}, i
) where {P}
    M = get_manifold(amp)
    # use subsolver to minimize the smoothed penalized function
    set_manopt_parameter!(epms.sub_problem, :Cost, :ρ, epms.ρ)
    set_manopt_parameter!(epms.sub_problem, :Cost, :u, epms.u)
    set_manopt_parameter!(epms.sub_problem, :Gradient, :ρ, epms.ρ)
    set_manopt_parameter!(epms.sub_problem, :Gradient, :u, epms.u)
    set_iterate!(epms.sub_state, M, copy(M, epms.p))
    update_stopping_criterion!(epms, :MinIterateChange, epms.ϵ)

    epms.p = get_solver_result(solve!(epms.sub_problem, epms.sub_state))

    # get new evaluation of penalty
    cost_ineq = get_inequality_constraints(amp, epms.p)
    cost_eq = get_equality_constraints(amp, epms.p)
    max_violation = max(max(maximum(cost_ineq; init=0), 0), maximum(abs.(cost_eq); init=0))
    # update ρ if necessary
    (max_violation > epms.u) && (epms.ρ = epms.ρ / epms.θ_ρ)
    # update u and ϵ
    epms.u = max(epms.u_min, epms.u * epms.θ_u)
    epms.ϵ = max(epms.ϵ_min, epms.ϵ * epms.θ_ϵ)
    return epms
end
get_solver_result(epms::ExactPenaltyMethodState) = epms.p
