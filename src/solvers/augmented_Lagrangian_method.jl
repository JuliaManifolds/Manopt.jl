#
# State
#
@doc raw"""
    AugmentedLagrangianMethodState{P,T} <: AbstractManoptSolverState

Describes the augmented Lagrangian method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `p` – a point on a manifold as starting point and current iterate
* `sub_problem` – an [`AbstractManoptProblem`](@ref) problem for the subsolver
* `sub_state` – an [`AbstractManoptSolverState`](@ref) for the subsolver
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `λ` – (`ones(len(`[`get_equality_constraints`](@ref)`(p,x))`) the Lagrange multiplier with respect to the equality constraints
* `λ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the equality constraints
* `λ_min` – (`- λ_max`) a lower bound for the Lagrange multiplier belonging to the equality constraints
* `μ` – (`ones(len(`[`get_inequality_constraints`](@ref)`(p,x))`) the Lagrange multiplier with respect to the inequality constraints
* `μ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the inequality constraints
* `ρ` – (`1.0`) the penalty parameter
* `τ` – (`0.8`) factor for the improvement of the evaluation of the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `θ_ϵ` – (`(ϵ_min/ϵ)^(ϵ_exponent)`) the scaling factor of the accuracy tolerance
* `penalty` – evaluation of the current penalty term, initialized to `Inf`.
* `stopping_criterion` – (`(`[`StopAfterIteration`](@ref)`(300) | (`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min) & `[`StopWhenChangeLess`](@ref)`(1e-10))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.


# Constructor

    AugmentedLagrangianMethodState(M::AbstractManifold, co::ConstrainedManifoldObjective, p; kwargs...)

construct an augmented Lagrangian method options with the fields and defaults as above,
where the manifold `M` and the [`ConstrainedManifoldObjective`](@ref) `co` are used for defaults
in the keyword arguments.

# See also

[`augmented_Lagrangian_method`](@ref)
"""
mutable struct AugmentedLagrangianMethodState{
    P,Pr<:AbstractManoptProblem,Op<:AbstractManoptSolverState,TStopping<:StoppingCriterion
} <: AbstractManoptSolverState
    p::P
    sub_problem::Pr
    sub_state::Op
    ϵ::Real
    ϵ_min::Real
    λ_max::Real
    λ_min::Real
    μ_max::Real
    μ::Vector
    λ::Vector
    ρ::Real
    τ::Real
    θ_ρ::Real
    θ_ϵ::Real
    penalty::Real
    stop::TStopping
    function AugmentedLagrangianMethodState(
        M::AbstractManifold,
        co::ConstrainedManifoldObjective,
        p::P,
        sub_problem::Pr,
        sub_state::St;
        ϵ::Real=1e-3,
        ϵ_min::Real=1e-6,
        λ_max::Real=20.0,
        λ_min::Real=-λ_max,
        μ_max::Real=20.0,
        μ::Vector=ones(length(get_inequality_constraints(M, co, p))),
        λ::Vector=ones(length(get_equality_constraints(M, co, p))),
        ρ::Real=1.0,
        τ::Real=0.8,
        θ_ρ::Real=0.3,
        ϵ_exponent=1 / 100,
        θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
        stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
            StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(1e-10)
        ),
    ) where {P,Pr<:AbstractManoptProblem,St<:AbstractManoptSolverState}
        alms = new{P,Pr,St,typeof(stopping_criterion)}()
        alms.p = p
        alms.sub_problem = sub_problem
        alms.sub_state = sub_state
        alms.ϵ = ϵ
        alms.ϵ_min = ϵ_min
        alms.λ_max = λ_max
        alms.λ_min = λ_min
        alms.μ_max = μ_max
        alms.μ = μ
        alms.λ = λ
        alms.ρ = ρ
        alms.τ = τ
        alms.θ_ρ = θ_ρ
        alms.θ_ϵ = θ_ϵ
        alms.penalty = Inf
        alms.stop = stopping_criterion
        return alms
    end
end

get_iterate(alms::AugmentedLagrangianMethodState) = alms.p
function set_iterate!(alms::AugmentedLagrangianMethodState, M, p)
    alms.p = p
    return alms
end
function get_message(alms::AugmentedLagrangianMethodState)
    # for now only the sub solver might have messages
    return get_message(alms.sub_state)
end
function get_message_type(alms::AugmentedLagrangianMethodState)
    # for now only the sub solver might have messages
    return get_message_type(alms.sub_state)
end
function show(io::IO, alms::AugmentedLagrangianMethodState)
    i = get_count(alms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(alms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Augmented Lagrangian Method
    $Iter
    ## Parameters
    * ϵ: $(alms.ϵ) (ϵ_min: $(alms.ϵ_min), θ_ϵ: $(alms.θ_ϵ))
    * λ: $(alms.λ) (λ_min: $(alms.λ_min), λ_max: $(alms.λ_max))
    * μ: $(alms.μ) (μ_max: $(alms.μ_max))
    * ρ: $(alms.ρ) (θ_ρ: $(alms.θ_ρ))
    * τ: $(alms.τ)
    * current penalty: $(alms.penalty)

    ## Stopping Criterion
    $(status_summary(alms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    augmented_Lagrangian_method(M, f, grad_f, p=rand(M); kwargs...)

perform the augmented Lagrangian method (ALM)[^LiuBoumal2020].
The aim of the ALM is to find the solution of the constrained optimisation task

```math
\begin{aligned}
\min_{p ∈\mathcal{M}} &f(p)\\
\text{subject to } &g_i(p)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(p)=0 \quad \text{ for } j=1,…,n,
\end{aligned}
```

where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^p`` are twice continuously differentiable functions from `M` to ℝ.
In every step ``k`` of the algorithm, the [`AugmentedLagrangianCost`](@ref)
``\mathcal{L}_{ρ^{(k-1)}}(p, μ^{(k-1)}, λ^{(k-1)})`` is minimized on ``\mathcal{M}``,
where ``μ^{(k-1)} \in \mathbb R^n`` and ``λ^{(k-1)} \in \mathbb R^m`` are the current iterates of the Lagrange multipliers and ``ρ^{(k-1)}`` is the current penalty parameter.

The Lagrange multipliers are then updated by
```math
λ_j^{(k)} =\operatorname{clip}_{[λ_{\min},λ_{\max}]} (λ_j^{(k-1)} + ρ^{(k-1)} h_j(p^{(k)})) \text{for all} j=1,…,p,
```
and
```math
μ_i^{(k)} =\operatorname{clip}_{[0,μ_{\max}]} (μ_i^{(k-1)} + ρ^{(k-1)} g_i(p^{(k)})) \text{ for all } i=1,…,m,
```
where ``λ_{\min} \leq λ_{\max}`` and ``μ_{\max}`` are the multiplier boundaries.

Next, we update the accuracy tolerance ``ϵ`` by setting
```math
ϵ^{(k)}=\max\{ϵ_{\min}, θ_ϵ ϵ^{(k-1)}\},
```
where ``ϵ_{\min}`` is the lowest value ``ϵ`` is allowed to become and ``θ_ϵ ∈ (0,1)`` is constant scaling factor.

Last, we update the penalty parameter ``ρ``. For this, we define
```math
σ^{(k)}=\max_{j=1,…,p, i=1,…,m} \{\|h_j(p^{(k)})\|, \|\max_{i=1,…,m}\{g_i(p^{(k)}), -\frac{μ_i^{(k-1)}}{ρ^{(k-1)}} \}\| \}.
```
Then, we update `ρ` according to
```math
ρ^{(k)} = \begin{cases}
ρ^{(k-1)}/θ_ρ,  & \text{if } σ^{(k)}\leq θ_ρ σ^{(k-1)} ,\\
ρ^{(k-1)}, & \text{else,}
\end{cases}
```
where ``θ_ρ \in (0,1)`` is a constant scaling factor.

# Input
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `grad_f` – the gradient of the cost function

# Optional
* `g` – the inequality constraints
* `h` – the equality constraints
* `grad_g` – the gradient of the inequality constraints
* `grad_h` – the gradient of the equality constraints
* `ϵ` – (`1e-3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `ϵ_exponent` – (`1/100`) exponent of the ϵ update factor;
   also 1/number of iterations until maximal accuracy is needed to end algorithm naturally
* `θ_ϵ` – (`(ϵ_min / ϵ)^(ϵ_exponent)`) the scaling factor of the exactness
* `μ` – (`ones(size(G(M,x),1))`) the Lagrange multiplier with respect to the inequality constraints
* `μ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the inequality constraints
* `λ` – (`ones(size(H(M,x),1))`) the Lagrange multiplier with respect to the equality constraints
* `λ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the equality constraints
* `λ_min` – (`- λ_max`) a lower bound for the Lagrange multiplier belonging to the equality constraints
* `τ` – (`0.8`) factor for the improvement of the evaluation of the penalty parameter
* `ρ` – (`1.0`) the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `sub_cost` – ([`AugmentedLagrangianCost`](@ref)`(problem, ρ, μ, λ)`) use augmented Lagranian, expecially with the same numbers `ρ,μ` as in the options for the sub problem
* `sub_grad` – ([`AugmentedLagrangianGrad`](@ref)`(problem, ρ, μ, λ)`) use augmented Lagranian gradient, expecially with the same numbers `ρ,μ` as in the options for the sub problem
* `sub_kwargs` – keyword arguments to decorate the sub options, e.g. with debug.
* `sub_stopping_criterion` – ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenGradientNormLess`](@ref)`(ϵ) | `[`StopWhenStepsizeLess`](@ref)`(1e-8)`) specify a stopping criterion for the subsolver.
* `sub_problem` – ([`DefaultManoptProblem`](@ref)`(M, `[`ConstrainedManifoldObjective`](@ref)`(subcost, subgrad; evaluation=evaluation))`) problem for the subsolver
* `sub_state` – ([`QuasiNewtonState`](@ref)) using [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) with [`InverseBFGS`](@ref) and `sub_stopping_criterion` as a stopping criterion. See also `sub_kwargs`.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(300)` | ([`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min)` & [`StopWhenChangeLess`](@ref)`(1e-10))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details

[^LiuBoumal2020]:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > In: Applied Mathematics & Optimization, vol 82, 949–981 (2020),
    > doi [10.1007/s00245-019-09564-3](https://doi.org/10.1007/s00245-019-09564-3),
    > arXiv: [1901.10000](https://arxiv.org/abs/1901.10000)
    > Matlab source: [https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints](https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints)
"""
function augmented_Lagrangian_method(
    M::AbstractManifold, f::TF, grad_f::TGF, p=rand(M); kwargs...
) where {TF,TGF}
    q = copy(M, p)
    return augmented_Lagrangian_method!(M, f, grad_f, q; kwargs...)
end
@doc raw"""
    augmented_Lagrangian_method!(M, f, grad_f p=rand(M); kwargs...)

perform the augmented Lagrangian method (ALM) in-place of `p`.

For all options, see [`augmented_Lagrangian_method`](@ref).
"""
function augmented_Lagrangian_method!(
    M::AbstractManifold,
    f::TF,
    grad_f::TGF,
    p=rand(M);
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    evaluation=AllocatingEvaluation(),
    ϵ::Real=1e-3,
    ϵ_min::Real=1e-6,
    ϵ_exponent=1 / 100,
    θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
    _m=isnothing(g) ? 0 : (g isa Function ? size(g(M, p)) : length(g)),
    μ::Vector=ones(_m),
    μ_max::Real=20.0,
    _n=isnothing(h) ? 0 : (h isa Function ? size(h(M, p)) : length(h)),
    λ::Vector=ones(_n),
    λ_max::Real=20.0,
    λ_min::Real=-λ_max,
    τ::Real=0.8,
    ρ::Real=1.0,
    θ_ρ::Real=0.3,
    _objective=ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h; evaluation=evaluation
    ),
    sub_cost=AugmentedLagrangianCost(_objective, ρ, μ, λ),
    sub_grad=AugmentedLagrangianGrad(_objective, ρ, μ, λ),
    sub_kwargs=[],
    sub_stopping_criterion=StopAfterIteration(300) |
                           StopWhenGradientNormLess(ϵ) |
                           StopWhenStepsizeLess(1e-8),
    sub_state::AbstractManoptSolverState=decorate_state!(
        QuasiNewtonState(
            M,
            copy(p);
            initial_vector=zero_vector(M, p),
            direction_update=QuasiNewtonLimitedMemoryDirectionUpdate(
                M, copy(M, p), InverseBFGS(), 30
            ),
            stopping_criterion=sub_stopping_criterion,
            stepsize=default_stepsize(M, QuasiNewtonState),
        );
        sub_kwargs...,
    ),
    sub_problem::AbstractManoptProblem=DefaultManoptProblem(
        M, ManifoldGradientObjective(sub_cost, sub_grad; evaluation=evaluation)
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
        StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(1e-10)
    ),
    kwargs...,
) where {TF,TGF}
    alms = AugmentedLagrangianMethodState(
        M,
        _objective,
        p,
        sub_problem,
        sub_state;
        ϵ=ϵ,
        ϵ_min=ϵ_min,
        λ_max=λ_max,
        λ_min=λ_min,
        μ_max=μ_max,
        μ=μ,
        λ=λ,
        ρ=ρ,
        τ=τ,
        θ_ρ=θ_ρ,
        θ_ϵ=θ_ϵ,
        stopping_criterion=stopping_criterion,
    )
    obj = decorate_objective!(M, _objective; kwargs...)
    mp = DefaultManoptProblem(M, obj)
    alms = decorate_state!(alms; kwargs...)
    return get_solver_return(solve!(mp, alms))
end

#
# Solver functions
#
function initialize_solver!(::AbstractManoptProblem, alms::AugmentedLagrangianMethodState)
    alms.penalty = Inf
    return alms
end
function step_solver!(mp::AbstractManoptProblem, alms::AugmentedLagrangianMethodState, iter)
    M = get_manifold(mp)
    # use subsolver to minimize the augmented Lagrangian
    set_manopt_parameter!(alms.sub_problem, :Cost, :ρ, alms.ρ)
    set_manopt_parameter!(alms.sub_problem, :Cost, :μ, alms.μ)
    set_manopt_parameter!(alms.sub_problem, :Cost, :λ, alms.λ)
    set_manopt_parameter!(alms.sub_problem, :Gradient, :ρ, alms.ρ)
    set_manopt_parameter!(alms.sub_problem, :Gradient, :μ, alms.μ)
    set_manopt_parameter!(alms.sub_problem, :Gradient, :λ, alms.λ)
    set_iterate!(alms.sub_state, M, copy(M, alms.p))

    update_stopping_criterion!(alms, :MinIterateChange, alms.ϵ)

    copyto!(M, alms.p, get_solver_result(solve!(alms.sub_problem, alms.sub_state)))

    # update multipliers
    cost_ineq = get_inequality_constraints(mp, alms.p)
    n_ineq_constraint = size(cost_ineq, 1)
    alms.μ = convert(
        Vector{Float64},
        min.(
            ones(n_ineq_constraint) .* alms.μ_max,
            max.(alms.μ + alms.ρ .* cost_ineq, zeros(n_ineq_constraint)),
        ),
    )
    cost_eq = get_equality_constraints(mp, alms.p)
    n_eq_constraint = size(cost_eq, 1)
    alms.λ = convert(
        Vector{Float64},
        min.(
            ones(n_eq_constraint) .* alms.λ_max,
            max.(ones(n_eq_constraint) .* alms.λ_min, alms.λ + alms.ρ .* cost_eq),
        ),
    )
    # get new evaluation of penalty
    penalty = maximum(
        [abs.(max.(-alms.μ ./ alms.ρ, cost_ineq))..., abs.(cost_eq)...]; init=0
    )
    # update ρ if necessary
    (penalty > alms.τ * alms.penalty) && (alms.ρ = alms.ρ / alms.θ_ρ)
    alms.penalty = penalty

    # update the tolerance ϵ
    alms.ϵ = max(alms.ϵ_min, alms.ϵ * alms.θ_ϵ)
    return alms
end
get_solver_result(alms::AugmentedLagrangianMethodState) = alms.p
