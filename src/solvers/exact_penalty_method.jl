@doc raw"""
    exact_penalty_method(M, F, gradF, x=random_point(M); kwargs...)

perform the exact penalty method (EPM)[^LiuBoumal2020]. The aim of the EPM is to find the solution of the [`ConstrainedProblem`](@ref)
```math
\begin{aligned}
\min_{x ∈\mathcal{M}} &f(x)\\
\text{subject to } &g_i(x)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(x)=0 \quad  \text{ for } j=1,…,p,
\end{aligned}
```
where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^p`` are twice continuously differentiable functions from `M` to ℝ.
For that a weighted ``L_1``-penalty term for the violation of the constraints is added to the objective
```math
f(x) + ρ (\sum_{i=1}^m \max\left\{0, g_i(x)\right\} + \sum_{j=1}^p \vert h_j(x)\vert),
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
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF` – the gradient of the cost function

# Optional
* `G` – the inequality constraints
* `H` – the equality constraints
* `gradG` – the gradient of the inequality constraints
* `gradH` – the gradient of the equality constraints
* `x` – initial point
* `smoothing` – ([`LogarithmicSumOfExponentials`](@ref)) [`SmoothingTechnique`](@ref) to use
* `ϵ` – (`1e–3`) the accuracy tolerance
* `u_exponent` – (`1/100`) exponent of the ϵ update factor;
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `u` – (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_exponent` – (`1/100`) exponent of the ϵ update factor;
* `u_min` – (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ` – (`1.0`) the penalty parameter
* `min_stepsize` – (`1e-10`) the minimal step size
* `sub_cost` – ([`ExactPenaltyCost`](@ref)`(problem, ρ, u; smoothing=smoothing)`) use this exact penality cost, expecially with the same numbers `ρ,u` as in the options for the sub problem
* `sub_grad` – ([`ExactPenaltyGrad`](@ref)`(problem, ρ, u; smoothing=smoothing)`) use this exact penality gradient, expecially with the same numbers `ρ,u` as in the options for the sub problem
* `sub_kwargs` – keyword arguments to decorate the sub options, e.g. with debug.
* `sub_stopping_criterion` – ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenGradientNormLess`](@ref)`(ϵ) | `[`StopWhenStepsizeLess`](@ref)`(1e-10)`) specify a stopping criterion for the subsolver.
* `sub_problem` – ([`GradientProblem`](@ref)`(M, subcost, subgrad)`) problem for the subsolver
* `sub_options` – ([`QuasiNewtonOptions`](@ref)) using [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) with [`InverseBFGS`](@ref) and `sub_stopping_criterion` as a stopping criterion. See also `sub_kwargs`.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(300)` | ([`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min)` & [`StopWhenChangeLess`](@ref)`(1e-10)`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function exact_penalty_method(
    M::AbstractManifold, F::TF, gradF::TGF, x=random_point(M); kwargs...
) where {TF,TGF}
    x_res = copy(M, x)
    return exact_penalty_method!(M, F, gradF, x_res; kwargs...)
end
@doc raw"""
    exact_penalty_method!(M, F, gradF, x=random_point(M); kwargs...)

perform the exact penalty method (EPM)[^LiuBoumal2020] in place of `x`.

For all options, especially `x` for the initial point and `smoothing_technique` for the smoothing technique, see [`exact_penalty_method`](@ref).
"""
function exact_penalty_method!(
    M::AbstractManifold,
    F::TF,
    gradF::TGF,
    x=random_point(M);
    G::Function=(M, x) -> [],
    H::Function=(M, x) -> [],
    gradG::Function=(M, x) -> [],
    gradH::Function=(M, x) -> [],
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
    problem=ConstrainedProblem(M, F, gradF, G, gradG, H, gradH; evaluation=evaluation),
    sub_cost=ExactPenaltyCost(problem, ρ, u; smoothing=smoothing),
    sub_grad=ExactPenaltyGrad(problem, ρ, u; smoothing=smoothing),
    sub_problem::Problem=GradientProblem(M, sub_cost, sub_grad),
    sub_kwargs=[],
    sub_stopping_criterion=StopAfterIteration(200) |
                           StopWhenGradientNormLess(ϵ) |
                           StopWhenStepsizeLess(1e-10),
    sub_options::Options=decorate_options(
        QuasiNewtonOptions(
            M,
            copy(x);
            initial_vector=zero_vector(M, x),
            direction_update=QuasiNewtonLimitedMemoryDirectionUpdate(
                M, copy(M, x), InverseBFGS(), 30
            ),
            stopping_criterion=sub_stopping_criterion,
            stepsize=WolfePowellLinesearch(M, 1e-4, 0.999),
        ),
        sub_kwargs...,
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
        StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(1e-10)
    ),
    kwargs...,
) where {TF,TGF}
    o = ExactPenaltyMethodOptions(
        M,
        x,
        sub_problem,
        sub_options;
        ϵ=ϵ,
        ϵ_min=ϵ_min,
        u=u,
        u_min=u_min,
        ρ=ρ,
        θ_ρ=θ_ρ,
        θ_ϵ=θ_ϵ,
        θ_u=θ_u,
    )
    o = decorate_options(o; kwargs...)
    return get_solver_return(solve(problem, o))
end
#
# Solver functions
#
function initialize_solver!(::ConstrainedProblem, o::ExactPenaltyMethodOptions)
    return o
end
function step_solver!(p::ConstrainedProblem, o::ExactPenaltyMethodOptions, iter)
    # use subsolver to minimize the smoothed penalized function
    o.sub_problem.cost.ρ = o.ρ
    o.sub_problem.cost.u = o.u
    o.sub_problem.gradient!!.ρ = o.ρ
    o.sub_problem.gradient!!.u = o.u
    o.sub_options.x = copy(o.x)
    update_stopping_criterion!(o, :MinIterateChange, o.ϵ)

    o.x = get_solver_result(solve(o.sub_problem, o.sub_options))

    # get new evaluation of penalty
    cost_ineq = get_inequality_constraints(p, o.x)
    cost_eq = get_equality_constraints(p, o.x)
    max_violation = max(max(maximum(cost_ineq; init=0), 0), maximum(abs.(cost_eq); init=0))
    # update ρ if necessary
    (max_violation > o.u) && (o.ρ = o.ρ / o.θ_ρ)
    # update u and ϵ
    o.u = max(o.u_min, o.u * o.θ_u)
    o.ϵ = max(o.ϵ_min, o.ϵ * o.θ_ϵ)
    return o
end
get_solver_result(o::ExactPenaltyMethodOptions) = o.x
