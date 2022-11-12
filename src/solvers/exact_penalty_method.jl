@doc raw"""
    exact_penalty_method(M, F, gradF; G, H, gradG, gradH)

perform the exact penalty method (EPM)[^LiuBoumal2020]. The aim of the EPM is to find the solution of the [`ConstrainedProblem`](@ref)
```math
\begin{aligned}
\min_{x ∈\mathcal{M}} &f(x)\\
\text{subject to } &g_i(x)\leq 0 \quad ∀ i= 1, …, m,\\
\quad &h_j(x)=0 \quad ∀ j=1,…,p,
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
* `smoothing_technique` – (`"log_sum_exp"`) smoothing technique with which the penalized objective is smoothed (either `"log_sum_exp"` or `"linear_quadratic_huber"`)
* `sub_problem` – (`GradientProblem(M,F,gradF)`) problem for the subsolver
* `sub_options` – (`GradientDescentOptions(M,x)`) options of the subproblem
* `max_inner_iter` – (`200`) the maximum number of iterations the subsolver should perform in each iteration
* `num_outer_itertgn` – (`30`) number of iterations until maximal accuracy is needed to end algorithm naturally
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `u` – (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_min` – (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ` – (`1.0`) the penalty parameter
* `min_stepsize` – (`1e-10`) the minimal step size
* `sub_problem` – ([`GradientProblem`](@ref)`(M,`[`ExactPenaltyCost`](@ref)`(F, G, H, "linear_quadratic_huber", ρ, u),`[`ExactPenaltyGrad`](@ref)`(F, gradF, G, gradG, H, gradH, "linear_quadratic_huber", ρ, u))`) problem for the subsolver
* `sub_options` – ([`QuasiNewtonOptions`](@ref)`(copy(x), zero_vector(M,x), `[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)`(M, copy(M,x), `[`InverseBFGS`](@ref)`(),30), `[`StopAfterIteration`](@ref)`(max_inner_iter) | `[`StopWhenGradientNormLess`](@ref)`(ϵ) | `[`StopWhenStepsizeLess`](@ref)`(min_stepsize), `[`WolfePowellLinesearch`](@ref)`(M,10^(-4),0.999))`) options of the subproblem
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min), `[`StopWhenChangeLess`](@ref)`(1e-6)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `return_options` – (`false`) – if activated, the extended result, i.e. the complete [`Options`](@ref) are returned. This can be used to access recorded values. If set to false (default) just the optimal value `x` is returned.

# Output
* `x` – the resulting point of EPM
or
* `options` – the options returned by the solver (see `return_options`)
"""
function exact_penalty_method(
    M::AbstractManifold, F::TF, gradF::TGF; x=random_point(M), kwargs...
) where {TF,TGF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return exact_penalty_method!(M, F, gradF; x=x_res, kwargs...)
end
@doc raw"""
    exact_penalty_method!(M, F, gradF; G, H, gradG, gradH; x=random_point(M))

perform the exact penalty method (EPM)[^LiuBoumal2020] in place of `x`.

For all options, especially `x` for the initial point and `smoothing_technique` for the smoothing technique, see [`exact_penalty_method`](@ref).
"""
function exact_penalty_method!(
    M::AbstractManifold,
    F::TF,
    gradF::TGF;
    G::Function=(M, x) -> [],
    H::Function=(M, x) -> [],
    gradG::Function=(M, x) -> [],
    gradH::Function=(M, x) -> [],
    evaluation=AllocatingEvaluation(),
    x=random_point(M),
    max_inner_iter::Int=200,
    ϵ::Real=1e-3,
    ϵ_min::Real=1e-6,
    u::Real=1e-1,
    u_min::Real=1e-6,
    ρ::Real=1.0,
    min_stepsize=1e-10,
    sub_problem::Problem=GradientProblem(
        M,
        ExactPenaltyCost(
            ConstrainedProblem(M, F, gradF, F, gradG, H, gradH; evaluation=evaluation),
            ρ,
            u,
        ),
        ExactPenaltyGrad(
            ConstrainedProblem(M, F, gradF, F, gradG, H, gradH; evaluation=evaluation),
            ρ,
            u,
        ),
    ),
    sub_options::Options=QuasiNewtonOptions(
        copy(x),
        zero_vector(M, x),
        QuasiNewtonLimitedMemoryDirectionUpdate(M, copy(M, x), InverseBFGS(), 30),
        StopAfterIteration(max_inner_iter) |
        StopWhenGradientNormLess(ϵ) |
        StopWhenStepsizeLess(min_stepsize),
        WolfePowellLinesearch(M, 10^(-4), 0.999),
    ),
    num_outer_itertgn::Int=30,
    θ_ρ::Real=0.3,
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
        StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenEuclideanChangeLess(min_stepsize)
    ),
    return_options=false,
    kwargs...,
) where {TF,TGF}
    p = ConstrainedProblem(M, F, gradF, G, gradG, H, gradH; evaluation=evaluation)
    o = EPMOptions(
        M,
        p,
        x,
        sub_problem,
        sub_options;
        max_inner_iter=max_inner_iter,
        num_outer_itertgn=num_outer_itertgn,
        ϵ=ϵ,
        ϵ_min=ϵ_min,
        u=u,
        u_min=u_min,
        ρ=ρ,
        θ_ρ=θ_ρ,
        min_stepsize=min_stepsize,
        stopping_criterion=stopping_criterion,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end

#
# Solver functions
#
function initialize_solver!(::ConstrainedProblem, o::EPMOptions)
    o.θ_u = (o.u_min / o.u)^(1 / o.num_outer_itertgn)
    o.θ_ϵ = (o.ϵ_min / o.ϵ)^(1 / o.num_outer_itertgn)
    update_stopping_criterion!(o, :MaxIteration, o.max_inner_iter)
    update_stopping_criterion!(o, :MinStepsize, o.min_stepsize)
    return o
end
function step_solver!(p::ConstrainedProblem, o::EPMOptions, iter)
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
    if max_violation > o.u
        o.ρ = o.ρ / o.θ_ρ
    end

    # update u and ϵ
    o.u = max(o.u_min, o.u * o.θ_u)
    return o.ϵ = max(o.ϵ_min, o.ϵ * o.θ_ϵ)
end
get_solver_result(o::EPMOptions) = o.x
