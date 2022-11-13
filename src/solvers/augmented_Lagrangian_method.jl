@doc raw"""
    augmented_Lagrangian_method(M, F, gradF, sub_problem, sub_options, G, H, gradG, gradH)

perform the augmented Lagrangian method (ALM)[^LiuBoumal2020]. The aim of the ALM is to find the solution of the [`ConstrainedProblem`](@ref)
```math
\begin{aligned}
\min_{x ∈\mathcal{M}} &f(x)\\
\text{subject to } &g_i(x)\leq 0 \quad ∀ i= 1, …, m,\\
\quad &h_j(x)=0 \quad ∀ j=1,…,n,
\end{aligned}
```
where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^p`` are twice continuously differentiable functions from `M` to ℝ.
For that, in every step ``k`` of the algorithm, the [`AugemtedLagrangianCost`](@ref)
``\mathcal{L}_{ρ^{(k-1)}}(x, μ^{(k-1)}, λ^{(k-1)})`` is minimized on ``\mathcal{M}``,
where ``μ^{(k-1)} \in \mathbb R^n`` and ``λ^{(k-1)} \in \mathbb R^m` are the current iterates of the Lagrange multipliers and ``ρ^{(k-1)}`` is the current penalty parameter.

The Lagrange multipliers are then updated by
```math
λ_j^{(k)} =\operatorname{clip}_{[λ_{\min},λ_{\max}]} (λ_j^{(k-1)} + ρ^{(k-1)} h_j(x^{(k)})) \text{for all} j=1,…,p,
```
and
```math
μ_i^{(k)} =\operatorname{clip}_{[0,μ_{\max}]} (μ_i^{(k-1)} + ρ^{(k-1)} g_i(x^{(k)})) \text{for all}  i=1,…,m,
```
where ``λ_{\min} \leq λ_{\max}`` and ``μ_{\max}`` are the multiplier boundaries.

Next, we update the accuracy tolerance ``ϵ`` by setting
```math
ϵ^{(k)}=\max\{ϵ_{\min}, θ_ϵ ϵ^{(k-1)}\},
```
where ``ϵ_{\min}`` is the lowest value ``ϵ`` is allowed to become and ``θ_ϵ ∈ (0,1)`` is constant scaling factor.

Last, we update the penalty parameter ``ρ``. For this, we define
```math
σ^{(k)}=\max_{j=1,…,p, i=1,…,m} \{\|h_j(x^{(k)})\|, \|\max_{i=1,…,m}\{g_i(x^{(k)}), -\frac{μ_i^{(k-1)}}{ρ^{(k-1)}} \}\| \}.
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
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF` – the gradient of the cost function

# Optional
* `G` – the inequality constraints
* `H` – the equality constraints
* `gradG` – the gradient of the inequality constraints
* `gradH` – the gradient of the equality constraints
* `x` – initial point
* `sub_problem` – problem for the subsolver
* `sub_options` – options of the subproblem
* `max_inner_iter` – (`200`) the maximum number of iterations the subsolver should perform in each iteration
* `num_outer_itertgn` – (`100`) number of iterations until maximal accuracy is needed to end algorithm naturally
* `ϵ` – (`1e-3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `μ` – (`ones(size(G(M,x),1))`) the Lagrange multiplier with respect to the inequality constraints
* `λ` – (`ones(size(H(M,x),1))`) the Lagrange multiplier with respect to the equality constraints
* `ρ` – (`1.0`) the penalty parameter
* `min_stepsize` – (`1e-10`) the minimal step size
* `sub_problem` – ([`GradientProblem`](@ref)`(M,`[`AugmentedLagrangianCost`](@ref)`(F, G, H, ρ, μ, λ),`[`AugmentedLagrangianGrad`](@ref)`(F, gradF, G, gradG, H, gradH, ρ, μ, λ))`) problem for the subsolver
* `sub_options` – ([`QuasiNewtonOptions`](@ref)`(copy(x), zero_vector(M,x), `[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)`(M, copy(M,x), `[`InverseBFGS`](@ref)`(),30), `[`StopAfterIteration`](@ref)`(max_inner_iter) | `[`StopWhenGradientNormLess`](@ref)`(ϵ) | `[`StopWhenStepsizeLess`](@ref)`(min_stepsize), `[`WolfePowellLinesearch`](@ref)`(M,10^(-4),0.999))`) options of the subproblem
* `λ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the equality constraints
* `λ_min` – (`- λ_max`) a lower bound for the Lagrange multiplier belonging to the equality constraints
* `μ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the inequality constraints
* `τ` – (`0.8`) factor for the improvement of the evaluation of the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(300)` | [`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min)` & [`StopWhenChangeLess`](@ref)`(min_stepsize)`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `return_options` – (`false`) – if activated, the extended result, i.e. the complete [`Options`](@ref) are returned. This can be used to access recorded values. If set to false (default) just the optimal value `x` is returned.

# Output
* `x` – the resulting point of ALM
OR
* `options` – the options returned by the solver (see `return_options`)

[^LiuBoumal2020]:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > In: Applied Mathematics & Optimization, vol 82, 949–981 (2020),
    > doi [10.1007/s00245-019-09564-3](https://doi.org/10.1007/s00245-019-09564-3),
    > Matlab source: [https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints](https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints)
"""
function augmented_Lagrangian_method(
    M::AbstractManifold, F::TF, gradF::TGF; x=random_point(M), kwargs...
) where {TF,TGF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return augmented_Lagrangian_method!(M, F, gradF; x=x_res, kwargs...)
end
@doc raw"""
    augmented_Lagrangian_method!(M, F, gradF; x=random_point(M))

perform the augmented Lagrangian method (ALM) in-place of `x`.

For all options, see [`augmented_Lagrangian_method`](@ref).
"""
function augmented_Lagrangian_method!(
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
    μ::Vector=ones(size(G(M, x), 1)),
    λ::Vector=ones(size(H(M, x), 1)),
    ρ::Real=1.0,
    min_stepsize=1e-10,
    sub_problem::Problem=GradientProblem(
        M,
        AugmentedLagrangianCost(
            ConstrainedProblem(M, F, gradF, F, gradG, H, gradH; evaluation=evaluation),
            ρ,
            μ,
            λ,
        ),
        AugmentedLagrangianGrad(
            ConstrainedProblem(M, F, gradF, F, gradG, H, gradH; evaluation=evaluation),
            ρ,
            μ,
            λ,
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
    num_outer_itertgn::Int=100,
    λ_max::Real=20.0,
    λ_min::Real=-λ_max,
    μ_max::Real=20.0,
    τ::Real=0.8,
    θ_ρ::Real=0.3,
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
        StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(min_stepsize)
    ),
    return_options=false,
    kwargs...,
) where {TF,TGF}
    p = ConstrainedProblem(M, F, gradF, G, gradG, H, gradH; evaluation=evaluation)
    o = ALMOptions(
        M,
        p,
        x,
        sub_problem,
        sub_options;
        max_inner_iter=max_inner_iter,
        num_outer_itertgn=num_outer_itertgn,
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
        min_stepsize=min_stepsize,
        stopping_criterion=stopping_criterion,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    return_options && return resultO
    return get_solver_result(resultO)
end

#
# Solver functions
#
function initialize_solver!(::ConstrainedProblem, o::ALMOptions)
    o.θ_ϵ = (o.ϵ_min / o.ϵ)^(1 / o.num_outer_itertgn)
    o.old_acc = Inf
    update_stopping_criterion!(o, :MaxIteration, o.max_inner_iter)
    update_stopping_criterion!(o, :MinStepsize, o.min_stepsize)
    return o
end
function step_solver!(p::ConstrainedProblem, o::ALMOptions, iter)
    # use subsolver to minimize the augmented Lagrangian
    o.sub_problem.cost.ρ = o.ρ
    o.sub_problem.cost.μ = o.μ
    o.sub_problem.cost.λ = o.λ
    o.sub_problem.gradient!!.ρ = o.ρ
    o.sub_problem.gradient!!.μ = o.μ
    o.sub_problem.gradient!!.λ = o.λ
    o.sub_options.x = copy(o.x)
    update_stopping_criterion!(o, :MinIterateChange, o.ϵ)

    o.x = get_solver_result(solve(o.sub_problem, o.sub_options))

    # update multipliers
    cost_ineq = get_inequality_constraints(p, o.x)
    n_ineq_constraint = size(cost_ineq, 1)
    o.μ = convert(
        Vector{Float64},
        min.(
            ones(n_ineq_constraint) .* o.μ_max,
            max.(o.μ + o.ρ .* cost_ineq, zeros(n_ineq_constraint)),
        ),
    )
    cost_eq = get_equality_constraints(p, o.x)
    n_eq_constraint = size(cost_eq, 1)
    o.λ = convert(
        Vector{Float64},
        min.(
            ones(n_eq_constraint) .* o.λ_max,
            max.(ones(n_eq_constraint) .* o.λ_min, o.λ + o.ρ .* cost_eq),
        ),
    )

    # get new evaluation of penalty
    new_acc = max(
        maximum(abs.(max.(-o.μ ./ o.ρ, cost_ineq)); init=0), maximum(abs.(cost_eq); init=0)
    )

    # update ρ if necessary
    (iter == 1 || new_acc > o.τ * o.old_acc) && (o.ρ = o.ρ / o.θ_ρ)
    o.old_acc = new_acc

    # update the tolerance ϵ
    o.ϵ = max(o.ϵ_min, o.ϵ * o.θ_ϵ)
    return o
end
get_solver_result(o::ALMOptions) = o.x
