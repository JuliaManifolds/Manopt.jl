@doc raw"""
    augmented_Lagrangian_method(M, F, gradF, x=random_point(M); kwargs...)

perform the augmented Lagrangian method (ALM)[^LiuBoumal2020]. The aim of the ALM is to find the solution of the [`ConstrainedProblem`](@ref)
```math
\begin{aligned}
\min_{x ∈\mathcal{M}} &f(x)\\
\text{subject to } &g_i(x)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(x)=0 \quad \text{ for } j=1,…,n,
\end{aligned}
```
where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^p`` are twice continuously differentiable functions from `M` to ℝ.
For that, in every step ``k`` of the algorithm, the [`AugmentedLagrangianCost`](@ref)
``\mathcal{L}_{ρ^{(k-1)}}(x, μ^{(k-1)}, λ^{(k-1)})`` is minimized on ``\mathcal{M}``,
where ``μ^{(k-1)} \in \mathbb R^n`` and ``λ^{(k-1)} \in \mathbb R^m`` are the current iterates of the Lagrange multipliers and ``ρ^{(k-1)}`` is the current penalty parameter.

The Lagrange multipliers are then updated by
```math
λ_j^{(k)} =\operatorname{clip}_{[λ_{\min},λ_{\max}]} (λ_j^{(k-1)} + ρ^{(k-1)} h_j(x^{(k)})) \text{for all} j=1,…,p,
```
and
```math
μ_i^{(k)} =\operatorname{clip}_{[0,μ_{\max}]} (μ_i^{(k-1)} + ρ^{(k-1)} g_i(x^{(k)})) \text{ for all } i=1,…,m,
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
* `sub_problem` – ([`GradientProblem`](@ref)`(M, subcost, subgrad)`) problem for the subsolver
* `sub_options` – ([`QuasiNewtonOptions`](@ref)) using [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) with [`InverseBFGS`](@ref) and `sub_stopping_criterion` as a stopping criterion. See also `sub_kwargs`.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(300)` | ([`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min)` & [`StopWhenChangeLess`](@ref)`(1e-10))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details

[^LiuBoumal2020]:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > In: Applied Mathematics & Optimization, vol 82, 949–981 (2020),
    > doi [10.1007/s00245-019-09564-3](https://doi.org/10.1007/s00245-019-09564-3),
    > arXiv: [1901.10000](https://arxiv.org/abs/1901.10000)
    > Matlab source: [https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints](https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints)
"""
function augmented_Lagrangian_method(
    M::AbstractManifold, F::TF, gradF::TGF, x=random_point(M); kwargs...
) where {TF,TGF}
    x_res = copy(M, x)
    return augmented_Lagrangian_method!(M, F, gradF, x_res; kwargs...)
end
@doc raw"""
    augmented_Lagrangian_method!(M, F, gradF x=random_point(M); kwargs...)

perform the augmented Lagrangian method (ALM) in-place of `x`.

For all options, see [`augmented_Lagrangian_method`](@ref).
"""
function augmented_Lagrangian_method!(
    M::AbstractManifold,
    F::TF,
    gradF::TGF,
    x=random_point(M);
    G=nothing,
    H=nothing,
    gradG=nothing,
    gradH=nothing,
    evaluation=AllocatingEvaluation(),
    ϵ::Real=1e-3,
    ϵ_min::Real=1e-6,
    ϵ_exponent=1 / 100,
    θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
    _m=isnothing(G) ? 0 : (G isa Function ? size(G(M, x)) : length(G)),
    μ::Vector=ones(_m),
    μ_max::Real=20.0,
    _n=isnothing(H) ? 0 : (H isa Function ? size(H(M, x)) : length(H)),
    λ::Vector=ones(_n),
    λ_max::Real=20.0,
    λ_min::Real=-λ_max,
    τ::Real=0.8,
    ρ::Real=1.0,
    θ_ρ::Real=0.3,
    _problem=ConstrainedProblem(M, F, gradF, G, gradG, H, gradH; evaluation=evaluation),
    sub_cost=AugmentedLagrangianCost(_problem, ρ, μ, λ),
    sub_grad=AugmentedLagrangianGrad(_problem, ρ, μ, λ),
    sub_kwargs=[],
    sub_stopping_criterion=StopAfterIteration(300) |
                           StopWhenGradientNormLess(ϵ) |
                           StopWhenStepsizeLess(1e-8),
    sub_options::Options=decorate_options(
        QuasiNewtonOptions(
            M,
            copy(x);
            initial_vector=zero_vector(M, x),
            direction_update=QuasiNewtonLimitedMemoryDirectionUpdate(
                M, copy(M, x), InverseBFGS(), 30
            ),
            stopping_criterion=sub_stopping_criterion,
            stepsize=WolfePowellLinesearch(M, 1e-3, 0.999, linesearch_stopsize=1e-8),
        );
        sub_kwargs...,
    ),
    sub_problem::Problem=GradientProblem(M, sub_cost, sub_grad),
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
        StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(1e-10)
    ),
    kwargs...,
) where {TF,TGF}
    o = AugmentedLagrangianMethodOptions(
        M,
        _problem,
        x,
        sub_problem,
        sub_options;
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
    o = decorate_options(o; kwargs...)
    return get_solver_return(solve(_problem, o))
end

#
# Solver functions
#
function initialize_solver!(::ConstrainedProblem, o::AugmentedLagrangianMethodOptions)
    o.penalty = Inf
    return o
end
function step_solver!(p::ConstrainedProblem, o::AugmentedLagrangianMethodOptions, iter)
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
    penalty = maximum([abs.(max.(-o.μ ./ o.ρ, cost_ineq))..., abs.(cost_eq)...]; init=0)
    # update ρ if necessary
    (penalty > o.τ * o.penalty) && (o.ρ = o.ρ / o.θ_ρ)
    o.penalty = penalty

    # update the tolerance ϵ
    o.ϵ = max(o.ϵ_min, o.ϵ * o.θ_ϵ)
    return o
end
get_solver_result(o::AugmentedLagrangianMethodOptions) = o.x
