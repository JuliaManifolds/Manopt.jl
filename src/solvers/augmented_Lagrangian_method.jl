@doc raw"""
    augmented_Lagrangian_method(M, F, sub_problem, sub_options, n_ineq_constraint, n_eq_constraint)

perform the augmented Lagrangian method (ALM)[^Liu2020]. The aim of the ALM is to find the solution of the `ConstrainedProblem`
```math
\begin{aligned}
\min_{x ∈\mathcal{M}} &f(x)\\
\text{subject to } &g_i(x)\leq 0 \quad ∀ i= 1, …, m,\\
\quad &h_j(x)=0 \quad ∀ j=1,…,p,
\end{aligned}
```
where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^p`` are twice continuously differentiable functions from `M` to ℝ.
For that, in every step ``k`` of the algorithm, the augmented Lagrangian function
```math
\mathcal{L}_{ρ^{(k-1)}}(x, λ^{(k-1)}, γ^{(k-1)}) = f(x) + \frac{ρ^{(k-1)}}{2} (\sum_{j=1}^p (h_j(x)+\frac{γ_j^{(k-1)}}{ρ^{(k-1)}})^2 + \sum_{i=1}^m \max\left\{0,\frac{λ_i^{(k-1)}}{ρ^{(k-1)}}+ g_i(x)\right\}^2)
```
is minimized over all ``x ∈\mathcal{M}``, where ``λ^{(k-1)}`` and ``γ_j^{(k-1)}`` are the current iterations of the Lagrange multipliers and ``ρ^{(k-1)}`` is the current penalty parameter.

Then, the Lagrange multipliers are updated by 
```math
γ_j^{(k)} =\operatorname{clip}_{[γ_j^{\min},γ_j^{\max}]} (γ_j^{(k-1)} + ρ^{(k-1)} h_j(x^{(k)})) \text{for all} j=1,…,p
```
and
```math
λ_i^{(k)} =\operatorname{clip}_{[0,λ_i^{\max}]} (λ_i^{(k-1)} + ρ^{(k-1)} g_i(x^{(k)}))
```
for all ``i=1,…,m``, where ``γ_j^{\min} \leq γ_j^{\max}`` and ``λ_i^{\max}`` are the multiplier boundaries. ###### check how they are implemented

Next, we update the accuracy tolerance ``ϵ`` by setting
```math
ϵ^{(k)}=\max\{ϵ_{\min}, θ_ϵ ϵ^{(k-1)}\},

```
where ``ϵ_{\min}`` is the lowest value ``ϵ`` is allowed to become and ``θ_ϵ ∈ (0,1)`` is constant scaling factor.

[^Liu2020]:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > In: Applied Mathematics & Optimization, vol 82, 949–981 (2020),
    > doi [10.1007/s00245-019-09564-3](https://doi.org/10.1007/s00245-019-09564-3)


# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `sub_problem` – problem for the subsolver
* `sub_options` – options of the subproblem
* `n_ineq_constraint` – the number of inequality constraints of the problem
* `n_eq_constraint` – the number of equality constraints of the problem

# Optional
* `x` - initial point
* `max_inner_iter` - (`200`) the maximum number of iterations the subsolver should perform in each iteration ##### 
* `num_outer_itertgn` - (`30`)
* `ϵ` - (`1e-3`) the accuracy tolerance
* `ϵ_min` - (`1e-6`) the lower bound for the accuracy tolerance
* `bound` - (`20`) 
* `λ` - (`ones(n_ineq_constraint)`) the Lagrange multiplier with respect to the inequality constraints
* `γ` - (`ones(n_eq_constraint)`) the Lagrange multiplier with respect to the equality constraints
* `ρ` - (`1.0`) the penalty parameter
* `τ` - (`0.8`) 
* `θ_ρ` - (`0.3`) the scaling factor of the penalty parameter
* `θ_ϵ` - (`(ϵ_min/ϵ)^(1/num_outer_itertgn)`) the scaling factor of the accuracy tolerance
* `oldacc` - (`Inf`) 
* `stopping_criterion` - 


# Output
* `g` – the resulting point of ALM
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function augmented_Lagrangian_method(
    M::AbstractManifold,
    F::TF,
    gradF::TGF,
    sub_problem::Problem,
    sub_options::Options,
    G::Function=x->[], 
    H::Function=x->[],
    gradG::Function=x->[],
    gradH::Function=x->[];
    x=random_point(M), 
    kwargs...,
) where {TF, TGF}
    x_res=allocate(x)
    copyto!(Ref(M), x_res, x)
    return augmented_Lagrangian_method!(M, F, gradF, sub_problem, sub_options, G, H, gradG, gradH; x=x_res, kwargs...)
end

function augmented_Lagrangian_method!(
    M::AbstractManifold,
    F::TF,
    gradF::TGF,
    sub_problem::Problem,
    sub_options::Options,
    G::Function=x->[],
    H::Function=x->[],
    gradG::Function=x->[],
    gradH::Function=x->[];
    x=random_point(M),
    max_inner_iter::Int=200,
    num_outer_itertgn::Int=30,
    ϵ::Real=1e-3, #(starting)tolgradnorm
    ϵ_min::Real=1e-6, #endingtolgradnorm
    bound::Real=20.0, 
    λ::Vector=ones(n_ineq_constraint),
    γ::Vector=ones(n_eq_constraint),
    ρ::Real=1.0, 
    τ::Real=0.8,
    θ_ρ::Real=0.3, 
    θ_ϵ::Real=(ϵ_min/ϵ)^(1/num_outer_itertgn), 
    oldacc::Real=Inf, 
    stopping_criterion::StoppingCriterion=StopWhenAny(StopAfterIteration(300), StopWhenAll(StopWhenSmallerOrEqual(:ϵ, ϵ_min), StopWhenChangeLess(1e-6))), 
    kwargs...,
) where {TF, TGF}
    p = ConstrainedProblem(M, F, G, H, gradF, gradG, gradH)
    o = ALMOptions(
        x,
        max_inner_iter,
        num_outer_itertgn,
        ϵ,
        ϵ_min,
        bound,
        λ,
        γ,
        ρ,
        τ,
        θ_ρ,
        θ_ϵ,
        oldacc,
        stopping_criterion,
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
function initialize_solver!(p::CostProblem, o::ALMOptions)
    o.θ_ϵ = (o.ϵ_min/o.ϵ)^(1/o.num_outer_itertgn)
    o.old_acc = Inf
    return o
end
function step_solver!(p::CostProblem, o::ALMOptions, iter)
    # use subsolver to minimize the augmented Lagrangian within a tolerance ϵ and with max_inner_iter
    cost = get_Lagrangian_cost_function(p, o) 
    grad = get_Lagrangian_gradient(p, o)
    # # put these in the subproblem
    # o.sub_problem.M = p.M
    # o.sub_problem.cost = cost
    # o.sub_problem.gradient = grad
    o.x = gradient_descent(p.M, cost, grad, o.x, stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.ϵ)))

    # update multipliers
    cost_ineq = get_cost_ineq(p, o.x)
    n_ineq_constraint = len(cost_ineq)
    λ = min.(ones(n_ineq_constraint)* o.bound, max.(λ + o.ρ .* cost_ineq, zeros(n_ineq_constraint)))
    cost_eq = get_cost_eq(p, o.x)
    n_eq_constraint = len(cost_eq)
    γ = min.(ones(n_eq_constraint)* o.bound, max.(ones(n_eq_constraint) * (-o.bound), γ + o.ρ .* cost_eq))

    # get new evaluation of penalty
    new_acc = max(max(abs.(max.(-λ./ρ, Ref(cost_ineq)))), max(abs.(cost_eq)))

    # update ρ if necessary
    if iter == 1 || new_acc > o.τ * o.old_acc 
        o.ρ = o.ρ/o.θ_ρ 
    end
    o.old_acc = new_acc

    # update the tolerance ϵ
    ϵ = max(o.ϵ_min, ϵ * o.θ_ϵ)
end
get_solver_result(o::ALMOptions) = o.x

function get_Lagrangian_cost_function(p::CostProblem, o::ALMOptions)
    cost = x -> get_cost(p, x)
    cost_ineq = x -> sum(max.(zeros(o.n_ineq), o.λ ./ o.ρ .+ get_inequality_constraints(p, x)))
    cost_eq = x -> sum((get_equality_constraints(p, x) .+ o.γ./o.ρ)^2)
    return x -> cost(x) + (o.ρ/2) * (cost_ineq(x) + cost_eq(x))
end

function get_Lagrangian_gradient_function(p::CostProblem, o::ALMOptions)
    grad = x -> get_gradient(p, x)
    grad_ineq = x -> sum(
        ((get_inequality_constraints(p, x) .* o.ρ .+ o.λ) .* get_grad_ineq(p, x)).*(get_inequality_constraints(p, x) .+ o.λ./o.ρ .>0)
        )
    grad_eq = x-> sum((get_equality_constraints(p, x) .* o.ρ .+ o.γ) .* get_grad_eq(p, x))
    return x -> grad(x) + grad_ineq(x) + grad_eq(x)
end

# Recreating Changshuo Liu's Matlab source code in Julia
# original code by Changshuo Liu: https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints/blob/master/solvers/almbddmultiplier.m