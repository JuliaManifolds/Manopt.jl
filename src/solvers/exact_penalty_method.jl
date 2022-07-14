@doc raw"""
    exact_penalty_method(M, F, gradF; G, H, gradG, gradH)

perform the exact penalty method (EPM)[^LiuBoumal2020][^source_code]. The aim of the EPM is to find the solution of the [`ConstrainedProblem`](@ref)
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
f(x) + ρ (\sum_{i=1}^m \max\left\{0, g_i(x)\right\} + \sum_{j=1}^p \vert h_j(x)\vert)
```


[^LiuBoumal2020]:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > In: Applied Mathematics & Optimization, vol 82, 949–981 (2020),
    > doi [10.1007/s00245-019-09564-3](https://doi.org/10.1007/s00245-019-09564-3)

[^source_code]:
    > original source code to the paper:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > src: [https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints](https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints)


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
* `num_outer_itertgn` – (`30`)
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `u` – (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_min` – (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ` – (`1.0`) the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `θ_u` – (`(u_min/u)^(1/num_outer_itertgn)`) the scaling factor of the smoothing parameter and threshold for violation of the constraints
* `θ_ϵ` – (`(ϵ_min/ϵ)^(1/num_outer_itertgn)`) the scaling factor of the accuracy tolerance
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min), `[`StopWhenChangeLess`](@ref)`(1e-6)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `return_options` – (`false`) – if activated, the extended result, i.e. the complete [`Options`](@ref) are returned. This can be used to access recorded values. If set to false (default) just the optimal value `x` is returned.

# Output
* `x` – the resulting point of EPM
OR
* `options` – the options returned by the solver (see `return_options`)
"""
function exact_penalty_method(
    M::AbstractManifold,
    F::TF,
    gradF::TGF;
    G::Function=(M, x) -> [],
    H::Function=(M, x) -> [],
    gradG::Function=(M, x) -> [],
    gradH::Function=(M, x) -> [],
    x=random_point(M),
    kwargs...,
) where {TF,TGF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return exact_penalty_method!(
        M, F, gradF; G=G, H=H, gradG=gradG, gradH=gradH, x=x_res, kwargs...
    )
end
@doc raw"""
    exact_penalty_method!(M, F, gradF; G, H, gradG, gradH)

perform the exact penalty method (EPM)[^LiuBoumal2020][^source_code]. The aim of the EPM is to find the solution of the [`ConstrainedProblem`](@ref)
```math
\begin{aligned}
\min_{x ∈\mathcal{M}} &f(x)\\
\text{subject to } &g_i(x)\leq 0 \quad ∀ i= 1, …, m,\\
\quad &h_j(x)=0 \quad ∀ j=1,…,p,
\end{aligned}
```
where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^p`` are twice continuously differentiable functions from `M` to ℝ.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF` – the gradient of the cost function
## Optional
* `G` – the inequality constraints
* `H` – the equality constraints
* `gradG` – the gradient of the inequality constraints
* `gradH` – the gradient of the equality constraints

For more options, especially `x` for the initial point and `smoothing_technique` for the smoothing technique, see [`exact_penalty_method`](@ref).
"""
function exact_penalty_method!(
    M::AbstractManifold,
    F::TF,
    gradF::TGF;
    G::Function=x -> [],
    H::Function=x -> [],
    gradG::Function=x -> [],
    gradH::Function=x -> [],
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
        ExactPenaltyCost(F, G, H, "linear_quadratic_huber", ρ, u),
        ExactPenaltyGrad(F, gradF, G, gradG, H, gradH, "linear_quadratic_huber", ρ, u),
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
    p = ConstrainedProblem(M, F, gradF, G, gradG, H, gradH)
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

mutable struct ExactPenaltyCost{P,T,R}
    constrained_problem::P
    smoothing_technique::T
    ρ::R
    u::R
end
function (L::ExactPenaltyCost)(M::AbstractManifold, x::P) where {P}
    inequality_constraints = get_inequality_constraints(L.contrained_problem, x)
    equality_constraints = get_equality_constraints(L.constrained_problem, x)
    num_inequality_constraints = length(inequality_constraints)
    num_equality_constraints = length(equality_constraints)

    # compute the cost functions of the constraints for the chosen smoothing technique
    if L.smoothing_technique == "log_sum_exp"
        if num_inequality_constraints != 0
            cost_ineq = sum(L.u .* log.(1 .+ exp.(inequality_constraints ./ L.u)))
        end
        if num_equality_constraints != 0
            cost_eq = sum(
                L.u .*
                log.(
                    exp.(equality_constraints ./ L.u) .+ exp.(-equality_constraints ./ L.u)
                ),
            )
        end
    elseif L.smoothing_technique == "linear_quadratic_huber"
        if num_inequality_constraints != 0
            cost_eq_greater_u = sum(
                (inequality_constraints .- L.u / 2) .* (inequality_constraints .> L.u)
            )
            cost_eq_pos_smaller_u = sum(
                (inequality_constraints .^ 2 ./ (2 * L.u)) .*
                ((inequality_constraints .> 0) .& (inequality_constraints .<= L.u)),
            )
            cost_ineq = cost_eq_greater_u + cost_eq_pos_smaller_u
        end
        if num_equality_constraints != 0
            cost_eq = sum(sqrt.(equality_constraints .^ 2 .+ L.u^2))
        end
    end

    # add up to the smoothed penalized objective
    if num_inequality_constraints != 0
        if num_equality_constraints != 0
            return get_cost(L.contrained_problem, x) + (L.ρ) * (cost_ineq + cost_eq)
        else
            return get_cost(L.contrained_problem, x) + (L.ρ) * cost_ineq
        end
    else
        if num_equality_constraints != 0
            return get_cost(L.contrained_problem, x) + (L.ρ) * cost_eq
        else
            return get_cost(L.contrained_problem, x)
        end
    end
end

mutable struct ExactPenaltyGrad{P,T,R}
    constrained_problem::P
    smoothing_technique::T
    ρ::R
    u::R
end
function (LG::ExactPenaltyGrad)(M::AbstractManifold, x::P) where {P}
    inequality_constraints = get_inequality_constraints(L.contrained_problem, x)
    equality_constraints = get_equality_constraints(L.constrained_problem, x)
    num_inequality_constraints = length(inequality_constraints)
    num_equality_constraints = length(equality_constraints)

    # compute the gradient functions of the constraints for the chosen smoothing technique
    if LG.smoothing_technique == "log_sum_exp"
        if num_inequality_constraints != 0
            coef =
                LG.ρ .* exp.(inequality_constraints ./ LG.u) ./
                (1 .+ exp.(inequality_constraints ./ LG.u))
            grad_ineq = sum(
                get_grad_equality_constraints(LG.constrained_problem, x) .* coef
            )
        end
        if num_equality_constraints != 0
            coef =
                LG.ρ .* (
                    exp.(equality_constraints ./ LG.u) .-
                    exp.(-equality_constraints ./ LG.u)
                ) ./
                (exp.(equality_constraints ./ LG.u) .+ exp.(-equality_constraints ./ LG.u))
            grad_eq = sum(
                get_grad_inequality_constraints(LG.constrained_problem, x) .* coef
            )
        end
    elseif LG.smoothing_technique == "linear_quadratic_huber"
        if num_inequality_constraints != 0
            # grad_inequality_constraints = LG.gradG(M, x)
            # grad_ineq_cost_greater_u = sum(grad_inequality_constraints .* ((inequality_constraints .>= 0) .& (inequality_constraints .>= LG.u)) .* LG.ρ)
            # grad_ineq_cost_smaller_u = sum(grad_inequality_constraints .* (inequality_constraints./LG.u .* ((inequality_constraints .>= 0) .& (inequality_constraints .< LG.u))) .* LG.ρ)
            # grad_ineq = grad_ineq_cost_greater_u + grad_ineq_cost_smaller_u
            grad_ineq = zero_vector(M, x)
            for i in 1:num_inequality_constraints
                coef = 0
                if inequality_constraints[i] >= 0
                    if inequality_constraints[i] >= LG.u
                        coef = 1
                    else
                        coef = inequality_constraints[i] / LG.u
                    end
                    grad_ineq .+=
                        (LG.ρ * coef) .*
                        get_grad_equality_constraint(LG.constrained_problem, x, i)
                end
            end
        end
        if num_equality_constraints != 0
            grad_eq = sum(
                get_grad_inequality_constraints(LG.constrained_problem, x) .*
                (equality_constraints ./ sqrt.(equality_constraints .^ 2 .+ LG.u^2)) .*
                LG.ρ,
            )
        end
    end

    # add up to the gradient of the smoothed penalized objective
    if num_inequality_constraints != 0
        if num_equality_constraints != 0
            return get_gradient(LG.constrained_problem, x) + grad_ineq + grad_eq
        else
            return get_gradient(LG.constrained_problem, x) + grad_ineq
        end
    else
        if num_equality_constraints != 0
            return get_gradient(LG.constrained_problem, x) + grad_eq
        else
            return get_gradient(LG.constrained_problem, x)
        end
    end
end
