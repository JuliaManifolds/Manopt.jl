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
* `tolgradnorm` – (`1e–3`) the accuracy tolerance
* `ending_tolgradnorm` – (`1e-6`) the lower bound for the accuracy tolerance
* `ϵ` – (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `ϵ_min` – (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ` – (`1.0`) the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `θ_ϵ` – (`(ϵ_min/ϵ)^(1/num_outer_itertgn)`) the scaling factor of the smoothing parameter and threshold for violation of the constraints
* `θ_tolgradnorm` – (`(ending_tolgradnorm/tolgradnorm)^(1/num_outer_itertgn)`) the scaling factor of the accuracy tolerance
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(tolgradnorm, ending_tolgradnorm), `[`StopWhenChangeLess`](@ref)`(1e-6)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
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
    G::Function=x->[], 
    H::Function=x->[],
    gradG::Function=x->[],
    gradH::Function=x->[],
    x=random_point(M), 
    smoothing_technique::String = "log_sum_exp",
    sub_problem::Problem = GradientProblem(M,F,gradF),
    sub_options::Options = GradientDescentOptions(M,x),
    kwargs...,
) where {TF, TGF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return exact_penalty_method!(M, F, gradF; G=G, H=H, gradG=gradG, gradH=gradH, x=x_res, smoothing_technique=smoothing_technique, sub_problem=sub_problem, sub_options=sub_options,kwargs...)
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
    G::Function=x->[],
    H::Function=x->[],
    gradG::Function=x->[],
    gradH::Function=x->[],
    x=random_point(M),
    smoothing_technique = "log_sum_exp",
    sub_problem::Problem = GradientProblem(M,F,gradF),
    sub_options::Options = GradientDescentOptions(M,x),
    max_inner_iter::Int=200,
    num_outer_itertgn::Int=30,
    tolgradnorm::Real=1e-3, 
    ending_tolgradnorm::Real=1e-6,
    ϵ::Real=1e-1,           # smoothing parameter u and threshold τ
    ϵ_min::Real=1e-6,
    ρ::Real=1.0, 
    θ_ρ::Real=0.3, 
    stopping_criterion::StoppingCriterion=StopWhenAny(StopAfterIteration(300), StopWhenAll(StopWhenSmallerOrEqual(:tolgradnorm, ending_tolgradnorm), StopWhenChangeLess(1e-6))), 
    #### look into minstepsize again
    return_options=false,
    kwargs...,
) where {TF, TGF}
    p = ConstrainedProblem(M, F, gradF, G, gradG, H, gradH)
    o = EPMOptions(
        M,
        p,
        x,
        smoothing_technique,
        sub_problem,
        sub_options;
        max_inner_iter = max_inner_iter,
        num_outer_itertgn = num_outer_itertgn,
        tolgradnorm = tolgradnorm, 
        ending_tolgradnorm = ending_tolgradnorm,
        ϵ = ϵ,
        ϵ_min = ϵ_min,
        ρ = ρ,
        θ_ρ = θ_ρ,
        stopping_criterion = stopping_criterion,
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
function initialize_solver!(p::ConstrainedProblem, o::EPMOptions)
    o.θ_ϵ = (o.ϵ_min/o.ϵ)^(1/o.num_outer_itertgn)
    o.θ_tolgradnorm = (o.ending_tolgradnorm/o.tolgradnorm)^(1/o.num_outer_itertgn)
    return o
end
function step_solver!(p::ConstrainedProblem, o::EPMOptions, iter)
    # use subsolver to minimize the smoothed penalized function within a tolerance ϵ and with max_inner_iter
    cost = get_exact_penalty_cost_function(p, o) 
    grad = get_exact_penalty_gradient_function(p, o)
    o.x = gradient_descent(p.M, cost, grad, o.x, stepsize=ArmijoLinesearch(), stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.tolgradnorm)))
    ######add minstepsize to stopping criteria of subsolver in both methods 
    
    # get new evaluation of penalty
    cost_ineq = get_inequality_constraints(p, o.x)
    n_ineq_constraint = length(cost_ineq)
    cost_eq = get_equality_constraints(p, o.x)
    n_eq_constraint = length(cost_eq)
    if n_ineq_constraint == 0
        cost_ineq = 0
    end
    if n_eq_constraint == 0
        cost_eq = 0
    end
    max_violation = max(max(maximum(cost_ineq),0),maximum(abs.(cost_eq)))

    # update ρ if necessary
    if max_violation > o.ϵ 
        o.ρ = o.ρ/o.θ_ρ 
    end
    
   # update ϵ and tolgradnorm
    o.ϵ = max(o.ϵ_min, o.ϵ * o.θ_ϵ)
    o.tolgradnorm = max(o.ending_tolgradnorm, o.tolgradnorm * o.θ_tolgradnorm);
end
get_solver_result(o::EPMOptions) = o.x

function get_exact_penalty_cost_function(p::ConstrainedProblem, o::EPMOptions)
    cost = x -> get_cost(p, x)
    num_inequality_constraints = length(get_inequality_constraints(p,o.x))
    num_equality_constraints = length(get_equality_constraints(p,o.x))
    
    # compute the cost functions of the constraints for the chosen smoothing technique
    if o.smoothing_technique == "log_sum_exp"
        if num_inequality_constraints != 0 
            s = (p, x) -> max.(0, get_inequality_constraints(p, x))
            cost_ineq = x -> sum(s(p, x) .+ o.ϵ .* log.( exp.((get_inequality_constraints(p, x) .- s(p, x))./o.ϵ) + exp.(-s(p, x)./o.ϵ)))
        end ### why is s used like that?
        if num_equality_constraints != 0
            s = (p, x) -> max.(-get_equality_constraints(p, x), get_equality_constraints(p, x))
            cost_eq = x -> sum(s(p, x) .+ o.ϵ .* log.( exp.((get_equality_constraints(p, x) .- s(p, x))./o.ϵ) .+ exp.((-get_equality_constraints(p, x) .- s(p, x))./o.ϵ)))
        end ### why is s used like that?
    elseif o.smoothing_technique == "linear_quadratic_huber"
        if num_inequality_constraints != 0 
            cost_eq_greater_ϵ = x -> sum((get_inequality_constraints(p, x) .- o.ϵ/2) .* (get_inequality_constraints(p, x) .> o.ϵ))
            cost_eq_pos_smaller_ϵ = x -> sum( (get_inequality_constraints(p, x).^2 ./(2*o.ϵ)) .* ((get_inequality_constraints(p, x) .> 0) .& (get_inequality_constraints(p, x) .<= o.ϵ)))
            cost_ineq = x -> cost_eq_greater_ϵ(x) + cost_eq_pos_smaller_ϵ(x)
        end 
        if num_equality_constraints != 0
            cost_eq = x -> sum(sqrt.(get_equality_constraints(p, x).^2 .+ o.ϵ^2))
        end 
    end

    # add up to the smoothed penalized objective
    if num_inequality_constraints != 0
        if num_equality_constraints != 0
            return (M,x) -> cost(x) + (o.ρ) * (cost_ineq(x) + cost_eq(x))
        else
            return (M,x) -> cost(x) + (o.ρ) * cost_ineq(x)
        end
    else
        if num_equality_constraints != 0
            return (M,x) -> cost(x) + (o.ρ) * cost_eq(x)
        else
            return (M,x) -> cost(x) 
        end
    end
end

function get_exact_penalty_gradient_function(p::ConstrainedProblem, o::EPMOptions)
    grad = x -> get_gradient(p, x)
    num_inequality_constraints = length(get_inequality_constraints(p,o.x))
    num_equality_constraints = length(get_equality_constraints(p,o.x))

    # compute the gradient functions of the constraints for the chosen smoothing technique
    if o.smoothing_technique == "log_sum_exp"
        if num_inequality_constraints != 0 
            s = (p, x) -> max.(0, get_inequality_constraints(p, x))
            coef = (p, x) -> o.ρ .* exp.((get_inequality_constraints(p, x).-s(p, x))./o.ϵ) ./ (exp.((get_inequality_constraints(p, x).-s(p, x))./o.ϵ) .+ exp.(-s(p, x) ./ o.ϵ))
            grad_ineq = x -> sum(get_grad_ineq(p, x) .* coef(p, x)) 
        end
        if num_equality_constraints != 0
            s = (p, x) -> max.(-get_equality_constraints(p, x), get_equality_constraints(p, x))
            coef = (p, x) -> o.ρ .* (exp.((get_equality_constraints(p, x) .- s(p, x)) ./o.ϵ) .- exp.((-get_equality_constraints(p, x) .- s(p, x)) ./o.ϵ)) ./ (exp.((get_equality_constraints(p, x) .- s(p, x)) ./o.ϵ) .+ exp.((-get_equality_constraints(p, x) .- s(p, x)) ./o.ϵ))
            grad_eq = x-> sum(get_grad_eq(p, x) .* coef(p, x))
        end
    elseif o.smoothing_technique == "linear_quadratic_huber"
        if num_inequality_constraints != 0
            grad_ineq_cost_greater_ϵ = x -> sum(get_grad_ineq(p, x) .* ((get_inequality_constraints(p, x) .>= 0) .& (get_inequality_constraints(p, x) .>= o.ϵ)) .* o.ρ)
            grad_ineq_cost_smaller_ϵ = x -> sum(get_grad_ineq(p, x) .* (get_inequality_constraints(p, x)./o.ϵ .* ((get_inequality_constraints(p, x) .>= 0) .& (get_inequality_constraints(p, x) .< o.ϵ))) .* o.ρ)
            grad_ineq = x -> grad_ineq_cost_greater_ϵ(x) + grad_ineq_cost_smaller_ϵ(x) 
        end
        if num_equality_constraints != 0
            grad_eq = x-> sum(get_grad_eq(p, x) .* (get_inequality_constraints(p, x)./sqrt.(get_inequality_constraints(p, x).^2 .+ o.ϵ^2)) .* o.ρ) 
        end
    end

    # add up to the gradient of the smoothed penalized objective
    if num_inequality_constraints != 0
        if num_equality_constraints != 0
            return (M,x) -> grad(x) + grad_ineq(x) + grad_eq(x)
        else
            return (M,x) -> grad(x) + grad_ineq(x)
        end
    else
        if num_equality_constraints != 0
            return (M,x) -> grad(x) + grad_eq(x)
        else
            return (M,x) -> grad(x) 
        end
    end
end

