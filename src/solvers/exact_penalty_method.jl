function exact_penalty_method(
    M::AbstractManifold,
    F::TF,
    gradF::TGF;
    G::Function=x->[], 
    H::Function=x->[],
    gradG::Function=x->[],
    gradH::Function=x->[],
    x=random_point(M), 
    sub_problem::Problem = GradientProblem(M,F,gradF),
    sub_options::Options = GradientDescentOptions(M,x),
    kwargs...,
) where {TF, TGF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return augmented_Lagrangian_method!(M, F, gradF; G=G, H=H, gradG=gradG, gradH=gradH, x=x_res, sub_problem=sub_problem, sub_options=sub_options,kwargs...)
end

function exact_penalty_method!(
    M::AbstractManifold,
    F::TF,
    gradF::TGF;
    G::Function=x->[],
    H::Function=x->[],
    gradG::Function=x->[],
    gradH::Function=x->[],
    x=random_point(M),
    smoothing_technique = log_sum_exp,
    sub_problem::Problem = GradientProblem(M,F,gradF),
    sub_options::Options = GradientDescentOptions(M,x),
    max_inner_iter::Int=200,
    num_outer_itertgn::Int=30,
    tolgradnorm::Real=1e-3, 
    ending_tolgradnorm::Real=1e-6,
    ϵ::Real=1e-1, 
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
function initialize_solver!(p::ConstrainedProblem, o::ALMOptions)
    o.θ_ϵ = (o.ϵ_min/o.ϵ)^(1/o.num_outer_itertgn)
    o.θ_tolgradnorm = (o.ending_tolgradnorm/o.tolgradnorm)^(1/o.num_outer_itertgn)
    return o
end
function step_solver!(p::ConstrainedProblem, o::ALMOptions, iter)
    # use subsolver to minimize the smoothed penalized function within a tolerance ϵ and with max_inner_iter
    cost = get_exact_penalty_cost_function(p, o) 
    grad = get_exact_penalty_gradient_function(p, o)
    o.x = gradient_descent(p.M, cost, grad, o.x, stepsize=ArmijoLinesearch(), stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.ϵ)))
    ######vgl stopping_criterion
    
    # get new evaluation of penalty
    max_violation = ...

    # update ρ if necessary
    if max_violation > o.ϵ 
        o.ρ = o.ρ/o.θ_ρ 
    end
    
   # update ϵ and tolgradnorm
    o.ϵ = max(o.ϵ_min, o.ϵ * o.θ_ϵ)
    o.tolgradnorm = max(o.ending_tolgradnorm, o.tolgradnorm * o.θ_tolgradnorm);
end
get_solver_result(o::ALMOptions) = o.x

function get_exact_penalty_cost_function(p::ConstrainedProblem, o::EPMOptions)
    cost = x -> get_cost(p, x)
    num_inequality_constraints = length(get_inequality_constraints(p,o.x))
    num_equality_constraints = length(get_equality_constraints(p,o.x))
    
    if o.smoothing_technique == log_sum_exp
        if num_inequality_constraints != 0 
            s = max(0, get_inequality_constraints(p, x))
            cost_ineq = x -> 
        end
        if num_equality_constraints != 0
            cost_eq = x -> 
        end
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
    elseif o.smoothing_technique == linear_quadratic_loss
        ...
    end
end

function get_exact_penalty_gradient_function(p::ConstrainedProblem, o::EPMOptions)
    if o.smoothing_technique == log_sum_exp
        ...
    elseif o.smoothing_technique == linear_quadratic_loss
        ...
    end
end

