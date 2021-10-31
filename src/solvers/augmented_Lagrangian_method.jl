# Recreating Changshuo Liu's Matlab source code in Julia
# original code by Changshuo Liu: https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints/blob/master/solvers/almbddmultiplier.m
# uses Riemannian limited memory BFGS solver
# get n_ineq_constraint, n_eq_constraint

function augmented_Lagrangian_method(
    M::Manifold,
    F::TF,
    n_ineq_constraint::Int,
    n_eq_constraint::Int;
    x::random_point(M),
    kwargs...,
) where {TF}
    x_res = allocate(x)
    copyto!(Ref(M), x_res, x)
    return augmented_Lagrangian_method!(M, F, sub_problem, sub_options, n_ineq_constraint, n_eq_constraint; x=x_res, kwargs...)
end

function augmented_Lagrangian_method!(
    M::Manifold,
    F::TF,
    sub_problem,
    sub_options,
    n_ineq_constraint::Int,
    n_eq_constraint::Int;
    x::random_point(M),
    max_inner_iter::Int=200,
    num_outer_itertgn::Int=30,
    ϵ::Real=1e-3, #(starting)tolgradnorm
    ϵ_min::Real=1e-6, #endingtolgradnorm
    bound::Int=20, 
    λ::Vector=ones(n_ineq_constraint),
    γ::Vector=ones(n_eq_constraint),
    ρ::Real=1.0, 
    τ::Real=0.8,
    θ_ρ::Real=0.3, 
    θ_ϵ::Real=(ϵ_min/ϵ)^(1/num_outer_itertgn), 
    oldacc::Real=Inf, 
    stopping_criterion::StoppingCriterion=StopWhenAny(StopAfterIteration(300), StopWhenAll(StopIfSmallerOrEqual(ϵ, ϵ_min), StopWhenChangeLess(1e-6))), 
    kwargs...,
) where {TF}
    p = CostProblem(M, F, n_ineq_constraint, n_eq_constraint)
    o = RALMOptions(
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
###x0
old_acc=Inf, 
end
function step_solver!(p::CostProblem, o::ALMOptions, iter)
    # use subsolver to minimize the augmented Lagrangian within a tolerance ϵ and with max_inner_iter
    cost = @get_Lagrangian_cost(p, o) ### how to not asign the return value of the function, but the function as a whole here?
    grad = @get_Lagrangian_gradient(p, o)
    ### put these in the subproblem
    ###o.x=

    # update multipliers
    cost_ineq = get_cost_ineq(p, o.x)
    λ = min.(ones(n_ineq_constraint)* o.bound, max.(λ + o.ρ .* cost_ineq, zeros(n_ineq_constraint)))
    cost_eq = get_cost_eq(p, o.x)
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

function get_Lagrangian_cost(p::CostProblem, o::ALMOptions) #als Funktion
    cost = get_cost(p, o.x)
    cost_ineq = sum(max.(zeros(o.n_ineq), o.λ ./ o.ρ .+ get_cost_ineq(p, o.x)))
    cost_eq = sum((get_cost_eq(p, o.x) .+ o.γ./o.ρ)^2)
    return cost + (o.ρ/2) * (cost_ineq + cost_eq)
end

function get_Lagrangian_gradient(p::CostProblem, o::ALMOptions) #als Funktion
    grad = get_gradient(p, o.x)
    cost_shifted_ineq = get_cost_ineq(p, o.x) + o.λ./o.ρ 
    index = findall(x -> x > 0, cost_shifted_ineq) #gets the index of all elements greater 0
    grad_ineq = get_grad_ineq(p, o.x)
    ### transform with egrad2rgrad
    grad_ineq = grad_ineq[index]
      
    cost_eq = get_cost_eq(p, o.x)
    grad_eq = get_grad_eq(p, o.x)
    ### transform with egrad2rgrad
    return ### linear combination
end