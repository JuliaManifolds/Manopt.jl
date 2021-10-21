# Recreating Changshuo Liu's Matlab source code in Julia
# original code by Changshuo Liu: https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints/blob/master/solvers/almbddmultiplier.m
# uses Riemannian limited memory BFGS solver
# get n_ineq_constraint, n_eq_constraint

function Riemannian_augmented_Lagrangian_method(
    M::Manifold,
    F::TF,
    n_ineq_constraint::Int,
    n_eq_constraint::Int;
    x::random_point(M),
    kwargs...,
) where {TF}
    x_res = allocate(x)
    copyto!(Ref(M), x_res, x)
    return particle_swarm!(M, F, n_ineq_constraint, n_eq_constraint; x=x_res, kwargs...)
end

function Riemannian_augmented_Lagrangian_method!(
    M::Manifold,
    F::TF,
    n_ineq_constraint::Int,
    n_eq_constraint::Int;
    x::random_point(M),
    max_inner_iter::Int=200,
    num_outer_itertgn::Int=30,
    ϵ::Real=1e-3, #(starting)tolgradnorm
    ϵ_min::Real=1e-6, #endingtolgradnorm
    bound::Int=20, ###why not Real?
    λ::Vector=ones(n_ineq_constraint,1),
    γ::Vector=ones(n_eq_constraint,1),
    ρ::Real=1.0, ###why int in Matlab code?
    τ::Real=0.8,
    θ_ρ::Real=0.3, 
    θ_ϵ::Real=(ϵ_min/ϵ)^(1/num_outer_itertgn), ###this does not need to be a parameter, just defined somewhere
    oldacc::Real=Inf, ###this does not need to be a parameter, just defined somewhere
    min_stepsize::Real= ### find realistic value, put in stopping criterion alongside the ϵ condition
    stopping_criterion::StoppingCriterion==StopWhenAny(StopAfterIteration(300), StopWhenAll()), #maxOuterIter
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
function initialize_solver!(p::CostProblem, o::RALMOptions)
###
end
function step_solver!(p::CostProblem, o::RALMOptions, iter)
    # use subsolver(Riemannian limited memory BFGS) to minimize the augmented Lagrangian within a tolerance ϵ and with max_inner_iter
    
    ###o.x=

    # update multipliers
    newacc=0
    for i in 1:o.n_ineq_constraint
        ### how can I access the cost of the inequality (cost_iter)?
        newacc = max(newacc, abs(max(-λ[i]/ρ, cost_iter)))
        λ[i]=min(o.bound, max(λ[i] + o.ρ * cost_iter, 0))
    end 
    for j in 1:o.n_eq_constraint
        ### how can I access the cost of the equality (cost_iter)?
        newacc = max(newacc, abs(cost_iter))
        γ[j] = min(o.bound, max(-o.bound, γ[j] + o.ρ * cost_iter))
    end

    # update ρ if necessary
    if iter == 1 || newacc > o.τ * o.oldacc ###iter == OuterIter?
        o.ρ = o.ρ/o.θ_ρ 
    end
    o.oldacc = newacc

    # update the tolerance ϵ
    ϵ = max(o.ϵ_min, ϵ * o.θ_ϵ)



end
get_solver_result(o::RALMOptions) = o.x

function cost_alm(p::CostProblem, o::RALMOptions)
    val=get_cost(p, o.x)
    
end

function constraints_detail(p::CostProblem)###put this in separate file, but where does it belong?
    mutable struct condet 
            has_ineq_cost::Bool
            has_ineq_grad::Bool
            has_eq_cost::Bool
            has_eq_grad::Bool
            n_ineq_constraint_cost::Int
            n_ineq_constraint_grad::Int
            n_eq_constraint_cost::Int
            n_eq_constraint_grad::Int
        end

        # check if problem has conditions
        fields = fieldnames(p)
        if "ineq_constraint_cost" in fields
            condet.has_ineq_cost=true
        else
            condet.has_ineq_cost=false
        if "ineq_constraint_grad" in fields
            condet.has_ineq_grad=true
        else
            condet.has_ineq_grad=false

        if "eq_constraint_cost" in fields
            condet.has_eq_cost=true
        else
            condet.has_eq_cost=false
        if "eq_constraint_grad" in fields
            condet.has_eq_grad=true
        else
            condet.has_eq_grad=false
        
        # in case problem does have conditions, count how many
        if condet.has_ineq_cost
            condet.n_ineq_constraint_cost  = length(problem.ineq_constraint_cost)
        else
            condet.n_ineq_constraint_cost = 0
        end
        if condet.has_ineq_grad
            condet.n_ineq_constraint_grad  = length(problem.ineq_constraint_grad)
        else
            condet.n_ineq_constraint_grad  = 0
        end
        
        if condet.has_eq_cost
            condet.n_eq_constraint_cost  = length(problem.eq_constraint_cost)
        else
            condet.n_eq_constraint_cost = 0
        end
        if condet.has_eq_grad
            condet.n_eq_constraint_grad  = length(problem.eq_constraint_grad)
        else 
            condet.n_eq_constraint_grad = 0
        end

        # give a warning if the number of cost and grad functions are not equal
        if condet.n_ineq_constraint_cost != condet.n_ineq_constraint_grad
            #throw a warning
        end
        
        if condet.n_eq_constraint_cost != condet.n_eq_constraint_grad
            #throw a warning
        end

    return condet
end