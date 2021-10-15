# Recreating Changshuo Liu's Matlab source code in Julia
# uses Riemannian limited memory BFGS solver
# get n_ineq_constraint, n_eq_constraint

function Riemannian_augmented_Lagrangian_method(
    M::Manifold,
    F::TF,
    x::random_point(M),
    n_ineq_constraint::Int,
    n_eq_constraint::Int;
    max_inner_iter::Int=200,
    num_outer_itertgn::Int=30,
    ϵ::Real=1e-3, #(starting)tolgradnorm
    ϵ_min::Real=1e-6, #endingtolgradnorm
    bound::Int=20, ###why not Real?
    λ::Vector=ones(n_ineq_constraint,1),
    γ::Vector=ones(n_eq_constraint,1),
    ρ::Real=1.0, ###why int in Matlab?
    τ::Real=0.8,
    θ_ρ::Real=0.3, 
    θ_ϵ::Real=(ϵ_min/ϵ)^(1/num_outer_itertgn), ###this does not need to be a parameter, just defined somewhere
    oldacc=Inf, ###this does not need to be a parameter, just defined somewhere
    stopping_criterion::StoppingCriterion=StopAfterIteration(300),
) where {TF}
    x_res = allocate(x)
    copyto!(Ref(M), x_res, x)
    return particle_swarm!(M, F; kwargs...)
end

function Riemannian_augmented_Lagrangian_method!(
    M::Manifold,
    F::TF,
    x0::?;
    #x0::?=random_point(M),
    λ::Vector=ones(n_ineq_constraint,1),
    γ::Vector=ones(n_eq_constraint,1),
    ρ::Real=1.0,
    θ_ρ::Real=0.3, #>1 im paper?
    stopping_criterion::StoppingCriterion=StopAfterIteration(300),
    kwargs...,
) where {TF}
    p = CostProblem(M, F)
    o = RALMOptions(
        #x0,
        λ,
        γ,
        ρ,
        θ_ρ,
        stopping_criterion
        return particle_swarm!(M, F; kwargs...)
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

end
function step_solver!(p::CostProblem, o::RALMOptions, iter)
    # use subsolver(Riemannian limited memory BFGS) to minimize the augmented Lagrangian within a tolerance ϵ and with max_inner_iter

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
    ϵ = max(o.ϵ_min, ϵ * o.θ_ϵ);

end
get_solver_result(o::RALMOptions) = o.x