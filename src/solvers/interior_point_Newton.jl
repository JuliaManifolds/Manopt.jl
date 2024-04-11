@doc raw"""
InteriorPointNewtonState <: AbstractManoptSolverState

# Fields

* `p`:                  point on M
* `μ`:                  Lagrange multiplier corresponding to inequality constrains
* `λ`:                  Lagrange multiplier corresponding to equality constraints
* `s`:                  slack variable
* `γ`:
* `σ`:
* `ρ`:
* `stopping_criterion`: stopping criterion
* `step_size`
* `retraction_method`:  retraction method

# Constructor

InteriorPointNewtonState( M::AbstractManifold,
                          co::ConstrainedManifoldObjective,
                          p = rand(M),
                          μ = ones(m),
                          λ = ones(n),
                          stopping_criterion = StopAfterIteration(300);
                          kwargs...
                        )
"""
mutable struct InteriorPointNewtonState{
    P,
    T,
    R<:Real,
    TStop<:StoppingCriterion,
    TStepsize<:Stepsize,
    TRTM<:AbstractRetractionMethod,
} <: AbstractManoptSolverState
    p::P
    μ::T
    λ::T
    s::T
    γ::R
    σ::R
    ρ::R
    stopping_criterion::TStop
    step_size::TStepsize
    retraction_method::TRTM
    function InteriorPointNewtonState{P,T}(
        M::AbstractManifold,
        p::P,
        μ::T,
        λ::T,
        s::T,
        γ::Real                                     = 0.5*rand() + 1,
        σ::Real                                     = rand(),
        ρ::Real                                     = 0.0,
        stopping_criterion::StoppingCriterion       = StopAfterIteration(100),
        step_size::Stepsize                         = default_stepsize(M, InteriorPointNewtonState),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    ) where {P,T}
        state = new{P, T, Real, typeof(stopping_criterion), typeof(step_size), typeof(retraction_method)}()
        state.p = p
        state.μ = μ
        state.λ = λ
        state.s = s
        state.γ = γ
        state.σ = σ
        state.ρ = ρ
        state.stopping_criterion = stopping_criterion
        state.step_size = step_size
        state.retraction_method = retraction_method
        return state
    end
end

function InteriorPointNewtonState(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective,
    p::P                                        = rand(M);
    μ::T                                        = ones( length( get_inequality_constraints(M, co, p) ) ),
    λ::T                                        = ones( length( get_equality_constraints(M, co, p) ) ),
    s::T                                        = ones( length( get_inequality_constraints(M, co, p) ) ),
    γ::Real                                     = 0.5*rand() + 1,
    σ::Real                                     = rand(),
    ρ::Real                                     = 0.0,
    stopping_criterion::StoppingCriterion       = StopAfterIteration(100),
    retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    step_size::Stepsize                         = default_stepsize(M, InteriorPointNewtonState; retraction_method = retraction_method),
) where {P,T}
    return InteriorPointNewtonState{P,T}(
        M, p, μ, λ, s, γ, σ, ρ, stopping_criterion, step_size, retraction_method
    )
end

function default_stepsize(
    M::AbstractManifold,
    ::Type{InteriorPointNewtonState};
    retraction_method = default_retraction_method(M),
)
    # take a default with a slightly defensive initial step size.
    return ArmijoLinesearch(M; retraction_method=retraction_method, initial_stepsize=1.0)
end

function is_feasible(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective,
    p
)
    # evaluate constraint functions at p
    gp = get_inequality_constraints(M, co, p)
    hp = get_equality_constraints(M, co, p)

    # check feasibility
    return all(gp .<= 0) && all(hp .== 0)
end

# function KKTVectorField(
#     M::AbstractManifold,
#     co::ConstrainedManifoldObjective,
#     p,
#     λ,
#     μ,
#     s
# )
#     gradL = LagrangianGrad(co, λ, μ)
#     gp = get_inequality_constraints(M, p)
#     hp = get_equality_constraints(M, p)
#     diagM = Diagonal(μ)
#     diagS = Diagonal(s)
#     e = ones(length(hp))
#     return [gradL, gp, hp+s, diagM*diagS*e]
# end

# function covariant_derivative_KKTVectorField(
#     M::AbstractManifold,
#     co::ConstrainedManifoldObjective,
#     p,
#     λ,
#     μ,
#     s
# )
#     L = LagrangianCost(co, λ, μ)
#     gp = get_equality_constraints(M, p)
#     hp = get_inequality_constraints(M, p)
#     diagM = Diagonal(μ)
#     diagS = Diagonal(s)
#     I     = ones()
#     e = ones(length(hp))


# function subsolver(
#     M::AbstractManifold,
#     co::ConstrainedManifoldObjective,
#     ipns::InteriorPointNewtonState,
# )
#     F = KKTVectorField(M, co, ipns.p, ipns.λ, ipns.μ, ipns.s)
#     # compute covariant derivative of kkt
#     # solve for X: ∇F[X] = -F + σ_k ρ_k ̂e
#     # return X

#     # The matrix representation of the covariant derivative
#     # of the KKT vector field is given by
#     #
#     #       | ∇^2 L   J_g^T   J_h^T   0 |
#     #  ∇F = |  J_g      0       0     0 |
#     #       |  J_h      0       0     I |
#     #       |   0       0       S     M |
# end

# function step_solver(M::AbstractManifold, co::ConstrainedManifoldObjective, ipns::InteriorPointNewtonState, iteration)

#     Δq = subsolver(M, co, ipns)
#     γ = (1-γ_hat)*rand() + γ_hat
#     α = min( 1, γ*min(...), γ*min(...) )
#     q_next = R(α*Δq)
#     β_next = β*rand()
#     ipns

#     return ipns


# Proto-RIPM(q_00, R, ̂γ, β_0)
# 1 k ← 0
# 2 while stopping criterion not satisfied
# 3   Solve for ∆qk :
#         ∇F(q_k)[∆q_k] = −F(q_k) + β_k ̂e
# 4   Choose γ_k ∈ [̂γ, 1]
# 5 α_k ← min { 1, γ_k min_i { -μ_k^i / Δμ_k^i : μ_k^i < 0 }, γ_k min_i { -s_k^i / Δs_k^i : s_k^i < 0 }}
# 6 q_{k+1} ← ̄R_{q_k}(α_k Δq_k)
# 7 Choose β_{k+1} ∈ (0, β_k)
# 8 k ← k + 1
