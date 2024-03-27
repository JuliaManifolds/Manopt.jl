# @doc raw
# """
# InteriorPointNewtonState <: AbstractManoptSolverState

# # Fields

# * `p`:                  position on M
# * `λ`:                  Lagrange multiplier corresponding to equality constraints
# * `μ`:                  Lagrange multiplier corresponding to inequality constrains 
# * `s`:                  slack variable
# * `stopping_criterion`: stopping criterion

# # Constructor

# InteriorPointNewtonState( M::AbstractManifold, 
#                           co::ConstrainedManifoldObjective, 
#                           p = rand(M),
#                           λ = ones(m),
#                           μ = ones(n),
#                           stopping_criterion = StopAfterIteration(300); 
#                           kwargs...
#                         )
# """

mutable struct InteriorPointNewtonState{
    P,
    T,
    R<:Real,
    TStop<:StoppingCriterion,
    TStepsize<:Stepsize,
    TRTM<:AbstractRetractionMethod,
} <: AbstractManoptSolverState
    p::P
    λ::T
    μ::T
    s::T
    γ::R
    σ::R
    ρ::R
    stop::TStop
    stepsize::TStepsize
    retraction_method::TRTM
    function InteriorPointNewtonState{P,T}(
        M::AbstractManifold,
        p::P,
        λ::T,
        μ::T,
        s::T,
        γ::Real                                     = 0.5*rand() + 1,
        σ::Real                                     = rand(),
        ρ::Real                                     = 0.0,
        stop::StoppingCriterion                     = StopAfterIteration(100),
        step::Stepsize                              = default_stepsize(M, InteriorPointNewtonState),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    ) where {P,T}
        state = new{P, T, Real, typeof(stop), typeof(step), typeof(direction), typeof(retraction_method)}()
        state.p = p
        state.λ = λ
        state.μ = μ
        state.s = s
        state.γ = γ
        state.σ = σ
        state.ρ = ρ
        state.stop = stop
        state.stepsize = step
        state.retraction_method = retraction_method
        return state
    end
end

function InteriorPointNewtonState(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective,
    p::P                                        = rand(M);
    λ::T                                        = ones( length( get_equality_constraints(M, co, p) ) ),
    μ::T                                        = ones( length( get_inequality_constraints(M, co, p) ) ),
    s::T                                        = ones( length( get_inequality_constraints(M, co, p) ) ),
    γ::Real                                     = 0.5*rand() + 1,
    σ::Real                                     = rand(),
    ρ::Real                                     = 0.0,
    stopping_criterion::StoppingCriterion       = StopAfterIteration(100),
    retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    stepsize::Stepsize                          = default_stepsize(M, InteriorPointNewtonState; retraction_method = retraction_method),
) where {P,T}
    return InteriorPointNewtonState{P,T}(
        M, p, λ, μ, s, γ, σ, ρ, stopping_criterion, stepsize, direction, retraction_method
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
    gp = get_equality_constraints(M, co, p)
    hp = get_inequality_constraints(M, co, p)

    # check feasibility
    return all(gp .== 0) && all(hp .<= 0)
end

function KKTVectorField(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective,
    p,
    λ,
    μ,
    s
)
    gradL = LagrangianGrad(co, λ, μ)
    gp = get_equality_constraints(M, p)
    hp = get_inequality_constraints(M, p)
    diagM = Diagonal(μ)
    diagS = Diagonal(s)
    e = ones(length(hp))
    return [gradL, gp, hp+s, diagM*diagS*e]
end

function subsolver(
    M::AbstractManifold, 
    co::ConstrainedManifoldObjective,
    ipns::InteriorPointNewtonState,
    X
)
    F = KKTVectorField(M, co, ipns.p, ipns.λ, ipns.μ, ipns.s)
    # compute covariant derivative of kkt
    # solve for X: ∇F[X] = -F + σ_k ρ_k ̂e
    # return X

    # The matrix representation of the covariant derivative
    # of the KKT vector field is given by
    #      
    #       | ∇^2 L   J_g^T   J_h^T   0 |
    #  ∇F = |  J_g      0       0     0 |
    #       |  J_h      0       0     I |
    #       |   0       0       S     M |
end
