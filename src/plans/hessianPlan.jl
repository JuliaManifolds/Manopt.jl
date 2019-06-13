
struct HessianProblem <: Problem
    M::mT where mT <: Manifold
    costFunction::Function
    gradient::Function
    hessian::Union{Function,Missing}
    precon::Function
end

# injectivity_radius.

abstract type HessianOptions <: Options end

struct TruncatedConjugateGradientOptions <: HessianOptions
    x::P where {P <: MPoint}

end

struct TrustRegionOptions <: HessianOptions
    # x::P where {P <: MPoint}
    # is actually passed as a parameter of the function
    stop::stoppingCriterion
    # do we pass the limits to the criterion or do we pass them the options?
    tolgradnorm::Float64
    maxiter::Int64
    miniter::Int64
    maxtime::Float64
    mininner::Int64
    maxinner::Int64
    Delta_bar::Float64
    Delta0::Float64
    useRand::Bool
    kappa::Float64
    theta::Float64
    rho_prime::Float64
    rho_regularization::Float64
end

getHessian(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} = error("Not yet implemented")
