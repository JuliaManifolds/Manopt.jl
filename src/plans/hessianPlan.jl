
struct HessianProblem <: Problem
    M::mT where mT <: Manifold
    costFunction::Function
    gradient::Function
    hessian::Union{Function,Missing}
end

# injectivity_radius.

abstract type HessianOptions <: Options end

struct TruncatedConjugateGradientOptions <: HessianOptions
    x::P where {P <: MPoint}
end

struct TrustRegionOptions <: HessianOptions
    x::P where {P <: MPoint}
    stop::stoppingCriterion
end

getHessian(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} = error("Not yet implemented")

