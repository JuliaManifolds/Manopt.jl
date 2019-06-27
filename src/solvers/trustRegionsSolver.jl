#
#   Riemannian Trust-Tegions Solver For Optimization On Manifolds
#
export trustRegionsSolver

@doc doc"""
    trustRegionsSolver(M, F, ∂F, x, η, H, P)
"""

function trustRegionsSolver(M::mT,
        F::Function, ∂F::Function,
        x::MP = randomMPoint(M),
        H::Union{Function,Missing}, P::Function;
        stoppingCriterion::StoppingCriterion = stopAfterIteration(5000),
        δ_bar::Float64 = injectivity_radius(M),
        δ0::Float64 = δ_bar/8,
        useRand::Bool = false, ρ_prime::Float64 = 0.1,
        ρ_regularization::Float64=10^(-3)
        ) where {mT <: Manifold, MP <: MPoint, T <: TVector}
    p = HessianProblem(M,F,∂F,H,P)
    o = TrustRegionOptions(x,stoppingCriterion,δ_bar,δ0,useRand,ρ_prime,ρ_regularization)

    resultO = solve(p,o)
    if hasRecord(resultO)
        return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
    end
    return getSolverResult(p,resultO)
end
function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TrustRegionOptions}
end
function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
end
function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
end
