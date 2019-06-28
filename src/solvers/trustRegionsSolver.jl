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
        δ_bar::Float64 = try injectivity_radius(M) catch; sqrt(manifoldDimension(M)) end,
        δ0::Float64 = δ_bar/8,
        uR::Bool = false, ρ_prime::Float64 = 0.1,
        ρ_regularization::Float64=10^(-3)
        ) where {mT <: Manifold, MP <: MPoint, T <: TVector}
    p = HessianProblem(M,F,∂F,H,P)
    o = TrustRegionOptions(x,stoppingCriterion,δ0,δ_bar,δ0,uR,ρ_prime,ρ_regularization,0)

    resultO = solve(p,o)
    if hasRecord(resultO)
        return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
    end
    return getSolverResult(p,resultO)
end

function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TrustRegionOptions}
        o.norm_grad = norm(p.M, o.x, getGradient(p, o.x))
end

function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
        if o.useRand == true
                eta = zeroTVector(p.M, o.x)
        else
                eta = 10.0^(-6)*randomTVector(p.M, o.x)
                while norm(p.M, o.x, eta) > o.δ
                        eta = sqrt(sqrt(eps(Float64)))*eta
                end
        end
        η = truncatedConjugateGradient(p.M,p.costFunction,p.gradient,p.x,eta,p.hessian,p.precon,o.δ,o.stop,o.useRand)
        Hη = getHessian(p.M, o.x, η)
end

function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
end
