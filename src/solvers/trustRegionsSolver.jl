#
#   Riemannian Trust-Tegions Solver For Optimization On Manifolds
#
export trustRegionsSolver

@doc doc"""
    trustRegionsSolver(M, F, ∂F, x, η, H, P)
"""

function trustRegionsSolver(M::mT,
        F::Function, ∂F::Function, x::MP, H::Union{Function,Missing},
        P::Function, stoppingCriterion::StoppingCriterion; δ_bar::Float64
        δ0::Float64, useRand::Bool, ρ_prime::Float64, ρ_regularization::Float64)
end
function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TrustRegionOptions}
end
function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
end
function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
end
