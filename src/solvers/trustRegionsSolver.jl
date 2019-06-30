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
        grad = getGradient(p, o.x)
        fx = getCost(p, o.x)

        if o.useRand == true
                used_cauchy = false
                Hgrad = getHessian(p, o.x, grad)
                gradHgrad = dot(p.M, o.x, grad, Hgrad)
                if gradHgrad <= 0
                        tau_c = 1
                else
                        tau_c = min( o.norm_grad^3 /(o.δ * gradHgrad), 1)
                end
                η_c = (-tau_c * o.δ / o.norm_grad) * grad
                Hη_c = (-tau_c * o.δ / o.norm_grad) * Hgrad
                mdle  = fx + dot(p.M, o.x, grad, η) + .5 * dot(p.M, o.x, Hη, η)
                mdlec = fx + dot(p.M, o.x, grad, η_c) + .5 * dot(p.M, o.x, Hη_c, η_c)
                if mdlec < mdle
                        η = η_c
                        Hη = Hη_c
                        used_cauchy = true
                end
        end

        norm_η = norm(p.M, o.x, η)

        x_prop  = retraction(p.M, o.x, η)
        fx_prop = getCost(p, x_prop)

        ρnum = fx - fx_prop
        vecρ = grad + 0.5 * Hη
        ρden = -dot(p.M, o.x, η, vecρ)

        ρ_reg = max(1, abs(fx)) * eps(Float64) * o.ρ_regularization
        ρnum = ρnum + ρ_reg
        ρden = ρden + ρ_reg

        if ρden >= 0
                model_decreased = true
        else
                model_decreased = false
        end

        ρ = ρnum / ρden

        if ρ < 1/4 || ~model_decreased || isnan(ρ)
                o.δ = o.δ/4
                consecutive_TRplus = 0
                consecutive_TRminus = consecutive_TRminus + 1
                if consecutive_TRminus >= 5
                        consecutive_TRminus = -Inf
                end
        elseif ρ > 3/4
                o.δ = min(2*o.δ, o.δ_bar)
                consecutive_TRminus = 0
                consecutive_TRplus = consecutive_TRplus + 1
                if consecutive_TRplus >= 5
                        consecutive_TRplus = -Inf
                end
        else
                consecutive_TRplus = 0
                consecutive_TRminus = 0
        end

        if model_decreased && ρ > o.ρ_prime
                accept = true
                o.x = x_prop
                fx = fx_prop
                grad = getGradient(p, o.x)
                o.norm_grad = norm(p.M, o.x, grad)
        else
                accept = false;
        end

end

function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
end
