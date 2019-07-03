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
        # Determine eta0
        if o.useRand == true
                # Pick the zero vector
                eta = zeroTVector(p.M, o.x)
        else
                # Random vector in T_x M (this has to be very small)
                eta = 10.0^(-6)*randomTVector(p.M, o.x)
                while norm(p.M, o.x, eta) > o.δ
                        # Must be inside trust-region
                        eta = sqrt(sqrt(eps(Float64)))*eta
                end
        end
        # Solve TR subproblem approximately
        η = truncatedConjugateGradient(p.M,p.costFunction,p.gradient,p.x,eta,p.hessian,p.precon,o.δ,o.stop,o.useRand)
        Hη = getHessian(p.M, o.x, η)
        # Initialize the cost function F und the gradient of the cost function
        # ∇F at the point x
        grad = getGradient(p, o.x)
        fx = getCost(p, o.x)
        # If using randomized approach, compare result with the Cauchy point.
        # Convergence proofs assume that we achieve at least (a fraction of)
        # the reduction of the Cauchy point. After this if-block, either all
        # eta-related quantities have been changed consistently, or none of
        # them have changed.
        if o.useRand == true
                # Check the curvature,
                Hgrad = getHessian(p, o.x, grad)
                gradHgrad = dot(p.M, o.x, grad, Hgrad)
                if gradHgrad <= 0
                        tau_c = 1
                else
                        tau_c = min( o.norm_grad^3 /(o.δ * gradHgrad), 1)
                end
                # and generate the Cauchy point.
                η_c = (-tau_c * o.δ / o.norm_grad) * grad
                Hη_c = (-tau_c * o.δ / o.norm_grad) * Hgrad
                # Now that we have computed the Cauchy point in addition to the
                # returned eta, we might as well keep the best of them.
                mdle  = fx + dot(p.M, o.x, grad, η) + .5 * dot(p.M, o.x, Hη, η)
                mdlec = fx + dot(p.M, o.x, grad, η_c) + .5 * dot(p.M, o.x, Hη_c, η_c)
                if mdlec < mdle
                        η = η_c
                        Hη = Hη_c
                end
        end

        norm_η = norm(p.M, o.x, η)
        # Compute the tentative next iterate (the proposal)
        x_prop  = retraction(p.M, o.x, η)
        # Compute the function value of the proposal
        fx_prop = getCost(p, x_prop)
        # Will we accept the proposal or not?
        # Check the performance of the quadratic model against the actual cost.
        ρnum = fx - fx_prop
        vecρ = grad + 0.5 * Hη
        ρden = -dot(p.M, o.x, η, vecρ)
        # rhonum could be anything.
        # rhoden should be nonnegative, as guaranteed by tCG, baring numerical
        # errors.
        # rhonum measures the difference between two numbers. Close to
        # convergence, these two numbers are very close to each other, so
        # that computing their difference is numerically challenging: there may
        # be a significant loss in accuracy. Since the acceptance or rejection
        # of the step is conditioned on the ratio between rhonum and rhoden,
        # large errors in rhonum result in a very large error in rho, hence in
        # erratic acceptance / rejection. Meanwhile, close to convergence,
        # steps are usually trustworthy and we should transition to a Newton-
        # like method, with rho=1 consistently. The heuristic thus shifts both
        # rhonum and rhoden by a small amount such that far from convergence,
        # the shift is irrelevant and close to convergence, the ratio rho goes
        # to 1, effectively promoting acceptance of the step.
        # he rationale is that close to convergence, both rhonum and rhoden
        # are quadratic in the distance between x and x_prop. Thus, when this
        # distance is on the order of sqrt(eps), the value of rhonum and rhoden
        # is on the order of eps, which is indistinguishable from the numerical
        # error, resulting in badly estimated rho's.
        # For abs(fx) < 1, this heuristic is invariant under offsets of f but
        # not under scaling of f. For abs(fx) > 1, the opposite holds. This
        # should not alarm us, as this heuristic only triggers at the very last
        # iterations if very fine convergence is demanded.
        ρ_reg = max(1, abs(fx)) * eps(Float64) * o.ρ_regularization
        ρnum = ρnum + ρ_reg
        ρden = ρden + ρ_reg
        # This is always true if a linear, symmetric operator is used for the
        # Hessian (approximation) and if we had infinite numerical precision.
        # In practice, nonlinear approximations of the Hessian such as the
        # built-in finite difference approximation and finite numerical
        # accuracy can cause the model to increase. In such scenarios, we
        # decide to force a rejection of the step and a reduction of the
        # trust-region radius. We test the sign of the regularized rhoden since
        # the regularization is supposed to capture the accuracy to which
        # rhoden is computed: if rhoden were negative before regularization but
        # not after, that should not be (and is not) detected as a failure.

        if ρden >= 0
                model_decreased = true
        else
                model_decreased = false
        end

        ρ = ρnum / ρden
        # Choose the new TR radius based on the model performance
        # If the actual decrease is smaller than 1/4 of the predicted decrease,
        # then reduce the TR radius.
        if ρ < 1/4 || ~model_decreased || isnan(ρ)
                o.δ = o.δ/4
        else ρ > 3/4
                o.δ = min(2*o.δ, o.δ_bar)
        end


        if model_decreased && ρ > o.ρ_prime
                accept = true
                o.x = x_prop
                fx = fx_prop # Probably not necessary
                grad = getGradient(p, o.x)
                o.norm_grad = norm(p.M, o.x, grad)
        else
                accept = false
        end

end

function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
end
