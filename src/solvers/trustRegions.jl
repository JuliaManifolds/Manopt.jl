#
#   Riemannian Trust-Tegions Solver For Optimization On Manifolds
#
using Manopt
import Base: identity
export trustRegions

@doc doc"""
    trustRegions(M, F, ∇F, x, H)

evaluate the Riemannian trust-regions solver for optimization on manifolds.
It will attempt to minimize the cost function F on the Manifold M.
If no Hessian H is provided, a standard approximation of the Hessian based on
the gradient ∇F will be computed.
For solving the the inner trust-region subproblem of finding an update-vector,
it uses the Steihaug-Toint truncated conjugate-gradient method.
For a description of the algorithm and theorems offering convergence guarantees,
see the reference:

* [ABG07] P.-A. Absil, C.G. Baker, K.A. Gallivan,
        Trust-region methods on Riemannian manifolds, FoCM, 2007.
* [AMS08] P.-A. Absil, R. Mahony and R. Sepulchre,
        Optimization Algorithms on Matrix Manifolds, Princeton University Press,
        2008.
* [CGT2000] A. R. Conn, N. I. M. Gould, P. L. Toint, Trust-region methods, SIAM,
        MPS, 2000.

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F \colon \mathcal M \to \mathbb R$ to minimize
* `∇F`- the gradient $\nabla F \colon \mathcal M \to T \mathcal M$ of F
* `x` – an initial value $x \in \mathcal M$
* `H` – the hessian $H( \mathcal M, x, \xi)$ of $F$

# Optional
* `P` – a preconditioner (a symmetric, positive definite operator that should
        approximate the inverse of the Hessian)
* `stoppingCriterion` – (`[`stopWhenAny`](@ref)`(`[`stopAfterIteration`](@ref)`(5000))
        a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `Δ_bar` – the maximum trust-region radius
* `Δ` - the (initial) trust-region radius
* `useRandom` – set to true if the trust-region solve is to be initiated with a
        random tangent vector. If set to true, no preconditioner will be
        used. This option is set to true in some scenarios to escape saddle
        points, but is otherwise seldom activated.
* `ρ_prime` – Accept/reject threshold: if ρ (the performance ratio for the
        iterate) is at least ρ_prime, the outer iteration is accepted.
        Otherwise, it is rejected. In case it is rejected, the trust-region
        radius will have been decreased. To ensure this, ρ_prime >= 0 must be
        strictly smaller than 1/4. If ρ_prime is negative, the algorithm is not
        guaranteed to produce monotonically decreasing cost values. It is
        strongly recommended to set ρ_prime > 0, to aid convergence.
* `ρ_regularization` – Close to convergence, evaluating the performance ratio ρ
        is numerically challenging. Meanwhile, close to convergence, the
        quadratic model should be a good fit and the steps should be
        accepted. Regularization lets ρ go to 1 as the model decrease and
        the actual decrease go to zero. Set this option to zero to disable
        regularization (not recommended). When this is not zero, it may happen
        that the iterates produced are not monotonically improving the cost
        when very close to convergence. This is because the corrected cost
        improvement could change sign if it is negative but very small.

# Output
* `x` – the last reached point on the manifold

# see also
[`truncatedConjugateGradient.jl`](@ref)
"""
function trustRegions(M::mT,
        F::Function, ∇F::Function,
        x::MP, H::Union{Function,Missing};
        preconditioner::Function = (M,x,ξ) -> ξ,
        stoppingCriterion::StoppingCriterion = stopWhenAny(
        stopAfterIteration(1000), stopWhenGradientNormLess(10^(-6))),
        Δ_bar::Float64 = sqrt(manifoldDimension(M)),
        Δ::Float64 = Δ_bar/8,
        useRandom::Bool = false, ρ_prime::Float64 = 0.1,
        ρ_regularization::Float64=1000.
        ,kwargs... #collect rest
        ) where {mT <: Manifold, MP <: MPoint, T <: TVector}

        if ρ_prime >= 0.25
                throw( ErrorException("ρ_prime must be strictly smaller than 0.25 but it is $ρ_prime.") )
        end

        if Δ_bar <= 0
                throw( ErrorException("Δ_bar must be positive but it is $Δ_bar.") )
        end

        if Δ <= 0 || Δ > Δ_bar
                throw( ErrorException("Δ must be positive and smaller than Δ_bar (=$Δ_bar) but it is $Δ.") )
        end

        p = HessianProblem(M,F,∇F,H,preconditioner)

        o = TrustRegionOptions(x,stoppingCriterion,Δ,Δ_bar,useRandom,ρ_prime,ρ_regularization)

        o = decorateOptions(o; kwargs...)
        resultO = solve(p,o)
        if hasRecord(resultO)
                return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
        end
        return getSolverResult(p,resultO)
end

function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TrustRegionOptions}
end

function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionOptions}
        # Determine eta0
        if o.useRand==false
                # Pick the zero vector
                eta = zeroTVector(p.M, o.x)
        else
                # Random vector in T_x M (this has to be very small)
                eta = 10.0^(-6)*randomTVector(p.M, o.x)
                while norm(p.M, o.x, eta) > o.Δ
                        # Must be inside trust-region
                        eta = sqrt(sqrt(eps(Float64)))*eta
                        # print("normetapre = $(norm(p.M, o.x, eta))\n")
                end
        end
        # Solve TR subproblem approximately
        (η, option) = truncatedConjugateGradient(p.M,p.costFunction,p.gradient,
        o.x,eta,p.hessian,o.Δ;preconditioner=p.precon,useRandom=o.useRand,
        debug = [:Iteration," ",:Stop])
        #print("η = $η\n")
        SR = getActiveStoppingCriteria(option.stop)
        #print("SR = $SR \n")
        Hη = getHessian(p, o.x, η)
        # Initialize the cost function F und the gradient of the cost function
        # ∇F at the point x
        grad = getGradient(p, o.x)
        fx = getCost(p, o.x)
        norm_grad = norm(p.M, o.x, grad)
        #print("norm_grad = $norm_grad\n")
        # If using randomized approach, compare result with the Cauchy point.
        # Convergence proofs assume that we achieve at least (a fraction of)
        # the reduction of the Cauchy point. After this if-block, either all
        # eta-related quantities have been changed consistently, or none of
        # them have changed.
        if o.useRand
                # Check the curvature,
                Hgrad = getHessian(p, o.x, grad)
                gradHgrad = dot(p.M, o.x, grad, Hgrad)
                if gradHgrad <= 0
                        tau_c = 1
                else
                        tau_c = min( norm_grad^3 /(o.Δ * gradHgrad), 1)
                end
                # and generate the Cauchy point.
                η_c = (-tau_c * o.Δ / norm_grad) * grad
                Hη_c = (-tau_c * o.Δ / norm_grad) * Hgrad
                # Now that we have computed the Cauchy point in addition to the
                # returned eta, we might as well keep the best of them.
                mdle  = fx + dot(p.M, o.x, grad, η) + .5 * dot(p.M, o.x, Hη, η)
                mdlec = fx + dot(p.M, o.x, grad, η_c) + .5 * dot(p.M, o.x, Hη_c, η_c)
                if mdlec < mdle
                        η = η_c
                        Hη = Hη_c
                end
        end

        # Compute the tentative next iterate (the proposal)
        x_prop  = retraction(p.M, o.x, η) #retraction ist auf 10^(-6) ungenau
        # print("norm = $(norm(p.M, o.x, η))\n")
        # Compute the function value of the proposal
        fx_prop = getCost(p, x_prop)
        # Will we accept the proposal or not?
        # Check the performance of the quadratic model against the actual cost.
        ρnum = fx - fx_prop
        ρden = -dot(p.M, o.x, η, grad) - 0.5*dot(p.M, o.x, η, Hη)
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
        # print("ρnum = $ρnum\n")
        # print("ρden = $ρden\n")
         print("ρ = $ρ\n")
        # Choose the new TR radius based on the model performance
        # If the actual decrease is smaller than 1/4 of the predicted decrease,
        # then reduce the TR radius.
        # print("o.Δ = $(o.Δ)\n")
        if ρ < 1/4 || model_decreased == false || isnan(ρ)
                o.Δ = o.Δ/4
        elseif ρ > 3/4 && (SR[4] != nothing || SR[5] != nothing)# we need to test the stopping criterions negative curvature and exceeded tr here.
                o.Δ = min(2*o.Δ, o.Δ_bar)
                print("bigger radius \n")
        else
                o.Δ = o.Δ
        end
        # Choose to accept or reject the proposed step based on the model
        # performance. Note the strict inequality.
        if model_decreased && ρ > o.ρ_prime
                o.x = x_prop
                print("accepted \n")
        end
        print("--------------------------\n")
end

function getSolverResult(p::P,o::O) where {P <: HessianProblem, O <: TrustRegionOptions}
        return o.x
end
