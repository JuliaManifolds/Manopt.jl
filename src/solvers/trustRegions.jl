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
* `∇F`- the gradient $\nabla F \colon \mathcal M \to T \mathcal M$ of $F$
* `x` – an initial value $x \in \mathcal M$
* `H` – the hessian $H( \mathcal M, x, \xi)$ of $F$

# Optional
* `preconditioner` – a preconditioner (a symmetric, positive definite operator
        that should approximate the inverse of the Hessian)
* `stoppingCriterion` – (`[`stopWhenAny`](@ref)`(`[`stopAfterIteration`](@ref)`(1000),
        `[`stopWhenGradientNormLess`](@ref)`(10^(-6))) a functor inheriting
        from [`StoppingCriterion`](@ref) indicating when to stop.
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
[`truncatedConjugateGradient`](@ref)
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

        o = TrustRegionsOptions(x,stoppingCriterion,Δ,Δ_bar,useRandom,ρ_prime,ρ_regularization)

        o = decorateOptions(o; kwargs...)
        resultO = solve(p,o)
        if hasRecord(resultO)
                return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
        end
        return getSolverResult(p,resultO)
end

function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TrustRegionsOptions}
end

function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TrustRegionsOptions}
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
                end
        end
        # Solve TR subproblem approximately
        (η, option) = truncatedConjugateGradient(p.M,p.costFunction,p.gradient,
        o.x,eta,p.hessian,o.Δ;preconditioner=p.precon,useRandom=o.useRand,
        debug = [:Iteration," ",:Stop])
        SR = getActiveStoppingCriteria(option.stop)
        Hη = getHessian(p, o.x, η)
        # Initialize the cost function F und the gradient of the cost function
        # ∇F at the point x
        grad = getGradient(p, o.x)
        fx = getCost(p, o.x)
        norm_grad = norm(p.M, o.x, grad)
        # If using randomized approach, compare result with the Cauchy point.
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
        x_prop  = retraction(p.M, o.x, η)
        # Compute the function value of the proposal
        fx_prop = getCost(p, x_prop)
        # Check the performance of the quadratic model against the actual cost.
        ρnum = fx - fx_prop
        ρden = -dot(p.M, o.x, η, grad) - 0.5*dot(p.M, o.x, η, Hη)

        # Since, at convergence, both ρnum and ρden become extremely small,
        # computing ρ is numerically challenging. The break with ρnum and ρden
        # can thus lead to a large error in rho, making the
        # acceptance / rejection erratic. Meanwhile, close to convergence,
        # steps are usually trustworthy and we should transition to a Newton-
        # like method, with rho=1 consistently. The heuristic thus shifts both
        # rhonum and rhoden by a small amount such that far from convergence,
        # the shift is irrelevant and close to convergence, the ratio rho goes
        # to 1, effectively promoting acceptance of the step.

        ρ_reg = max(1, abs(fx)) * eps(Float64) * o.ρ_regularization
        ρnum = ρnum + ρ_reg
        ρden = ρden + ρ_reg

        model_decreased = (ρden >= 0)
        ρ = ρnum / ρden
        # Choose the new TR radius based on the model performance.
        # If the actual decrease is smaller than 1/4 of the predicted decrease,
        # then reduce the TR radius.
        # print("o.Δ = $(o.Δ)\n")
        if ρ < 1/4 || model_decreased == false || isnan(ρ)
                o.Δ = o.Δ/4
        elseif ρ > 3/4 && any([typeof(s) in [stopExceededTrustRegion,stopNegativeCurvature] for s in SR] )
                o.Δ = min(2*o.Δ, o.Δ_bar)
        else
                o.Δ = o.Δ
        end
        # Choose to accept or reject the proposed step based on the model
        # performance. Note the strict inequality.
        if model_decreased && ρ > o.ρ_prime
                o.x = x_prop
        end
        print("--------------------------\n")
end

function getSolverResult(p::P,o::O) where {P <: HessianProblem, O <: TrustRegionsOptions}
        return o.x
end
