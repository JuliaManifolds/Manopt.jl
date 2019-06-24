#
#   Truncated Conjugate-Gradient Method
#
export truncatedConjugateGradient, model_fun

@doc doc"""
    truncatedConjugateGradient(M, F, ∂F, x, η; H, P, Δ, stoppingCriterion, uR)

solve the trust-region subproblem

$min_{\eta in T_{x}M} m_{x}(\eta) = F(x) + \langle \partialF(x), \eta \rangle_{x} + \frac{1}{2} \langle Η_{x} \eta, \eta \rangle_{x}$
$s.t. \langle \eta, \eta \rangle_{x} \leqq {\Delta}^2$

with the Steihaug-Toint truncated conjugate-gradient method.
"""
function truncatedConjugateGradient(M::mT,
        F::Function, ∂F::Function, x::MP, eta::T;
        H::Union{Function,Missing},
        P::Function,
        Δ::Float64,
        stoppingCriterion::StoppingCriterion = stopAfterIteration(5000),
        uR::Bool
    ) where {mT <: Manifold, MP <: MPoint, T <: TVector}
    p = HessianProblem(M,F,∂F,H,P)
    o = TruncatedConjugateGradientOptions(x,stoppingCriterion,eta,zeroTVector(M,x),zeroTVector(M,x),Δ,0,0,0,zeroTVector(M,x),zeroTVector(M,x),0,uR)

    resultO = solve(p,o)
    if hasRecord(resultO)
        return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
    end
    return getSolverResult(p,resultO)
end
function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    # If the falg useRand is on, eta is the zeroTVector.
    o.Hη = o.useRand ? zeroTVector(p.M,o.x) : getHessian(p, o.x, o.η)
    o.residual = getGradient(p,o.x) + o.Hη
    o.e_Pe = useRand ? 0 : dot(p.M, o.x, o.η, o.η)
    # Precondition the residual.
    o.z = o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual
    # Compute z'*r.
    o.z_r = dot(p.M, o.x, o.z, o.residual)
    o.d_Pd = o.z_r
    # Initial search direction (we maintain -delta in memory, called mdelta, to
    # avoid a change of sign of the tangent vector.)
    o.mδ = o.z
    o.e_Pd = o.useRand ? 0 : -dot(p.M, o.x, o.η, o.mδ)
    # If the Hessian or a linear Hessian approximation is in use, it is
    # theoretically guaranteed that the model value decreases strictly
    # with each iteration of tCG. Hence, there is no need to monitor the model
    # value. But, when a nonlinear Hessian approximation is used (such as the
    # built-in finite-difference approximation for example), the model may
    # increase. It is then important to terminate the tCG iterations and return
    # the previous (the best-so-far) iterate. The variable below will hold the
    # model value.
    o.model_value = o.useRand ? 0 : dot(p.M,o.x,o.η,getGradient(p,o.x)) + 0.5 * dot(p.M,o.x,o.η,o.Hη)
end
function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    # This call is the computationally expensive step.
    Hmdelta = getHessian(p, o.x, o.mδ)
    # Compute curvature (often called kappa).
    d_Hd = dot(p.M, o.x, o.mδ, Hmdelta)
    # Note that if d_Hd == 0, we will exit at the next "if" anyway.
    alpha = o.z_r/d_Hd
    # <neweta,neweta>_P =
    # <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
    e_Pe_new = o.e_Pe + 2.0*alpha*o.e_Pd + alpha*alpha*o.d_Pd
    # Check against negative curvature and trust-region radius violation.
    # If either condition triggers, we bail out.
    if d_Hd <= 0 || e_Pe_new >= o.Δ^2
        tau = (-o.e_Pd + sqrt(o.e_Pd*o.e_Pd + o.d_Pd*(o.Δ^2-o.e_Pe))) / o.d_Pd
        o.η  = o.η-tau*(o.δ)
        # % If only a nonlinear Hessian approximation is available, this is
        # only approximately correct, but saves an additional Hessian call.
        o.Hη = o.Hη-tau*Hmdelta
    end
    # No negative curvature and eta_prop inside TR: accept it.
    o.e_Pe = e_Pe_new
    new_eta  = o.η-alpha*(o.mδ)
    # If only a nonlinear Hessian approximation is available, this is
    # only approximately correct, but saves an additional Hessian call.
    new_Heta = o.Hη-alpha*Hmdelta
    # Verify that the model cost decreased in going from eta to new_eta. If
    # it did not (which can only occur if the Hessian approximation is
    # nonlinear or because of numerical errors), then we return the
    # previous eta (which necessarily is the best reached so far, according
    # to the model cost). Otherwise, we accept the new eta and go on.
    new_model_value = dot(p.M,o.x,new_eta,getGradient(p,o.x)) + 0.5 * dot(p.M,o.x,new_eta,new_Heta)
    if new_model_value >= o.model_value
        break
    end
    o.η = new_eta
    o.Hη = new_Heta
    o.model_value = new_model_value
    # Update the residual.
    o.residual = o.residual-alpha*Hmdelta
    # Precondition the residual.
    o.z = o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual
    # Save the old z'*r.
    zold_rold = o.z_r
    # Compute new z'*r.
    o.z_r = dot(p.M, o.x, o.z, o.residual)
    # Compute new search direction.
    beta = o.z_r/zold_rold
    o.mδ = o.z + beta*o.mδ
    # Since mdelta is passed to getHessian, which is the part of the code
    # we have least control over from here, we want to make sure mdelta is
    # a tangent vector up to numerical errors that should remain small.
    # For this reason, we re-project mdelta to the tangent space.
    # In limited tests, it was observed that it is a good idea to project
    # at every iteration rather than only every k iterations, the reason
    # being that loss of tangency can lead to more inner iterations being
    # run, which leads to an overall higher computational cost.
    # Not sure if this is necessary. We need to discuss this.
    o.mδ = tangent(p.M, o.x, getValue(o.mδ))
    # Update new P-norms and P-dots
    o.e_Pd = beta*(o.e_Pd + alpha*o.d_Pd)
    o.d_Pd = o.z_r + beta*beta*o.d_Pd

    return o.η
    return o.Hη
end
function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    return o.ηOptimal
    return o.HηOptimal
end
