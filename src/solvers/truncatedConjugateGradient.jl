#
#   Truncated Conjugate-Gradient Method
#
export truncatedConjugateGradient, model_fun

@doc doc"""
    truncatedConjugateGradient(M, F, ∂F, x, η, H, P, Δ, stoppingCriterion, uR)

solve the trust-region subproblem

```math
min_{\eta in T_{x}M} m_{x}(\eta) = F(x) + \langle \partialF(x), \eta \rangle_{x} + \frac{1}{2} \langle Η_{x} \eta, \eta \rangle_{x}
```
```math
\text{s.t.} \; \langle \eta, \eta \rangle_{x} \leqq {\Delta}^2
```

with the Steihaug-Toint truncated conjugate-gradient method.

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∂F`: the (sub)gradient $\partial F\colon\mathcal M\to T\mathcal M$ of F
  restricted to always only returning one value/element from the subgradient
* `x` – an initial value $x\in\mathcal M$
* `η` – an update tangential vector $\eta\in\mathcal{T_{x}M}$
* `H` – a hessian matrix
"""
function truncatedConjugateGradient(M::mT,
        F::Function, ∂F::Function, x::MP, eta::T,
        H::Union{Function,Missing},
        P::Function,
        Δ::Float64;
        θ::Float64 = 1.0,
        κ::Float64 = 0.1,
        stoppingCriterion::StoppingCriterion = stopWhenAny(
            stopAfterIteration(manifoldDimension(M)),
            stopResidualReducedByPower(1.0,θ),
            stopResidualReducedByFactor(1.0,κ),
        ),
        useRandom::Bool = false
    ) where {mT <: Manifold, MP <: MPoint, T <: TVector}
    p = HessianProblem(M,F,∂F,H,P)
    o = TruncatedConjugateGradientOptions(x,stoppingCriterion,eta,zeroTVector(M,x),zeroTVector(M,x),Δ,0,0,0,zeroTVector(M,x),zeroTVector(M,x),0,useRandom)

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
    o.zr = dot(p.M, o.x, o.z, o.residual)
    o.d_Pd = o.z_r
    # Initial search direction (we maintain -delta in memory, called mdelta, to
    # avoid a change of sign of the tangent vector.)
    o.δ = o.z
    o.e_Pd = o.useRand ? 0 : -dot(p.M, o.x, o.η, o.δ)
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
    ηOld = o.η
    zrOld = o.zr
    HηOld = o.Hη
    e_PeOld = o.e_Pe
    # This call is the computationally expensive step.
    Hδ = getHessian(p, o.x, o.δ)
    # Compute curvature (often called kappa).
    δHδ = dot(p.M, o.x, o.δ, Hδ)
    # Note that if d_Hd == 0, we will exit at the next "if" anyway.
    α = zrOld/δHδ
    # <neweta,neweta>_P =
    # <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
    o.e_Pe = e_PeOld + 2.0 * α * o.e_Pd + α^2 * o.d_Pd
    # Check against negative curvature and trust-region radius violation.
    # If either condition triggers, we bail out.
    if δHδ <= 0 || o.e_Pe >= o.Δ^2
        tau = (-o.e_Pd + sqrt(o.e_Pd^2 + o.d_Pd * (o.Δ^2 - e_PeOld))) / o.d_Pd
        ηOld  = ηOld - tau * (o.δ)
        # If only a nonlinear Hessian approximation is available, this is
        # only approximately correct, but saves an additional Hessian call.
        HηOld = HηOld - tau * Hδ
    end
    # No negative curvature and eta_prop inside TR: accept it.
    o.η = ηOld - α * (o.δ)
    # If only a nonlinear Hessian approximation is available, this is
    # only approximately correct, but saves an additional Hessian call.
    o.Hη = HηOld - α * Hδ
    # Verify that the model cost decreased in going from eta to new_eta. If
    # it did not (which can only occur if the Hessian approximation is
    # nonlinear or because of numerical errors), then we return the
    # previous eta (which necessarily is the best reached so far, according
    # to the model cost). Otherwise, we accept the new eta and go on.
    # -> Stopping Criterion
    # new_model_value = dot(p.M,o.x,new_eta,getGradient(p,o.x)) + 0.5 * dot(p.M,o.x,new_eta,new_Heta)
    # if new_model_value >= o.model_value
    #     break
    # end
    # o.model_value = new_model_value
    # Update the residual.
    o.residual = o.residual - α * Hδ
    # Precondition the residual.
    o.z = o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual
    # Save the old z'*r.
    # Compute new z'*r.
    o.zr = dot(p.M, o.x, o.z, o.residual)
    # Compute new search direction.
    β = o.zr/zrOld
    o.δ = o.z + β * o.δ
    # Since mdelta is passed to getHessian, which is the part of the code
    # we have least control over from here, we want to make sure mdelta is
    # a tangent vector up to numerical errors that should remain small.
    # For this reason, we re-project mdelta to the tangent space.
    # In limited tests, it was observed that it is a good idea to project
    # at every iteration rather than only every k iterations, the reason
    # being that loss of tangency can lead to more inner iterations being
    # run, which leads to an overall higher computational cost.
    # Not sure if this is necessary. We need to discuss this.
    o.δ = tangent(p.M, o.x, getValue(o.δ))
    # Update new P-norms and P-dots
    o.e_Pd = β * (o.e_Pd + α * o.d_Pd)
    o.d_Pd = o.zr + β^2 *o.d_Pd
end
function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    return o.η
end
