#
#   Truncated Conjugate-Gradient Method
#
export truncatedConjugateGradient

@doc doc"""
    truncatedConjugateGradient(M, F, ∇F, x, η, H, P, Δ)

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
* `∇F` – the gradient $\nabla F\colon\mathcal M\to T\mathcal M$ of F
* `x` – a point on the manifold $x\in\mathcal M$
* `η` – an update tangential vector $\eta\in\mathcal{T_{x}M}$
* `H` – a hessian matrix
* `P` – a preconditioner for the hessian matrix
* `Δ` – a trust-region radius

# Optional
* `θ` – 1+θ is the superlinear convergence target rate. The algorithm will
    terminate early if the residual was reduced by a power of 1+theta.
* `κ` – the linear convergence target rate: algorithm will terminate
    early if the residual was reduced by a factor of kappa
* `stoppingCriterion` – (`[`stopWhenAny`](@ref)`(`[`stopAfterIteration`](@ref)`([`manifoldDimension`](@ref)(M)),
  `[`stopResidualReducedByFactor`](@ref)`(κ))`, `[`stopResidualReducedByPower`](@ref)`(θ))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `useRandom` – set to true if the trust-region solve is to be initiated with a
    random tangent vector. If set to true, no preconditioner will be
    used. This option is set to true in some scenarios to escape saddle
    points, but is otherwise seldom activated.

# Output
* `η` – an approximate solution of the trust-region subproblem in
    $\mathcal{T_{x}M}.
"""
function truncatedConjugateGradient(M::mT,
        F::Function, ∇F::Function, x::MP, η::T,
        H::Union{Function,Missing},
        Δ::Float64;
        preconditioner::Function = x -> x,
        θ::Float64 = 1.0,
        κ::Float64 = 0.1,
        stoppingCriterion::StoppingCriterion = stopWhenAny(
            stopAfterIteration(manifoldDimension(M)),
            stopResidualReducedByPower(norm(M,x, ∇F(x) + ( useRandom ? zeroTVector(M,x) : H(η) ), 0,θ)),
            stopResidualReducedByFactor(norm(M,x, ∇F(x) + ( useRandom ? zeroTVector(M,x) : H(η) ), κ)),
        ),
        useRandom::Bool = false,
        kwargs... #collect rest
    ) where {mT <: Manifold, MP <: MPoint, T <: TVector}

    p = HessianProblem(M, F, ∇F, H, P)
    o = TruncatedConjugateGradientOptions(x,stoppingCriterion,η,zeroTVector(M,x),Δ,zeroTVector(M,x),useRandom)
    o = decorateOptions(o; kwargs...)
    resultO = solve(p,o)
    if hasRecord(resultO)
        return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
    end
    return getSolverResult(p,resultO)
end
function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    # If the falg useRand is on, eta is the zeroTVector.
    o.η = o.useRand ? zeroTVector(p.M,o.x) : o.η
    Hη = getHessian(p, o.x, o.η)
    o.residual = getGradient(p,o.x) + Hη
    # Precondition the residual.
    z = o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual
    # Compute z'*r.
    zr = dot(p.M, o.x, z, o.residual)
    # Initial search direction (we maintain -delta in memory, called mdelta, to
    # avoid a change of sign of the tangent vector.)
    o.δ = z
    # If the Hessian or a linear Hessian approximation is in use, it is
    # theoretically guaranteed that the model value decreases strictly
    # with each iteration of tCG. Hence, there is no need to monitor the model
    # value. But, when a nonlinear Hessian approximation is used (such as the
    # built-in finite-difference approximation for example), the model may
    # increase. It is then important to terminate the tCG iterations and return
    # the previous (the best-so-far) iterate. The variable below will hold the
    # model value.
    # o.model_value = o.useRand ? 0 : dot(p.M,o.x,o.η,getGradient(p,o.x)) + 0.5 * dot(p.M,o.x,o.η,Hη)
end
function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    ηOld = o.η
    δold = o.δ
    zold = o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual
    zrOld = dot(p.M, o.x, z, o.residual)
    HηOld = getHessian(p, o.x, o.η)
    # This call is the computationally expensive step.
    Hδ = getHessian(p, o.x, o.δ)
    # Compute curvature (often called kappa).
    δHδ = dot(p.M, o.x, o.δ, Hδ)
    # Note that if d_Hd == 0, we will exit at the next "if" anyway.
    α = zrOld/δHδ
    # <neweta,neweta>_P =
    # <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
    e_Pd = -dot(p.M, o.x, ηOld, getPreconditioner(p, o.x, δold)) # It must be clarified if it's negative or not
    d_Pd = dot(p.M, o.x, δold, getPreconditioner(p, o.x, δold))
    e_Pe = dot(p.M, o.x, ηOld, getPreconditioner(p, o.x, ηOld))
    ηαδ_Pηαδ = e_Pe + 2α*e_Pd + α^2*d_Pd
    # Check against negative curvature and trust-region radius violation.
    # If either condition triggers, we bail out.
    if δHδ <= 0 || ηαδ_Pηαδ >= o.Δ^2
        tau = (-e_Pd + sqrt(e_Pd^2 + d_Pd * (o.Δ^2 - e_Pe))) / d_Pd
        ηOld  = ηOld - tau * (o.δ)
    end
    # No negative curvature and eta_prop inside TR: accept it.
    o.η = ηOld - α * (o.δ)
    # Verify that the model cost decreased in going from eta to new_eta. If
    # it did not (which can only occur if the Hessian approximation is
    # nonlinear or because of numerical errors), then we return the
    # previous eta (which necessarily is the best reached so far, according
    # to the model cost). Otherwise, we accept the new eta and go on.
    # -> Stopping Criterion
    old_model_value = dot(p.M,o.x,ηOld,getGradient(p,o.x)) + 0.5 * dot(p.M,o.x,ηOld,HηOld)
    new_model_value = dot(p.M,o.x,o.η,getGradient(p,o.x)) + 0.5 * dot(p.M,o.x,o.η,getHessian(p, o.x, o.η))
    if new_model_value >= old_model_value
        o.η = ηOld
    end

    # Update the residual.
    o.residual = o.residual - α * Hδ
    # Precondition the residual.
    # It's actually the inverse of the preconditioner in o.residual
    z = o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual
    # Save the old z'*r.
    # Compute new z'*r.
    zr = dot(p.M, o.x, z, o.residual)
    # Compute new search direction.
    β = zr/zrOld
    o.δ = z + β * o.δ
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
end
function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    return o.η
end
