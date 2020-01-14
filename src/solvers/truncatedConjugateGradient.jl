#
#   Truncated Conjugate-Gradient Method
#
export truncatedConjugateGradient

@doc doc"""
    truncatedConjugateGradient(M, F, ∇F, x, η, H, Δ)

solve the trust-region subproblem

```math
\operatorname*{arg\,min}_{\eta \in T_{x}M} m_{x}(\eta) = F(x) + \langle \nabla F(x), \eta \rangle_{x} + \frac{1}{2} \langle \operatorname{Hess}[F](\eta)_ {x}, \eta \rangle_{x}
```
```math
\text{s.t.} \; \langle \eta, \eta \rangle_{x} \leqq {\Delta}^2
```

with the Steihaug-Toint truncated conjugate-gradient method.
For a description of the algorithm and theorems offering convergence guarantees,
see the reference:

* P.-A. Absil, C.G. Baker, K.A. Gallivan,
    Trust-region methods on Riemannian manifolds, FoCM, 2007.
    doi: [10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9)
* A. R. Conn, N. I. M. Gould, P. L. Toint, Trust-region methods, SIAM,
    MPS, 2000. doi: [10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∇F` – the gradient $\nabla F\colon\mathcal M\to T\mathcal M$ of F
* `x` – a point on the manifold $x\in\mathcal M$
* `η` – an update tangential vector $\eta\in\mathcal{T_{x}M}$
* `H` – the hessian $H( \mathcal M, x, \xi)$ of F
* `Δ` – a trust-region radius

# Optional
* `preconditioner` – a preconditioner for the hessian H
* `θ` – 1+θ is the superlinear convergence target rate. The algorithm will
    terminate early if the residual was reduced by a power of 1+theta.
* `κ` – the linear convergence target rate: algorithm will terminate
    early if the residual was reduced by a factor of kappa.
* `useRandom` – set to true if the trust-region solve is to be initiated with a
    random tangent vector. If set to true, no preconditioner will be
    used. This option is set to true in some scenarios to escape saddle
    points, but is otherwise seldom activated.
* `stoppingCriterion` – ([`stopWhenAny`](@ref), [`stopAfterIteration`](@ref),
    [`stopIfResidualIsReducedByFactor`](@ref), [`stopIfResidualIsReducedByPower`](@ref),
    [`stopWhenCurvatureIsNegative`](@ref), [`stopWhenTrustRegionIsExceeded`](@ref) )
    a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop,
    where for the default, the maximal number of iterations is [`manifoldDimension`](@ref),
    the power factor is `θ`, the reduction factor is `κ`.
    .
* `returnOptions` – (`false`) – if actiavated, the extended result, i.e. the
    complete [`Options`](@ref) re returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `xOpt` if returned

and the ones that are passed to [`decorateOptions`](@ref) for decorators.

# Output
* `η` – an approximate solution of the trust-region subproblem in
    $\mathcal{T_{x}M}$.
OR
* `options` - the options returned by the solver (see `returnOptions`)

# see also
[`trustRegions.jl`](@ref)
"""
function truncatedConjugateGradient(M::mT,
        F::Function, ∇F::Function, x::MP, η::T,
        H::Union{Function,Missing},
        Δ::Float64;
        preconditioner::Function = (M,x,ξ) -> ξ,
        θ::Float64 = 1.0,
        κ::Float64 = 0.1,
        useRandom::Bool = false,
        stoppingCriterion::StoppingCriterion = stopWhenAny(
            stopAfterIteration(manifoldDimension(M)),
            stopIfResidualIsReducedByPower( sqrt( dot(M,x, ∇F(M,x) + ( useRandom ? H(M,x,η) : zeroTVector(M,x) ), ∇F(M,x) + ( useRandom ? H(M,x,η) : zeroTVector(M,x) )) ), θ),
            stopIfResidualIsReducedByFactor( sqrt( dot(M,x, ∇F(M,x) + ( useRandom ? H(M,x,η) : zeroTVector(M,x) ), ∇F(M,x) + ( useRandom ? H(M,x,η) : zeroTVector(M,x) )) ), κ),
            stopWhenTrustRegionIsExceeded(),
            stopWhenCurvatureIsNegative()
        ),
        returnOptions = false,
        kwargs... #collect rest
    ) where {mT <: Manifold, MP <: MPoint, T <: TVector}
    p = HessianProblem(M, F, ∇F, H, preconditioner)
    o = TruncatedConjugateGradientOptions(x,stoppingCriterion,η,zeroTVector(M,x),Δ,zeroTVector(M,x),useRandom)
    o = decorateOptions(o; kwargs...)
    resultO = solve(p,o)
    if returnOptions
        resultO
    else
        return getSolverResult(resultO)
    end
end
function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    o.η = o.useRand ? o.η : zeroTVector(p.M,o.x)
    Hη = o.useRand ? getHessian(p, o.x, o.η) : zeroTVector(p.M,o.x)
    o.residual = getGradient(p,o.x) + Hη
    # Initial search direction (we maintain -delta in memory, called mdelta, to
    # avoid a change of sign of the tangent vector.)
    o.δ = o.useRand ? o.residual : getPreconditioner(p, o.x, o.residual)
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
    δOld = o.δ
    z = o.useRand ? o.residual : getPreconditioner(p, o.x, o.residual)
    # this is not correct, it needs to be the inverse of the preconditioner!
    zrOld = dot(p.M, o.x, z, o.residual)
    HηOld = getHessian(p, o.x, ηOld)
    # This call is the computationally expensive step.
    Hδ = getHessian(p, o.x, δOld)
    # Compute curvature (often called kappa).
    δHδ = dot(p.M, o.x, δOld, Hδ)
    # Note that if d_Hd == 0, we will exit at the next "if" anyway.
    α = zrOld/δHδ
    # <neweta,neweta>_P =
    # <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
    e_Pd = -dot(p.M, o.x, ηOld, o.useRand ? δOld : getPreconditioner(p, o.x, δOld)) # It must be clarified if it's negative or not
    d_Pd = dot(p.M, o.x, δOld, o.useRand ? δOld : getPreconditioner(p, o.x, δOld))
    e_Pe = dot(p.M, o.x, ηOld, o.useRand ? ηOld : getPreconditioner(p, o.x, ηOld))
    e_Pe_new = e_Pe + 2α*e_Pd + α^2*d_Pd # vielleicht müssen doch die weiteren Optionen gespeichert werden
    # Check against negative curvature and trust-region radius violation.
    # If either condition triggers, we bail out.
    if δHδ <= 0 || e_Pe_new >= o.Δ^2
        tau = (-e_Pd + sqrt(e_Pd^2 + d_Pd * (o.Δ^2 - e_Pe))) / d_Pd
        o.η = ηOld - tau * (δOld)
    else
        # No negative curvature and eta_prop inside TR: accept it.
        o.η = ηOld - α * (δOld)
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
    end
    # Update the residual.
    o.residual = o.residual - α * Hδ
    # Precondition the residual.
    # It's actually the inverse of the preconditioner in o.residual
    z = o.useRand ? o.residual : getPreconditioner(p, o.x, o.residual)
    # this is not correct, it needs to be the inverse of the preconditioner!
    # Save the old z'*r.
    # Compute new z'*r.
    zr = dot(p.M, o.x, z, o.residual)
    # Compute new search direction.
    β = zr/zrOld
    o.δ = tangent(p.M, o.x, z + β * o.δ)
end
getSolverResult(o::O) where {O <: TruncatedConjugateGradientOptions} = o.η
