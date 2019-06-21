#
#   Truncated Conjugate-Gradient Method
#
export truncatedConjugateGradient, model_fun

@doc doc"""
    truncatedConjugateGradient(M, F, ∂F, x, η; H, P, Δ, uR)

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
        uR::Bool
    ) where {mT <: Manifold, MP <: MPoint, T <: TVector}
    p = HessianProblem(M,F,∂F,H,P)
    ∇ = getGradient(p,x)
    o = TruncatedConjugateGradientOptions(x,eta,∇,zeroTVector(M,x),Δ,uR)
end
function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    Heta = o.useRand ? zeroTVector(p.M,o.x) : getHessian(p, o.x, o.η) # \eta in doe Options?
    r = o.∇ + Heta # radius in options? als Radius?
    e_Pe = useRand ? 0 : dot(p.M, o.x, o.η, o.η)

    z = o.useRand ? getPreconditioner(p, o.x, r) : r

    z_r = dot(p.M, o.x, z, r)
    d_Pd = z_r

    o.mδ = z

    e_Pd = o.useRand ? 0 : -dot(p.M, o.x, o.η, o.mδ)

    model_value = o.useRand ? 0 : dot(p.M,o.x,o.η,o.∇) + 0.5 * dot(p.M,o.x,o.η,Heta)
end
function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    Hmdelta = getHessian(p, o.x, o.mδ)
    d_Hd = dot(p.M, o.x, o.mδ, Hmdelta)
    alpha = z_r/d_Hd
    e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd

    if d_Hd <= 0 || e_Pe_new >= Δ^2
        tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Δ^2-e_Pe))) / d_Pd
        eta  = eta-tau*(o.δ)
        Heta = Heta-tau*Hmdelta
    end

    e_Pe = e_Pe_new
    new_eta  = eta-alpha*(o.mδ)
    new_Heta = Heta-alpha*Hmdelta

    new_model_value = dot(p.M,o.x,new_eta,o.∇) + 0.5 * dot(p.M,o.x,new_eta,new_Heta)

    eta = new_eta
    Heta = new_Heta
    model_value = new_model_value

    r = r-alpha*Hmdelta

    z = o.useRand ? getPreconditioner(p, o.x, r) : r

    zold_rold = z_r
    z_r = dot(p.M, o.x, z, r)

    beta = z_r/zold_rold
    o.mδ = z + beta*o.mδ

    # not sure if this is necessary. We need to discuss this. 
    o.mδ = tangent(p.M, o.x, getValue(o.mδ))

    e_Pd = beta*(e_Pd + alpha*d_Pd)
    d_Pd = z_r + beta*beta*d_Pd

    return eta
    return Heta
end
