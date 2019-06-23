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
    o = TruncatedConjugateGradientOptions(x,eta,zeroTVector(M,x),zeroTVector(M,x),Δ,0,0,0,zeroTVector(M,x),zeroTVector(M,x),uR)
end
function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    o.Hη = o.useRand ? zeroTVector(p.M,o.x) : getHessian(p, o.x, o.η)
    o.residual = getGradient(p,o.x) + o.Hη 
    o.e_Pe = useRand ? 0 : dot(p.M, o.x, o.η, o.η)

    o.z = o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual

    o.z_r = dot(p.M, o.x, o.z, o.residual)
    o.d_Pd = o.z_r

    o.mδ = o.z

    o.e_Pd = o.useRand ? 0 : -dot(p.M, o.x, o.η, o.mδ)

    model_value = o.useRand ? 0 : dot(p.M,o.x,o.η,getGradient(p,o.x)) + 0.5 * dot(p.M,o.x,o.η,o.Hη)
end
function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    Hmdelta = getHessian(p, o.x, o.mδ)
    d_Hd = dot(p.M, o.x, o.mδ, Hmdelta)
    alpha = o.z_r/d_Hd
    e_Pe_new = o.e_Pe + 2.0*alpha*o.e_Pd + alpha*alpha*o.d_Pd

    if d_Hd <= 0 || e_Pe_new >= o.Δ^2
        tau = (-o.e_Pd + sqrt(o.e_Pd*o.e_Pd + o.d_Pd*(o.Δ^2-o.e_Pe))) / o.d_Pd
        o.η  = o.η-tau*(o.δ)
        o.Hη = o.Hη-tau*Hmdelta
    end

    o.e_Pe = e_Pe_new
    new_eta  = o.η-alpha*(o.mδ)
    new_Heta = o.Hη-alpha*Hmdelta

    new_model_value = dot(p.M,o.x,new_eta,getGradient(p,o.x)) + 0.5 * dot(p.M,o.x,new_eta,new_Heta)

    o.η = new_eta
    o.Hη = new_Heta
    model_value = new_model_value

    o.residual = o.residual-alpha*Hmdelta

    o.z = o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual

    zold_rold = o.z_r
    o.z_r = dot(p.M, o.x, o.z, o.residual)

    beta = o.z_r/zold_rold
    o.mδ = o.z + beta*o.mδ

    # not sure if this is necessary. We need to discuss this.
    o.mδ = tangent(p.M, o.x, getValue(o.mδ))

    o.e_Pd = beta*(o.e_Pd + alpha*o.d_Pd)
    o.d_Pd = o.z_r + beta*beta*o.d_Pd

    return o.η
    return o.Hη
end

function getSolverResult(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    return o.ηOptimal
    return o.HηOptimal
end
