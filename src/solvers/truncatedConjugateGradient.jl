#
#   Truncated Conjugate-Gradient Method
#
export truncatedConjugateGradient, model_fun

@doc doc"""
    truncatedConjugateGradient(P, x, grad, η, Δ, options)

solve the trust-region subproblem

$min_{\eta in T_{x_k}M} m_{x_k}(\eta) = f(x_k) + \langle \nabla f(x_k), \eta \rangle_{x_k} + \frac{1}{2} \langle Η_{x_k} \eta, \eta \rangle_{x_k}$
$s.t. \langle \eta, \eta \rangle_{x_k} \leqq {\Delta_k}^2$

with the Steihaug-Toint truncated conjugate-gradient method.
"""
function truncatedConjugateGradient(M,F,∇F,x::P,η::T,H::Union{Function,Mussing}=missing)

end
function initializeSolver!(p::P,o::O) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    o.∇ = getGradient(p,o.x)
    Heta = o.useRand ? zeroTVector(p.M,ox) : getHessian(p, o.x, o.η) # \eta in doe Options?
    r = o.∇ + Heta # radius in options? als Radius?
    e_Pe = useRand ? 0 : dot(p.M, o.x, o.η, o.η)

    z = o.useRand ? getPreconditioner(p, o.x, r) : r

    e_Pd = o.useRand ? 0 : -dot(p.M, o.x, o.η, -o.δ) # \delta in Options?
    model_value = o.useRand ? 0 : dot(p.M,o.x,o.η,o.∇) + 0.5 * dot(p.M,o.x,o.η,Heta)
end
function doSolverStep!(p::P,o::O,iter) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    model_function(M,x,ξ,η,ν) = dot(M,x,ξ,ν) + 0.5 * dot(M,x,ξ,η)

        Hmdelta = getHessian(problem, x, mdelta)
        d_Hd = dot(M, x, mdelta, Hmdelta)
        alpha = z_r/d_Hd
        e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd

        if d_Hd <= 0 || e_Pe_new >= Delta^2
            tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd
            eta  = TVector(getValue(eta)-tau*getValue(mdelta))
            Heta = TVector(getValue(Heta)-tau*getValue(Hmdelta))
        end

        e_Pe = e_Pe_new
        new_eta  = TVector(getValue(eta)-alpha*getValue(mdelta))
        new_Heta = TVector(getValue(Heta)-alpha*getValue(Hmdelta))

        new_model_value = model_fun(new_eta, new_Heta)

        eta = new_eta
        Heta = new_Heta
        model_value = new_model_value

        r = r-alpha*Hmdelta

        r_r = dot(M, x, r, r)
        norm_r = sqrt(r_r)

        if options.useRand == true
            z = getPrecon(problem, x, r)
        else
            z = r
        end


        zold_rold = z_r
        z_r = dot(M, x, z, r)

        beta = z_r/zold_rold
        mdelta = z + beta * mdelta

        e_Pd = beta*(e_Pd + alpha*d_Pd)
        d_Pd = z_r + beta*beta*d_Pd
end
