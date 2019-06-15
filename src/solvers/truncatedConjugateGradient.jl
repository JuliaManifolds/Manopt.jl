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
function truncatedConjugateGradient(problem::HessianProblem, x::MP, grad::MTVec,
    eta::MTVec, Delta::Float64, options::TruncatedConjugateGradientOptions)
    where {MP <: MPoint, MTVec <: MTVector}

    M = getManifold(problem)
    theta = options.theta;
    kappa = options.kappa;

    if options.useRand == true
        Heta = zeroTVector(M,x)
        r = grad
        e_Pe = 0
    else
        Heta = getHessian(problem, x, eta)
        r = TVector(getValue(grad)+getValue(Heta))
        e_Pe = dot(M, x, eta, eta)
    end

    r_r = dot(M, x, r, r)
    norm_r = sqrt(r_r)
    norm_r0 = norm_r

    if options.useRand == true
        z = getPreconditioner(problem, x, r)
    else
        z = r
    end

    z_r = dot(M, x, z, r)
    d_Pd = z_r
    mdelta = z;

    if options.useRand == true
        e_Pd = 0
    else
        e_Pd = -dot(M, x, eta, mdelta)
    end

    if options.useRand == true
        model_value = 0
    else
        model_value = model_fun(M, eta, Heta, grad)
    end

    stop_tCG = 5
    j = 0

    for j in 1:options.maxinner
        Hmdelta = getHessian(problem, x, mdelta)
        d_Hd = dot(M, x, mdelta, Hmdelta)
        alpha = z_r/d_Hd
        e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd

        if d_Hd <= 0 || e_Pe_new >= Delta^2
            tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd
            eta  = TVector(getValue(eta)-tau*getValue(mdelta))
            Heta = TVector(getValue(Heta)-tau*getValue(Hmdelta))

            if d_Hd <= 0
                stop_tCG = 1
            else
                stop_tCG = 2
            end
            break
        end

        e_Pe = e_Pe_new
        new_eta  = TVector(getValue(eta)-alpha*getValue(mdelta))
        new_Heta = TVector(getValue(Heta)-alpha*getValue(Hmdelta))

        new_model_value = model_fun(new_eta, new_Heta)

        if new_model_value >= model_value
            stop_tCG = 6
            break
        end

        eta = new_eta
        Heta = new_Heta
        model_value = new_model_value

        r = TVector(getValue(r)-alpha*getValue(Hmdelta))

        r_r = dot(M, x, r, r)
        norm_r = sqrt(r_r)

        if j >= options.mininner && norm_r <= norm_r0*min(norm_r0^theta, kappa)
            if kappa < norm_r0^theta
                stop_tCG = 3
            else
                stop_tCG = 4
            end
            break
        end

        if options.useRand == true
            z = getPrecon(problem, x, r)
        else
            z = r
        end


        zold_rold = z_r
        z_r = dot(M, x, z, r)

        beta = z_r/zold_rold
        mdelta = TVector(getValue(z)+beta*getValue(mdelta))

        mdelta = tangent(M, x, getValue(mdelta))

        e_Pd = beta*(e_Pd + alpha*d_Pd)
        d_Pd = z_r + beta*beta*d_Pd
    end
    inner_it = j
end

function model_fun(M::Manifold, x::MPoint, ξ::MTVector, η::MTVector, ν::MTVector)
    return dot(M,x,ξ,ν) + 0.5*dot(M,x,ξ,η)
end
