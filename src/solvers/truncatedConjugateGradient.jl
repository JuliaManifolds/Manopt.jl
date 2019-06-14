#
#
#
export
@doc doc"""

"""
function truncatedConjugateGradient(problem::HessianProblem, x::MP, grad::MTVec,
    eta::MTVec, Delta::Float64, options::TruncatedConjugateGradientOptions)
    where {MP <: MPoint, MTVec <: MTVector}

    M = getManifold(problem)
    theta = options.theta;
    kappa = options.kappa;

    if options.useRand == true
        Heta = zeroTVector(M,x)
        r = getGradient(problem, x)
        e_Pe = 0
    else
        Heta = getHessian(problem, x, eta)
        r = lincomb(1, grad, 1, Heta) #No clue what it does yet
        e_Pe = dot(M, eta, eta)
    end

    r_r = dot(M, r, r)
    norm_r = sqrt(r_r)
    norm_r0 = norm_r

    if options.useRand == true
        z = getPreconditioner(problem, x, r)
    else
        z = r
    end

    z_r = dot(M, z, r)
    d_Pd = z_r
    mdelta = z;

    if options.useRand == true
        e_Pd = 0
    else
        e_Pd = -dot(M, eta, mdelta)
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
        d_Hd = dot(M, mdelta, Hmdelta)
        alpha = z_r/d_Hd
        e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd

        if d_Hd <= 0 || e_Pe_new >= Delta^2
            tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd
            eta  = lincomb(1,  eta, -tau,  mdelta)
            Heta = lincomb(1, Heta, -tau, Hmdelta)

            if d_Hd <= 0
                stop_tCG = 1
            else
                stop_tCG = 2
            end
            break
        end

        e_Pe = e_Pe_new
        new_eta  = lincomb(1,  eta, -alpha,  mdelta)
        new_Heta = lincomb(1, Heta, -alpha, Hmdelta)

        new_model_value = model_fun(new_eta, new_Heta)

        if new_model_value >= model_value
            stop_tCG = 6
            break
        end

        eta = new_eta
        Heta = new_Heta
        model_value = new_model_value

        r = lincomb(1, r, -alpha, Hmdelta)

        r_r = dot(M, r, r)
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
            z = getPrecon(problem, x, r, storedb, key);
        else
            z = r;
        end
end

function model_fun(M::Manifold, ξ::MTVector, η::MTVector, ν::MTVector)
    return dot(M,ξ,ν) + 0.5*dot(M,ξ,η)
end
