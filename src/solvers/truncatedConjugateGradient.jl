#
#
#
export
@doc doc"""

"""
function truncatedConjugateGradient(problem::HessianProblem, x::MP,
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
        Heta = getHessian(problem, x)
        r = lincomb(1, grad, 1, Heta) #No clue what it does jet
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

    if options.useRand == true
        e_Pd = 0
    else
        e_Pd = -dot(M, eta, mdelta)
    end
end
