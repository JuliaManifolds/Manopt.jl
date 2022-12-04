#
# For the manifolds that are nonmutating only, we have to introduce a few special cases
#
function get_gradient!(p::AbstractManoptProblem, ::AbstractFloat, x)
    X = get_objective(p).gradient!!(p.M, x)
    return X
end
function get_hessian!(p::HessianProblem, ::AbstractFloat, x, X)
    Y = get_objective(p).hessian!!(p.M, x, X)
    return Y
end
function linesearch_backtrack(
    M::NONMUTATINGMANIFOLDS,
    F::TF,
    x,
    gradF::T,
    s,
    decrease,
    contract,
    retr::AbstractRetractionMethod=ExponentialRetraction(),
    η::T=-gradF,
    f0=F(x);
    stop_step=0.0,
) where {TF,T}
    x_new = retract(M, x, s * η, retr)
    fNew = F(x_new)
    while fNew < f0 + decrease * s * inner(M, x, η, gradF) # increase
        x_new = retract(M, x, s * η, retr)
        fNew = F(x_new)
        s = s / contract
    end
    s = s * contract # correct last
    while fNew > f0 + decrease * s * inner(M, x, η, gradF) # decrease
        s = contract * s
        x_new = retract(M, x, s * η, retr)
        fNew = F(x_new)
        (s < stop_step) && break
    end
    return s
end
