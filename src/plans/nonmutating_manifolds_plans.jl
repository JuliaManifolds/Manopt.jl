#
# For the manifolds that are nonmutating only, we have to introduce a few special cases
#
function get_gradient!(mp::AbstractManoptProblem, ::AbstractFloat, p)
    X = get_objective(mp).gradient!!(get_manifold(mp), p)
    return X
end
function get_hessian!(mp::HessianProblem, ::AbstractFloat, p, X)
    Y = get_objective(mp).hessian!!(get_manifold(mp), p, X)
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
#
# Specific solver steps
#
function initialize_solver!(
    mp::AbstractManoptProblem{M}, s::GradientDescentState
) where {M<:NONMUTATINGMANIFOLDS}
    s.X = get_gradient(mp, s.p)
    return s
end
function step_solver!(
    p::AbstractManoptProblem{M}, s::GradientDescentState, i
) where {M<:NONMUTATINGMANIFOLDS}
    step, s.X = s.direction(p, s, i)
    s.p = retract(get_manifold(p), s.p, -step * s.X, s.retraction_method)
    return s
end
#Hack for now?
copy(::NONMUTATINGMANIFOLDS, p) = p
