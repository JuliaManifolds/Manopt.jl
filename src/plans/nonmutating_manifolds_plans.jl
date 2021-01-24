#
# For the manifolds that are nonmutating only, we have to introduce a few special cases
#
function get_gradient!(p::GradientProblem{AllocatingEvaluation}, X::AbstractFloat, x)
    X = p.gradient!!(x)
    return X
end
function linesearch_backtrack(
    M::Union{Circle, PositiveNumbers},
    F::TF,
    x,
    ∇F::T,
    s,
    decrease,
    contract,
    retr::AbstractRetractionMethod=ExponentialRetraction(),
    η::T=-∇F,
    f0=F(x),
) where {TF,T}
    x_new = retract(M, x, s * η, retr)
    fNew = F(x_new)
    while fNew < f0 + decrease * s * inner(M, x, η, ∇F) # increase
        x_new = retract(M, x, s * η, retr)
        fNew = F(x_new)
        s = s / contract
    end
    s = s * contract # correct last
    while fNew > f0 + decrease * s * inner(M, x, η, ∇F) # decrease
        s = contract * s
        x_new = retract(M, x, s * η, retr)
        fNew = F(x_new)
    end
    return s
end