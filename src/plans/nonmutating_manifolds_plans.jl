#
# For the manifolds that are nonmutating only, we have to introduce a few special cases
#
import Base: copy

function get_gradient!(mp::AbstractManoptProblem, ::AbstractFloat, p)
    X = get_objective(mp).gradient!!(get_manifold(mp), p)
    return X
end
function get_hessian!(mp::AbstractManoptProblem, ::AbstractFloat, p, X)
    Y = get_objective(mp).hessian!!(get_manifold(mp), p, X)
    return Y
end
function linesearch_backtrack(
    M::NONMUTATINGMANIFOLDS,
    F::TF,
    p,
    gradF::T,
    s,
    decrease,
    contract,
    retr::AbstractRetractionMethod=ExponentialRetraction(),
    η::T=-gradF,
    f0=F(p);
    stop_when_stepsize_less=0.0,
    stop_when_stepsize_larger=max_stepsize(M, p) / norm(M, p, η),
    max_increase_steps=100,
    max_decrease_steps=1000,
) where {TF,T}
    msg = ""
    p_new = retract(M, p, s * η, retr)
    fNew = F(p_new)
    i = 0
    while fNew < f0 + decrease * s * real(inner(M, p, η, gradF)) # increase
        i = i + 1
        s = s / contract
        p_new = retract(M, p, η, s, retr)
        fNew = F(p_new)
        if i == max_increase_steps
            msg = "Max increase steps ($(max_increase_steps) reached"
            break
        end
        if s > stop_when_stepsize_larger
            (length(msg) > 0) && (msg = "$msg\n")
            s = s * contract
            msg = "$msg Max step size ($(stop_when_stepsize_larger) reached, reducing to $s"
            break
        end
    end
    i = 0
    while fNew > f0 + decrease * s * real(inner(M, p, η, gradF)) # decrease
        i = i + 1
        s = contract * s
        p_new = retract(M, p, η, s, retr)
        fNew = F(p_new)
        if i == max_decrease_steps
            (length(msg) > 0) && (msg = "$msg\n")
            msg = "Max decrese steps ($(max_decrease_steps) reached"
            break
        end
        if s < stop_when_stepsize_less
            (length(msg) > 0) && (msg = "$msg\n")
            s = s / contract
            msg = "$msg Min step size ($(stop_when_stepsize_less) exceeded, increasing back to $s"
            break
        end
    end
    return (s, msg)
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
