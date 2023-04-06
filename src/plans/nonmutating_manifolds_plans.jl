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
    grad_f_at_p::T,
    s,
    decrease,
    contract,
    retr::AbstractRetractionMethod=ExponentialRetraction(),
    η::T=-grad_f_at_p,
    f0=F(M, p);
    stop_when_stepsize_less=0.0,
    stop_when_stepsize_exceeds=max_stepsize(M, p) / norm(M, p, η),
    stop_increasing_at_step=100,
    stop_decreasing_at_step=1000,
) where {TF,T}
    msg = ""
    p_new = retract(M, p, s * η, retr)
    fNew = F(M, p_new)
    search_dir_inner = real(inner(M, p, η, grad_f_at_p))
    if search_dir_inner >= 0
        msg = "The search direction η might not be a descent directon, since ⟨η, grad_f(p)⟩ ≥ 0."
    end
    i = 0
    while fNew < f0 + decrease * s * search_dir_inner # increase
        i = i + 1
        s = s / contract
        p_new = retract(M, p, η, s, retr)
        fNew = F(M, p_new)
        if i == stop_increasing_at_step
            (length(msg) > 0) && (msg = "$msg\n")
            msg = "$(msg)Max increase steps ($(stop_increasing_at_step)) reached"
            break
        end
        if s > stop_when_stepsize_exceeds
            (length(msg) > 0) && (msg = "$msg\n")
            s = s * contract
            msg = "$(msg)Max step size ($(stop_when_stepsize_exceeds)) reached, reducing to $s"
            break
        end
    end
    i = 0
    while fNew > f0 + decrease * s * search_dir_inner # decrease
        i = i + 1
        s = contract * s
        p_new = retract(M, p, η, s, retr)
        fNew = F(M, p_new)
        if i == stop_decreasing_at_step
            (length(msg) > 0) && (msg = "$msg\n")
            msg = "$(msg)Max decrease steps ($(stop_decreasing_at_step)) reached"
            break
        end
        if s < stop_when_stepsize_less
            (length(msg) > 0) && (msg = "$msg\n")
            s = s / contract
            msg = "$(msg)Min step size ($(stop_when_stepsize_less)) exceeded, increasing back to $s"
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
