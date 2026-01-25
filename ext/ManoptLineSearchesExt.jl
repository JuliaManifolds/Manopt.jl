module ManoptLineSearchesExt

using Manopt
import Manopt: LineSearchesStepsize
using ManifoldsBase
using LineSearches

Manopt.linesearches_get_max_alpha(ls::LineSearches.HagerZhang) = ls.alphamax
Manopt.linesearches_get_max_alpha(ls::LineSearches.MoreThuente) = ls.alphamax

function Manopt.linesearches_set_max_alpha(ls::LineSearches.HagerZhang{T, Tm}, max_alpha::T) where {T, Tm}
    return HagerZhang{T, Tm}(
        delta = ls.delta,
        sigma = ls.sigma,
        alphamax = max_alpha,
        rho = ls.rho,
        epsilon = ls.epsilon,
        gamma = ls.gamma,
        linesearchmax = ls.linesearchmax,
        psi3 = ls.psi3,
        display = ls.display,
        mayterminate = ls.mayterminate,
        cache = ls.cache,
        check_flatness = ls.check_flatness,
    )
end
function Manopt.linesearches_set_max_alpha(ls::LineSearches.MoreThuente{T}, max_alpha::T) where {T}
    return MoreThuente{T}(
        f_tol = ls.f_tol,
        gtol = ls.gtol,
        x_tol = ls.x_tol,
        alphamin = ls.alphamin,
        alphamax = max_alpha,
        maxfev = ls.maxfev,
        cache = ls.cache
    )
end

function (cs::Manopt.LineSearchesStepsize)(
        mp::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        k::Int,
        η = (-get_gradient(s));
        fp = get_cost(mp, get_iterate(s)),
        kwargs...,
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    X_tmp = zero_vector(M, p)
    p_tmp = copy(M, p)
    Y_tmp = zero_vector(M, p)
    f = Manopt.get_cost_function(get_objective(mp))
    dphi_0 = get_differential(mp, p, η; Y = X_tmp)

    # guess initial alpha
    α0 = cs.initial_guess(mp, s, k, cs.last_stepsize, η; lf0 = fp, Dlf0 = dphi_0)

    # handle stepsize limit
    local ls
    if :stop_when_stepsize_exceeds in keys(kwargs)
        new_max_alpha = min(
            kwargs[:stop_when_stepsize_exceeds],
            linesearches_get_max_alpha(cs.linesearch),
        )
        ls = linesearches_set_max_alpha(cs.linesearch, new_max_alpha)
    else
        ls = cs.linesearch
    end

    # perform actual line-search

    function ϕ(α)
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        return f(M, p_tmp)
    end
    function dϕ(α)
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, cs.vector_transport_method)
        return get_differential(mp, p_tmp, Y_tmp; Y = X_tmp)
    end
    function ϕdϕ(α)
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, cs.vector_transport_method)
        return Manopt.get_cost_and_differential(mp, p_tmp, Y_tmp; Y = X_tmp)
    end

    α, fp = ls(ϕ, dϕ, ϕdϕ, α0, fp, dphi_0)
    cs.last_stepsize = α
    return α
end

end
