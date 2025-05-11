module ManoptLineSearchesExt

using Manopt
import Manopt: LineSearchesStepsize
using ManifoldsBase
using LineSearches

function (cs::Manopt.LineSearchesStepsize)(
    mp::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    k::Int,
    η=-get_gradient(s);
    fp=get_cost(mp, get_iterate(s)),
    kwargs...,
)
    M = get_manifold(mp)
    p = get_iterate(s)
    X = get_gradient(s)
    p_tmp = copy(M, p)
    X_tmp = copy(M, p, X)
    Y_tmp = copy(M, p, X)
    f = Manopt.get_cost_function(get_objective(mp))
    dphi_0 = real(inner(M, p, X, η))

    # guess initial alpha
    α0 = 1.0

    # perform actual line-search

    function ϕ(α)
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        return f(M, p_tmp)
    end
    function dϕ(α)
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        get_gradient!(mp, X_tmp, p_tmp)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, cs.vector_transport_method)
        return real(inner(M, p_tmp, X_tmp, Y_tmp))
    end
    function ϕdϕ(α)
        # TODO: optimize?
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        get_gradient!(mp, X_tmp, p_tmp)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, cs.vector_transport_method)
        phi = f(M, p_tmp)
        dphi = real(inner(M, p_tmp, X_tmp, Y_tmp))
        return (phi, dphi)
    end

    α, fp = cs.linesearch(ϕ, dϕ, ϕdϕ, α0, fp, dphi_0)
    return α
end

end
