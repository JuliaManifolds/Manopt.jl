module ManoptLineSearchesExt

using Manopt
import Manopt: LineSearchesStepsize
using ManifoldsBase
using LineSearches

function (cs::Manopt.LineSearchesStepsize)(
    mp::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    k::Int,
    η=(-get_gradient(s));
    fp=get_cost(mp, get_iterate(s)),
    kwargs...,
)
    M = get_manifold(mp)
    p = get_iterate(s)
    X_tmp = zero_vector(M, p)
    p_tmp = copy(M, p)
    Y_tmp = zero_vector(M, p)
    f = Manopt.get_cost_function(get_objective(mp))
    dphi_0 = get_differential(mp, p, η; Y=X_tmp)

    # guess initial alpha
    α0 = 1.0

    # perform actual line-search

    function ϕ(α)
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        return f(M, p_tmp)
    end
    function dϕ(α)
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, cs.vector_transport_method)
        return get_differential(mp, p_tmp, Y_tmp; Y=X_tmp)
    end
    function ϕdϕ(α)
        ManifoldsBase.retract_fused!(M, p_tmp, p, η, α, cs.retraction_method)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, cs.vector_transport_method)
        return Manopt.get_cost_and_differential(mp, p_tmp, Y_tmp; Y=X_tmp)
    end

    α, fp = cs.linesearch(ϕ, dϕ, ϕdϕ, α0, fp, dphi_0)
    return α
end

end
