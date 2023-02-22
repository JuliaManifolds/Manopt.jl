
"""
    LineSearchesStepsize <: Stepsize

Wrapper for LineSearches.jl.
"""
struct LineSearchesStepsize{
    TLS,TRM<:AbstractRetractionMethod,TVTM<:AbstractVectorTransportMethod
} <: Stepsize
    linesearch::TLS
    retraction_method::TRM
    vector_transport_method::TVTM
end
function LineSearchesStepsize(
    M::AbstractManifold,
    ls;
    retraction_method=default_retraction_method(M),
    vector_transport_method=default_vector_transport_method(M),
)
    return LineSearchesStepsize{
        typeof(ls),typeof(retraction_method),typeof(vector_transport_method)
    }(
        ls, retraction_method, vector_transport_method
    )
end

function (cs::LineSearchesStepsize)(
    mp::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    i::Int,
    η=-get_gradient(mp, get_iterate(s));
    fp=get_cost(mp, get_iterate(s)),
    kwargs...,
)
    M = get_manifold(mp)
    p = get_iterate(s)
    p_tmp = copy(M, p)
    X_tmp = copy(M, p, η)
    Y_tmp = copy(M, p, η)
    f = get_cost_function(get_objective(mp))
    function ϕ(α)
        retract!(M, p_tmp, p, η, α, cs.retraction_method)
        return f(M, p_tmp)
    end
    function dϕ(α)
        retract!(M, p_tmp, p, η, α, cs.retraction_method)
        get_gradient!(mp, X_tmp, p_tmp)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, cs.vector_transport_method)
        return real(inner(M, p_tmp, Y_tmp, Y_tmp))
    end
    function ϕdϕ(α)
        # TODO: optimize?
        retract!(M, p_tmp, p, η, α, cs.retraction_method)
        get_gradient!(mp, X_tmp, p_tmp)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, cs.vector_transport_method)
        phi = f(M, p_tmp)
        dphi = real(inner(M, p_tmp, Y_tmp, Y_tmp))
        return (phi, dphi)
    end
    dϕ_0 = -norm(M, p, η)^2
    α, fp = cs.linesearch(ϕ, dϕ, ϕdϕ, 1.0, fp, dϕ_0)
    return α
end
#get_initial_stepsize(s::LineSearchesStepsize) = s.length
#show(io::IO, cs::LineSearchesStepsize) = print(io, "LineSearchesStepsize()")
