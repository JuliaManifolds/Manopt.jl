
"""
    LineSearchesStepsize <: Stepsize

Wrapper for line searches available in the `LineSearches.jl` library.

## Constructors

    LineSearchesStepsize(
        M::AbstractManifold,
        linesearch;
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(M),
    )
    LineSearchesStepsize(
        linesearch;
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    )

Wrap `linesearch` (for example [`HagerZhang`](https://julianlsolvers.github.io/LineSearches.jl/latest/reference/linesearch.html#LineSearches.HagerZhang)
or [`MoreThuente`)(https://julianlsolvers.github.io/LineSearches.jl/latest/reference/linesearch.html#LineSearches.MoreThuente)).
The initial step selection from Lineseaches.jl is not yet supported and the value 1.0 is used.
The retraction used for determining the line along which the search is performed can be
 provided as `retraction_method`. Gradient vectors are transported between points using
`vector_transport_method`.
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
    linesearch;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M
    ),
)
    return LineSearchesStepsize(
        linesearch;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    )
end
function LineSearchesStepsize(
    linesearch;
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
)
    return LineSearchesStepsize{
        typeof(linesearch),typeof(retraction_method),typeof(vector_transport_method)
    }(
        linesearch, retraction_method, vector_transport_method
    )
end

function (cs::LineSearchesStepsize)(
    mp::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    i::Int,
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
    f = get_cost_function(get_objective(mp))
    dphi_0 = real(inner(M, p, X, η))

    # guess initial alpha
    α0 = 1.0

    # perform actual linesearch

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

    try
        α, fp = cs.linesearch(ϕ, dϕ, ϕdϕ, α0, fp, dphi_0)
        return α
    catch ex
        if isa(ex, LineSearches.LineSearchException)
            # maybe indicate failure?
            return zero(dphi_0)
        else
            rethrow(ex)
        end
    end
end

function show(io::IO, cs::LineSearchesStepsize)
    return print(
        io,
        "LineSearchesStepsize($(cs.linesearch); retraction_method=$(cs.retraction_method), vector_transport_method=$(cs.vector_transport_method))",
    )
end
