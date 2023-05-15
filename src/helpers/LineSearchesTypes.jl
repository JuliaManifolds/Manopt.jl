
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
or [`MoreThuente`](https://julianlsolvers.github.io/LineSearches.jl/latest/reference/linesearch.html#LineSearches.MoreThuente)).
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
