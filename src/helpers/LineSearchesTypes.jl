"""
    LineSearchesStepsize <: Stepsize

Wrapper for line searches available in the `LineSearches.jl` library.

## Constructors

    LineSearchesStepsize(M::AbstractManifold, linesearch; kwargs...
    LineSearchesStepsize(
        linesearch;
        retraction_method=ExponentialRetraction(),
        vector_transport_method=ParallelTransport(),
    )

Wrap `linesearch` (for example [`HagerZhang`](https://julianlsolvers.github.io/LineSearches.jl/latest/reference/linesearch.html#LineSearches.HagerZhang)
or [`MoreThuente`](https://julianlsolvers.github.io/LineSearches.jl/latest/reference/linesearch.html#LineSearches.MoreThuente)).
The initial step selection from Linesearches.jl is not yet supported and the value 1.0 is used.

# Keyword Arguments

$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :vector_transport_method))
"""
struct LineSearchesStepsize{
        TLS, TRM <: AbstractRetractionMethod, TVTM <: AbstractVectorTransportMethod,
    } <: Stepsize
    linesearch::TLS
    retraction_method::TRM
    vector_transport_method::TVTM
end
function LineSearchesStepsize(
        M::AbstractManifold,
        linesearch;
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(
            M
        ),
    )
    return LineSearchesStepsize(
        linesearch;
        retraction_method = retraction_method,
        vector_transport_method = vector_transport_method,
    )
end
function LineSearchesStepsize(
        linesearch;
        retraction_method::AbstractRetractionMethod = ExponentialRetraction(),
        vector_transport_method::AbstractVectorTransportMethod = ParallelTransport(),
    )
    return LineSearchesStepsize{
        typeof(linesearch), typeof(retraction_method), typeof(vector_transport_method),
    }(
        linesearch, retraction_method, vector_transport_method
    )
end

function Base.show(io::IO, cs::LineSearchesStepsize)
    return print(
        io,
        "LineSearchesStepsize($(cs.linesearch); retraction_method=$(cs.retraction_method), vector_transport_method=$(cs.vector_transport_method))",
    )
end
