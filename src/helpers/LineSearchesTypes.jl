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

$(_kwargs([:retraction_method, :vector_transport_method]))
"""
mutable struct LineSearchesStepsize{
        TLS, TIG <: AbstractInitialLinesearchGuess, TRM <: AbstractRetractionMethod, TVTM <: AbstractVectorTransportMethod, TF <: Real,
    } <: Stepsize
    linesearch::TLS
    initial_guess::TIG
    retraction_method::TRM
    vector_transport_method::TVTM
    last_stepsize::TF
end
function LineSearchesStepsize(
        M::AbstractManifold,
        linesearch;
        initial_guess::AbstractInitialLinesearchGuess = ConstantInitialGuess(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(
            M
        ),
        last_stepsize::Real = NaN,
    )
    return LineSearchesStepsize(
        linesearch;
        initial_guess = initial_guess,
        retraction_method = retraction_method,
        vector_transport_method = vector_transport_method,
        last_stepsize = last_stepsize
    )
end
function LineSearchesStepsize(
        linesearch;
        initial_guess::AbstractInitialLinesearchGuess = ConstantInitialGuess(),
        retraction_method::AbstractRetractionMethod = ExponentialRetraction(),
        vector_transport_method::AbstractVectorTransportMethod = ParallelTransport(),
        last_stepsize::Real = NaN,
    )
    return LineSearchesStepsize{
        typeof(linesearch), typeof(initial_guess), typeof(retraction_method), typeof(vector_transport_method), typeof(last_stepsize),
    }(
        linesearch, initial_guess, retraction_method, vector_transport_method, last_stepsize
    )
end

function Base.show(io::IO, cs::LineSearchesStepsize)
    return print(
        io,
        "LineSearchesStepsize($(cs.linesearch); initial_guess=$(cs.initial_guess), retraction_method=$(cs.retraction_method), vector_transport_method=$(cs.vector_transport_method), last_stepsize=$(cs.last_stepsize))",
    )
end
