raw"""
    VectorbundleNewtonState{P,T} <: AbstractManoptSolverState

Is state for the vectorbundle Newton method

# Fields

* 'p': current iterate
* 'X': current Newton Direction
* `stopping_criterion`: stopping criterion
* `retraction_method`:  the retraction to used
* 'vector_transport_method': the vector transport to use
* 'connection_map': connection map used in the Newton equation 
* 

"""
mutable struct VectorbundleNewtonState{
    P,
    T,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    TVM<:AbstractVectorTransportMethod,
    F,
} <: AbstractGradientSolverState
    p::P
    X::T
    stop::TStop
    retraction_method::TRTM
    vector_transport_method::TVM
    connection_map::F
end

function VectorbundleNewtonState(
    M::AbstractManifold,
    p::P,
    connection_map::F;
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stopping_criterion::SC=StopAfterIteration(1000),
    vector_transport_method::VTM=default_vector_transport_method(M, typeof(p)),
) where {
    P,
    T,
    F,
    RM<:AbstractRetractionMethod,
    SC<:StoppingCriterion,
    VTM<:AbstractVectorTransportMethod,
}
    return VectorbundleNewtonState{P,T,SC,RM,VTM,F}(
        p, X, stopping_criterion, retraction_method, vector_transport_method, connection_map
    )
end
