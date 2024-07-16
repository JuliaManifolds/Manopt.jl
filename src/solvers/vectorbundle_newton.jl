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

# Constructor

    VectorbundleNewtonState(M, E, F, connection_map, p=rand(M); kwargs...)

# Input

* 'M': domain manifold
* 'E': range vector bundle
* 'F': bundle map ``F:\mathcal M \to \mathcal E`` from Newton's method
* 'connection_map': connection map to compute a generlalized covariant derivative of ``F``
* 'p': initial point

# Keyword arguments

* X=zero_vector(M, p)
* `retraction_method=``default_retraction_method`(M, typeof(p)),
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(1000)``,
* `vector_transport_method=``default_vector_transport_method`(E, typeof(F(p)))

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
    E::AbstractManifold,
    F, #bundle_map
    connection_map::F,
    p::P;
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stopping_criterion::SC=StopAfterIteration(1000),
    vector_transport_method::VTM=default_vector_transport_method(E, typeof(F(p))),
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

@doc raw"""
    VectorbundleObjective{T<:AbstractEvaluationType} <: AbstractManifoldObjective{T}

specify an objective containing a cost and its gradient

# Fields

* `bundle_map!!`:       a mapping ``F: \mathcal M → \mathcal E`` into a vector bundle
* `derivative!!`: the derivative ``F': T\mathcal M → T\mathcal E`` of the bundle map ``F``.

# Constructors
    VectorbundleObjective(bundle_map, derivative; evaluation=AllocatingEvaluation())

"""
struct VectorbundleObjective{T<:AbstractEvaluationType,C,G} <:
       AbstractManifoldGradientObjective{T,C,G}
    bundle_map!!::C
    derivative!!::G
end
function VectorbundleObjective(
    bundle_map::C, derivative::G; evaluation::E=AllocatingEvaluation()
) where {C,G,E<:AbstractEvaluationType}
    return VectorbundleObjective{E,C,G}(bundle_map, derivative)
end

# get_bundle_map, get_derivative similiar to get_gradient 

# We need a VectorbundleProblem with domain M, codomain vector bundle E, and an objective

function initialize_solver!(::AbstractManoptProblem, s::VectorbundleNewtonState)
    return s
end
function step_solver!(mp::AbstractManoptProblem, s::VectorbundleNewtonState, k)
    # compute Newton direction
    E = get_manifold(mp) # vector bundle (codomain of F)
    o = get_objective(mp)
    # We need a representation of the equation system (use basis of tangent spaces or constraint representation of the tangent space -> augmented system)

    # retract
    retract!(get_manifold(mp), s.p, s.p, s.X, s.retraction_method)
    return s
end
