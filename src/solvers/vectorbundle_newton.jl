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
    Pr,
    St,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    TVM<:AbstractVectorTransportMethod,
    F,
} <: AbstractGradientSolverState
    p::P
    X::T
    sub_problem::Pr
    sub_state::St
    stop::TStop
    retraction_method::TRTM
    vector_transport_method::TVM
    connection_map::F
end

function VectorbundleNewtonState(
    M::AbstractManifold,
    E::AbstractManifold,
    F, #bundle_map
    connection_map::CM,
    p::P,
    sub_problem::Pr,
    sub_state::Op;
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stopping_criterion::SC=StopAfterIteration(1000),
    vector_transport_method::VTM=default_vector_transport_method(E, typeof(F(p))),
) where {
    P,
    T,
    Pr,
    Op,
    CM,
    RM<:AbstractRetractionMethod,
    SC<:StoppingCriterion,
    VTM<:AbstractVectorTransportMethod,
}
    return VectorbundleNewtonState{P,T,Pr,Op,SC,RM,VTM,CM}(
        p, X, sub_problem, sub_state, stopping_criterion, retraction_method, vector_transport_method, connection_map
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

"""
"""
struct VectorbundleManoptProblem{TM<:AbstractManifold,TV<:AbstractManifold,O<:AbstractManifoldObjective} <:
    AbstractManoptProblem{TM}
 manifold::TM
 vectorbundle::TV
 objective::O
end

"""
"""
get_vectorbundle(vbp::VectorbundleManoptProblem) = vbp.vectorbundle

get_manifold(vbp::VectorbundleManoptProblem) = vbp.manifold

function get_objective(vbp::VectorbundleManoptProblem, recursive=false)
    return recursive ? get_objective(vbp.objective, true) : vbp.objective
end

function initialize_solver!(::VectorbundleManoptProblem, s::VectorbundleNewtonState)
    return s
end

function step_solver!(mp::VectorbundleManoptProblem, s::VectorbundleNewtonState, k)
    # compute Newton direction
    E = get_manifold(mp) # vector bundle (codomain of F)
    o = get_objective(mp)
    # We need a representation of the equation system (use basis of tangent spaces or constraint representation of the tangent space -> augmented system)

    # TODO: parse parameters to sub_state
    solve!(s.sub_problem, s.sub_state)
    s.X = get_solver_result(s.sub_state)
    # retract
    retract!(get_manifold(mp), s.p, s.p, s.X, s.retraction_method)
    return s
end

function step_solver!(mp::VectorbundleManoptProblem, s::VectorbundleNewtonState{P,T,PR,AllocatingEvaluation}, k) where {P,T,PR}
    # compute Newton direction
    E = get_manifold(mp) # vector bundle (codomain of F)
    o = get_objective(mp)
    # We need a representation of the equation system (use basis of tangent spaces or constraint representation of the tangent space -> augmented system)
    s.X = s.sub_problem(mp, s, k)
    # retract
    retract!(get_manifold(mp), s.p, s.p, s.X, s.retraction_method)
    return s
end

function step_solver!(mp::VectorbundleManoptProblem, s::VectorbundleNewtonState{P,T,PR,InplaceEvaluation}, k) where {P,T,PR}
    # compute Newton direction
    E = get_manifold(mp) # vector bundle (codomain of F)
    o = get_objective(mp)
    # We need a representation of the equation system (use basis of tangent spaces or constraint representation of the tangent space -> augmented system)
    s.sub_problem(mp, s.X, s, k)
    # retract
    retract!(get_manifold(mp), s.p, s.p, s.X, s.retraction_method)
    return s
end
