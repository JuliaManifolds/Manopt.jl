raw"""
    VectorbundleNewtonState{P,T} <: AbstractManoptSolverState

Is state for the vectorbundle Newton method

# Fields

* 'p': current iterate
* 'X': current Newton Direction
* `stopping_criterion`: stopping criterion
* `retraction_method`:  the retraction to used
* 'vector_transport_method': the vector transport to use

# Constructor

    VectorbundleNewtonState(M, E, F, connection_map, p=rand(M); kwargs...)

# Input

* 'M': domain manifold
* 'E': range vector bundle
* 'F': bundle map ``F:\mathcal M \to \mathcal E`` from Newton's method
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
} <: AbstractGradientSolverState
    p::P
    X::T
    sub_problem::Pr
    sub_state::St
    stop::TStop
    retraction_method::TRTM
    vector_transport_method::TVM
end

function VectorbundleNewtonState(
    M::AbstractManifold,
    E::AbstractManifold,
    F, #bundle_map
    p::P,
    sub_problem::Pr,
    sub_state::Op;
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stopping_criterion::SC=StopAfterIteration(1000),
    vector_transport_method::VTM=default_vector_transport_method(E, typeof(F(M, p))),
) where {
    P,
    T,
    Pr,
    Op,
    RM<:AbstractRetractionMethod,
    SC<:StoppingCriterion,
    VTM<:AbstractVectorTransportMethod,
}
    return VectorbundleNewtonState{P,T,Pr,Op,SC,RM,VTM}(
        p,
        X,
        sub_problem,
        sub_state,
        stopping_criterion,
        retraction_method,
        vector_transport_method,
    )
end

@doc raw"""
    VectorbundleObjective{T<:AbstractEvaluationType} <: AbstractManifoldObjective{T}

specify an objective containing a cost and its gradient

# Fields

* `bundle_map!!`:       a mapping ``F: \mathcal M → \mathcal E`` into a vector bundle
* `derivative!!`: the derivative ``F': T\mathcal M → T\mathcal E`` of the bundle map ``F``.
* 'connection_map': connection map used in the Newton equation

# Constructors
    VectorbundleObjective(bundle_map, derivative, connection_map; evaluation=AllocatingEvaluation())

"""
struct VectorbundleObjective{T<:AbstractEvaluationType,C,G,F} <:
       AbstractManifoldGradientObjective{T,C,G}
    bundle_map!!::C
    derivative!!::G
    connection_map!!::F
end
# TODO: Eventuell zweiter Parameter (a) Tensor/Matrix darstellung vs (b) action Darstellung
# oder über einen letzten parameter (a) ohne (b) mit

function VectorbundleObjective(
    bundle_map::C, derivative::G, connection_map::F; evaluation::E=AllocatingEvaluation()
) where {C,G,F,E<:AbstractEvaluationType}
    return VectorbundleObjective{E,C,G,F}(bundle_map, derivative, connection_map)
end


raw"""
    VectorbundleManoptProblem{
    TM<:AbstractManifold,TV<:AbstractManifold,O<:AbstractManifoldObjective
}

Model a vector bundle problem, that consists of the domain manifold ``\mathcal M`` that is a AbstractManifold, the range vector bundle ``\mathcal E`` and an VectorbundleObjective
"""
# sollte da O nicht ein VectorbundleObjective sein?
struct VectorbundleManoptProblem{
    TM<:AbstractManifold,TV<:AbstractManifold,O<:AbstractManifoldObjective
} <: AbstractManoptProblem{TM}
    manifold::TM
    vectorbundle::TV
    objective::O
end

raw"""
    get_vectorbundle(vbp::VectorbundleManoptProblem)

    returns the range vector bundle stored within a [`VectorbundleManoptProblem`](@ref)
"""
get_vectorbundle(vbp::VectorbundleManoptProblem) = vbp.vectorbundle

raw"""
    get_manifold(vbp::VectorbundleManoptProblem)

    returns the domain manifold stored within a [`VectorbundleManoptProblem`](@ref)
"""
get_manifold(vbp::VectorbundleManoptProblem) = vbp.manifold


raw"""
    get_objective(mp::VectorbundleManoptProblem, recursive=false)

return the objective [`VectorbundleObjective`](@ref) stored within an [`VectorbundleManoptProblem`](@ref).
If `recursive is set to true, it additionally unwraps all decorators of the objective`
"""

function get_objective(vbp::VectorbundleManoptProblem, recursive=false)
    return recursive ? get_objective(vbp.objective, true) : vbp.objective
end

raw"""
    get_bundle_map(M, E, vbo::VectorbundleObjective, p)
    get_bundle_map!(M, E, X, vbo::VectorbundleObjective, p)
    get_bundle_map(P::VectorBundleManoptProblem, p)
    get_bundle_map!(P::VectorBundleManoptProblem, X, p)

    Evaluate the vector field ``F: \mathcal M → \mathcal E`` at ``p``
"""
function get_bundle_map(M, E, vbo::VectorbundleObjective, p)
    return vbo.bundle_map!!(M, p)
end
function get_bundle_map(M, E, vbo::VectorbundleObjective{InplaceEvaluation}, p)
    X = zero_vector(E, p)
    return vbo.bundle_map!!(M, X, p)
end
function get_bundle_map(vpb::VectorbundleManoptProblem, p)
    return get_bundle_map(
        get_manifold(vpb), get_vectorbundle(vpb), get_objective(vpb, true), p
    )
end
function get_bundle_map!(M, E, X, vbo::VectorbundleObjective{AllocatingEvaluation}, p)
    copyto!(E, p, X, vbo.bundle_map!!(M, p))
    return X
end
function get_bundle_map!(M, E, X, vbo::VectorbundleObjective{InplaceEvaluation}, p)
    vbo.bundle_map!!(M, X, p)
    return X
end
function get_bundle_map!(vbp::VectorbundleManoptProblem, X, p)
    get_bundle_map!(
        get_manifold(vbp), get_vectorbundle(vb, p), X, get_objective(vbp, true), p
    )
    return X
end

# As a tensor not an action -> for now just matrix representation / tensor.
raw"""
    get_derivative(M, E, vbo::VectorbundleObjective, p)
    get_derivative(P::VectorBundleManoptProblem, p)

    Evaluate the vector field ``F'(p): T_p\mathcal M → T_{F(p)}\mathcal E`` at ``p``
    in a matrix form (TODO?? (a) matrix, (b) matrix action (c) something nice for Q?)
"""
function get_derivative(M, E, vbo::VectorbundleObjective, p)
    return vbo.derivative!!(M, p)
end
function get_derivative(vpb::VectorbundleManoptProblem, p)
    return get_derivative(
        get_manifold(vpb), get_vectorbundle(vpb), get_objective(vpb, true), p
    )
end

# As a tensor not an action -> for now just matrix representation / tensor.
raw"""
    get_connection_map(E, vbo::VectorbundleObjective, q)
    get_connection_map(vbp::VectorbundleManoptProblem, q)

Returns in matrix form the connection map ``Q_q: T_q\mathcal E → E_{π(q)}``
"""
function get_connection_map(E, vbo::VectorbundleObjective, q)
    return vbo.connection_map!!(E, q)
end
function get_connection_map(vbp::VectorbundleManoptProblem, q)
    return get_connection_map(get_vectorbundle(vbp), get_objective(vbp, true), q)
end

function initialize_solver!(::VectorbundleManoptProblem, s::VectorbundleNewtonState)
    return s
end

function step_solver!(mp::VectorbundleManoptProblem, s::VectorbundleNewtonState, k)
    # compute Newton direction
    E = get_vectorbundle(mp) # vector bundle (codomain of F)
    o = get_objective(mp)
    # We need a representation of the equation system (use basis of tangent spaces or constraint representation of the tangent space -> augmented system)

    # TODO: parse parameters to sub_state
    solve!(s.sub_problem, s.sub_state)
    s.X = get_solver_result(s.sub_state)
    # retract
    retract!(get_manifold(mp), s.p, s.p, s.X, s.retraction_method)
    return s
end

function step_solver!(
    mp::VectorbundleManoptProblem,
    s::VectorbundleNewtonState{P,T,PR,AllocatingEvaluation},
    k,
) where {P,T,PR}
    # compute Newton direction
    E = get_vectorbundle(mp) # vector bundle (codomain of F)
    o = get_objective(mp)
    # We need a representation of the equation system (use basis of tangent spaces or constraint representation of the tangent space -> augmented system)
    s.X = s.sub_problem(mp, s, k)
    # retract
    retract!(get_manifold(mp), s.p, s.p, s.X, s.retraction_method)
    return s
end

function step_solver!(
    mp::VectorbundleManoptProblem, s::VectorbundleNewtonState{P,T,PR,InplaceEvaluation}, k
) where {P,T,PR}
    # compute Newton direction
    E = get_vectorbundle(mp) # vector bundle (codomain of F)
    o = get_objective(mp)
    # We need a representation of the equation system (use basis of tangent spaces or constraint representation of the tangent space -> augmented system)
    s.sub_problem(mp, s.X, s, k)
    # retract
    retract!(get_manifold(mp), s.p, s.p, s.X, s.retraction_method)
    return s
end
