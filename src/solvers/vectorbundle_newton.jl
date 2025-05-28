raw"""
    VectorbundleNewtonState{P,T} <: AbstractManoptSolverState

Is state for the vectorbundle Newton method

# Fields

* 'p': current iterate
* 'X': current Newton Direction
* `stopping_criterion`: stopping criterion
* `stepsize`: damping factor for the Newton direction
* `retraction_method`:  the retraction to use in the Newton update

# Constructor

    VectorbundleNewtonState(M, E, F, connection_map, p=rand(M); kwargs...)

# Input

* 'M': domain manifold
* 'E': range vector bundle
* 'F': bundle map ``F:\mathcal M \to \mathcal E`` from Newton's method
* 'p': initial point

# Keyword arguments

* `X=`zero_vector(M, p)
* `retraction_method=``default_retraction_method`(M, typeof(p)),
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(1000)``,
* `stepsize=`1.0

"""
#TODO: sub_problem, sub_state, A, b dokumentieren?
mutable struct VectorbundleNewtonState{
    P,
    #P2,
    T,
    Pr,
    St,
    NM,
    Nrhs,
    TStop<:StoppingCriterion,
    TStep<:Stepsize,
    TRTM<:AbstractRetractionMethod
} <: AbstractGradientSolverState
    p::P
    p_trial::P
    X::T
    sub_problem::Pr
    sub_state::St
    A::NM
    b::Nrhs
    stop::TStop
    stepsize::TStep
    retraction_method::TRTM
end

function VectorbundleNewtonState(
    M::AbstractManifold,
    E::AbstractManifold,
    p::P,
    sub_problem::Pr,
    sub_state::Op,
    A::NM,
    b::Nrhs;
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stopping_criterion::SC=StopAfterIteration(1000),
    stepsize::S=default_stepsize(M, VectorbundleNewtonState)
) where {
    P,
    T,
    Pr,
    Op,
    NM,
    Nrhs,
    RM<:AbstractRetractionMethod,
    SC<:StoppingCriterion,
    S<:Stepsize,
}
    return VectorbundleNewtonState{P,T,Pr,Op,NM,Nrhs,SC,S,RM}(
        p,
        copy(M, p),
        X,
        sub_problem,
        sub_state,
        A,
        b,
        stopping_criterion,
        stepsize,
        retraction_method
    )
end

mutable struct AffineCovariantStepsize{T, R<:Real} <: Stepsize
    alpha::T
    theta::R
    theta_des::R
    theta_acc::R
    last_stepsize::R
end
function AffineCovariantStepsize(
    M::AbstractManifold=DefaultManifold(2);
    stepsize=1.0,
    theta=1.3,
    theta_des=0.1,
    theta_acc=1.1*theta_des,
    last_stepsize = 1.0
)
    return AffineCovariantStepsize{typeof(stepsize), typeof(theta)}(stepsize, theta, theta_des, theta_acc, last_stepsize)
end

function (acs::AffineCovariantStepsize)(
    amp::AbstractManoptProblem, ams::VectorbundleNewtonState, ::Any, args...; kwargs...
)
    acs.alpha = 1.0
    acs.theta = 1.3
    alpha_new = 1.0
    b = copy(ams.b)
    while acs.theta > acs.theta_acc && acs.alpha > 1e-10
        acs.alpha = copy(alpha_new)
        X_alpha = acs.alpha * ams.X
        M = get_manifold(amp)
        retract!(M, ams.p_trial, ams.p, X_alpha, ams.retraction_method)

        rhs_next = amp.NewtonEquation(M, get_vectorbundle(amp), ams.p, ams.p_trial)
        rhs_simplified = rhs_next - (1.0 - acs.alpha)*b
        ams.b .= rhs_simplified

        simplified_newton = ams.sub_problem(amp, ams, 1)
        acs.theta = norm(simplified_newton)/norm(ams.X)
        alpha_new = min(1.0, ((acs.alpha*acs.theta_des)/(acs.theta)))
        if acs.alpha < 1e-15
            println("Newton's method failed")
            return
        end
    end
    ams.b .= b
    acs.last_stepsize = acs.alpha
    return acs.alpha
end
get_initial_stepsize(s::AffineCovariantStepsize) = 1.0

function get_last_stepsize(step::AffineCovariantStepsize, ::Any...)
    return step.last_stepsize
end

function default_stepsize(M::AbstractManifold, ::Type{VectorbundleNewtonState})
    #return AffineCovariantStepsize(M)
    return _produce_type(ConstantLength(1.0), M)
end

function show(io::IO, vbns::VectorbundleNewtonState)
    i = get_count(vbns, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(vbns.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Vectorbundle Newton Method
    $Iter
    ## Parameters
    * retraction method: $(vbns.retraction_method)
    * step size: $(vbns.stepsize)

    ## Stopping criterion

    $(status_summary(vbns.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

#TODO: Objective gibt es so nicht mehr. Entweder entfernen oder überlegen, ob man da noch was sinnvolles machen kann.
@doc raw"""
    VectorbundleObjective{T<:AbstractEvaluationType} <: AbstractManifoldObjective{T}

specify an objective containing a vector bundle map, its derivative, and a connection map

# Fields

* `bundle_map!!`:       a mapping ``F: \mathcal M → \mathcal E`` into a vector bundle
* `derivative!!`: the derivative ``F': T\mathcal M → T\mathcal E`` of the bundle map ``F``.
* 'connection_map!!': connection map used in the Newton equation

# Constructors
    VectorbundleObjective(bundle_map, derivative, connection_map; evaluation=AllocatingEvaluation())

"""
mutable struct VectorbundleObjective{T<:AbstractEvaluationType,C,G,F} <:
       AbstractManifoldGradientObjective{T,C,G}
    bundle_map!!::C
    derivative!!::G
    connection_map!!::F
    scaling::Number
end


function VectorbundleObjective(
    bundle_map::C, derivative::G, connection_map::F; evaluation::E=AllocatingEvaluation()
) where {C,G,F,E<:AbstractEvaluationType}
    return VectorbundleObjective{E,C,G,F}(bundle_map, derivative, connection_map, 1.0)
end

raw"""
    VectorbundleManoptProblem{
    TM<:AbstractManifold,TV<:AbstractManifold,O<:AbstractManifoldObjective
}

Model a vector bundle problem, that consists of the domain manifold ``\mathcal M`` that is a AbstractManifold, the range vector bundle ``\mathcal E`` and the Newton equation ``Q_{F(x)}\circ F'(x) \delta x + F(x) = 0_{p(F(x))}`` given as a functor which returns a representation of the Newton matrix and the right hand side 
"""
struct VectorbundleManoptProblem{
    TM<:AbstractManifold,TV<:AbstractManifold,O
} <: AbstractManoptProblem{TM}
    manifold::TM
    vectorbundle::TV
    NewtonEquation::O # umbenennen: newton_equation
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
    get_newton_equation(mp::VectorbundleManoptProblem)

return the Newton equation [`NewtonEquation`](@ref) stored within an [`VectorbundleManoptProblem`](@ref).
"""

function get_newton_equation(vbp::VectorbundleManoptProblem)
    return vbp.NewtonEquation
end

raw"""
    get_objective(mp::VectorbundleManoptProblem, recursive=false)

return the objective [`VectorbundleObjective`](@ref) stored within an [`VectorbundleManoptProblem`](@ref).
If `recursive` is set to true, it additionally unwraps all decorators of the `objective`
"""

#function get_objective(vbp::VectorbundleManoptProblem, recursive=false)
#    return recursive ? get_objective(vbp.objective, true) : vbp.objective
#end


raw"""
    get_submersion(M, p)

    ```math
    c: ℝ^n → ℝ
    ```

    returns the submersion at point ``p`` which defines the manifold
    ``\mathcal M = \{p \in \bbR^n : c(p) = 0 \}``
"""
function get_submersion(M::AbstractManifold, p) end

raw"""
    get_submersion_derivative(M,p)

    returns the derivative ``c'(p) : T_p\mathcal{M} \to \mathcal R^{n-d}`` of the submersion at point ``p`` which defines the manifold in matrix form
"""
function get_submersion_derivative(M::AbstractManifold, p) end

@doc raw"""
    vectorbundle_newton(M, E, NE, p; kwargs...)
    vectorbundle_newton(M, E, vbo p0; kwargs...) # ich glaub das gibts nicht
    vectorbundle_newton!(M, E, NE, p; kwargs...)
    vectorbundle_newton(M, E, vbo, p0; kwargs...) # ich glaub das gibts nicht

Peform the Newton's method for finding a zero of a mapping `F:\mathcal M \to \mathcal E$ where `\mathcal M` is a manifold and `\mathcal E` is a vector bundle. 
`NE` is a functor which has to return a representation of the Newton matrix `Q_{F(p)}\circ F'(p)` (covariant derivative of `F`) and the right hand side `F(p)` at a point `p\in\mathcal M`. The point `p` denotes the start point. The algorithm can be run in-place of `p`.

"""
vectorbundle_newton(M::AbstractManifold, E::AbstractManifold, args...; kwargs...) #replace type of E with VectorBundle once this is available in ManifoldsBase

function vectorbundle_newton(
    M::AbstractManifold, E::AbstractManifold, NE, p; kwargs...
)
    q = copy(M, p)
    return vectorbundle_newton!(M, E, NE, q; kwargs...)
end

# function vectorbundle_newton!(
#     M::AbstractManifold,
#     E::AbstractManifold,
#     NE,
#     p;
#     evaluation=AllocatingEvaluation(),
#     kwargs...,
# )
#     #vbo = VectorbundleObjective(F, F_prime, Q; evaluation=evaluation)
#     return vectorbundle_newton!(M, E, NE, p; evaluation=evaluation, kwargs...)
# end
# die vermutlich auch weg, weil wir kein objective mehr bauen, das ist glaub ich auch das Warning, das beim Precompiling ausgegeben wird

function vectorbundle_newton!(
    M::AbstractManifold,
    E::AbstractManifold,
    NE::O,
    p::P;
    evaluation=AllocatingEvaluation(),
    sub_problem::Pr=nothing, #TODO: find/implement good default solver
    sub_state::Op=nothing, #TODO: find/implement good default solver
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stopping_criterion::SC=StopAfterIteration(1000),
    stepsize::Union{Stepsize,ManifoldDefaultsFactory}=default_stepsize(
        M, VectorbundleNewtonState
    ),
    kwargs...,
) where {
    O,
    P,
    T,
    Pr,
    Op,
    RM<:AbstractRetractionMethod,
    SC<:StoppingCriterion
}
    # Once we have proper defaults, these checks should be removed
    #isnothing(sub_problem) && error("Please provide a sub_problem")
    #isnothing(sub_state) && error("Please provide a sub_state")
    #dvbo = decorate_objective!(M, vbo; kwargs...)

    vbp = VectorbundleManoptProblem(M, E, NE)
    
    A, b = vbp.NewtonEquation(M, E, p) # das wird in der ersten Iteration nochmal gemacht, kann man A und b irgendwie als nothing setzen?
    vbs = VectorbundleNewtonState(
        M,
        E,
        p,
        sub_problem,
        sub_state,
        A,
        b;
        X=X,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        stepsize=_produce_type(stepsize, M)
    )
    dvbs = decorate_state!(vbs; kwargs...)
    solve!(vbp, dvbs)
    return get_solver_return(dvbs)
end

function initialize_solver!(::VectorbundleManoptProblem, s::VectorbundleNewtonState)
    return s
end

# TODO: anpassen, evtl kann man da was machen, damit A und b im sub_state stehen oder so

# function step_solver!(mp::VectorbundleManoptProblem, s::VectorbundleNewtonState, k)
#     # compute Newton direction
#     #println("Hallo 1")
#     E = get_vectorbundle(mp) # vector bundle (codomain of F)
#     o = get_objective(mp)
#     # We need a representation of the equation system (use basis of tangent spaces or constraint representation of the tangent space -> augmented system)

#     # TODO: pass parameters to sub_state
#     # set_iterate!(s.sub_state, get_manifold(s.sub_problem), zero_vector(N, q)) Set start point x0

#     set_manopt_parameter!(s.sub_problem, :Manifold, :Basepoint, s.p)

#     set_iterate!(s.sub_state, get_manifold(s.sub_problem), zero_vector(get_manifold(s.sub_problem), s.p))
#     #set_iterate!(s.sub_state, get_manifold(mp), zero_vector(get_manifold(mp), s.p))

#     solve!(s.sub_problem, s.sub_state)
#     s.X = get_solver_result(s.sub_state)


#     step = s.stepsize(mp, s, k)

#     # retract
#     retract!(get_manifold(mp), s.p, s.p, s.X, step, s.retraction_method)
#     s.p_trial = copy(get_manifold(mp),s.p)

#     return s
# end

function step_solver!(
    mp::VectorbundleManoptProblem,
    s::VectorbundleNewtonState{P,T,PR,AllocatingEvaluation},
    k,
) where {P,T,PR}
    
    M = get_manifold(mp) # domain manifold
    E = get_vectorbundle(mp) # vector bundle (codomain of F)

    # update Newton matrix and right hand side
    mp.NewtonEquation(M, E, s.A, s.b, s.p)
    #s.A, s.b = mp.NewtonEquation(M, E, s.p)

    # compute Newton direction
    s.X = s.sub_problem(mp, s, k)
    
    #compute a stepsize 
    step = s.stepsize(mp, s, k)
    
    # retract
    retract!(get_manifold(mp), s.p, s.p, s.X, step, s.retraction_method)
    s.p_trial = copy(get_manifold(mp),s.p)

    return s
end

function step_solver!(
    mp::VectorbundleManoptProblem, s::VectorbundleNewtonState{P,T,PR,InplaceEvaluation}, k
) where {P,T,PR}
    
    M = get_manifold(mp) # domain manifold
    E = get_vectorbundle(mp) # vector bundle (codomain of F)
    
    # update Newton matrix and right hand side
    #mp.NewtonEquation(M, E, s.A, s.b, s.p)
    s.A, s.b = mp.NewtonEquation(M, E, s.p)

    # compute Newton direction
    s.sub_problem(mp, s.X, s, k)

    step = s.stepsize(mp, s, k)
    # retract
    retract!(M, s.p, s.p, s.X, step, s.retraction_method)
    s.p_trial = copy(M, s.p)

    return s
end
