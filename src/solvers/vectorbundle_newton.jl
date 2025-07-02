raw"""
    VectorbundleNewtonState{P,T} <: AbstractManoptSolverState

Is state for the vectorbundle Newton method

# Fields

* 'p': current iterate
* 'p_trial': next iterate needed for simplified Newton (not needed if affine covariant damping is not used to compute stepsizes)
* 'X': current Newton direction
* 'sub_problem': method (closed form solution) that returns the solution of the Newton equation, i.e. the Newton direction 
* 'sub_state': sub_state to sub_problem, in most cases either AllocatingEvaluation() or InplaceEvaluation()
* `stop`: stopping criterion
* `stepsize`: damping factor for the Newton direction
* `retraction_method`:  the retraction to use in the Newton update

# Constructor

    VectorbundleNewtonState(M, E, p, sub_problem, sub_state; kwargs...)

# Input

* 'M': domain manifold
* 'E': range vector bundle
* 'p': initial point
* 'sub_problem': method (closed form solution) that gets the [`VectorbundleManoptProblem`](@ref) and the [`VectorbundleNewtonState`](@ref) and returns the solution of the Newton equation, i.e. the Newton direction `X`
* 'sub_state': sub_state to sub_problem, in most cases either AllocatingEvaluation() or InplaceEvaluation()


# Keyword arguments

* `X=`zero_vector(M, p)
* `retraction_method=``default_retraction_method`(M, typeof(p)),
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(1000)``,
* `stepsize=`1.0

"""

mutable struct VectorbundleNewtonState{
    P,
    T,
    Pr,
    St,
    TStop<:StoppingCriterion,
    TStep<:Stepsize,
    TRTM<:AbstractRetractionMethod
} <: AbstractGradientSolverState
    p::P
    p_trial::P
    X::T
    sub_problem::Pr
    sub_state::St
    stop::TStop
    stepsize::TStep
    retraction_method::TRTM
end

function VectorbundleNewtonState(
    M::AbstractManifold,
    E::AbstractManifold,
    p::P,
    sub_problem::Pr,
    sub_state::Op;
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stopping_criterion::SC=StopAfterIteration(1000),
    stepsize::S=default_stepsize(M, VectorbundleNewtonState)
) where {
    P,
    T,
    Pr,
    Op,
    RM<:AbstractRetractionMethod,
    SC<:StoppingCriterion,
    S<:Stepsize,
}
    return VectorbundleNewtonState{P,T,Pr,Op,SC,S,RM}(
        p,
        copy(M, p),
        X,
        sub_problem,
        sub_state,
        stopping_criterion,
        stepsize,
        retraction_method
    )
end

# TODO: paper zitieren, unten evtl ManoptExamples-Beispiel zitieren, falls möglich
@doc raw""" 
    AffineCovariantStepsize <: Stepsize

    A functor to provide an affine covariant stepsize generalizing the idea of following Newton paths introduced by [TODO](@cite). It can be used to derive a damped Newton method. The step sizes (damping factors) are computed by a predictor-corrector-loop using an affine covariant quantity ``\theta`` to measure local convergence.

    # Fields

    * `alpha`: the step size
    * `theta`: quantity that measures local convergence of Newton's method
    * `theta_des`: desired theta
    * `theta_acc`: acceptable theta
    * `last_stepsize`: last computed step size (helper)

    # Constructor 

        AffineCovariantStepsize(M::AbstractManifold=DefaultManifold(2);
        stepsize=1.0,
        theta=1.3,
        theta_des=0.1,
        theta_acc=1.1*theta_des,
        last_stepsize = 1.0
        )

        initializes all fields, where none of them is mandatory. The length is set to ``1.0``.

    Since the computation of the convergence monitor ``\theta`` requires simplified Newton directions a method for computing them has to be provided. This should be implemented as a method of the ``newton_equation`` getting ``(M, VB, p, p_trial)`` as parameters and returning a representation of the (transported) ``F(p_{trial})``.

"""

mutable struct AffineCovariantStepsize{T, R<:Real} <: Stepsize
    alpha::T
    theta::R
    theta_des::R
    theta_acc::R
    last_stepsize::R
end
function AffineCovariantStepsize(
    M::AbstractManifold=DefaultManifold(2);
    alpha=1.0,
    theta=1.3,
    theta_des=0.1,
    theta_acc=1.1*theta_des,
    last_stepsize = 1.0
)
    return AffineCovariantStepsize{typeof(stepsize), typeof(theta)}(alpha, theta, theta_des, theta_acc, last_stepsize)
end

function (acs::AffineCovariantStepsize)(
    amp::AbstractManoptProblem, ams::VectorbundleNewtonState, ::Any, args...; kwargs...
)
    acs.alpha = 1.0
    acs.theta = 1.3
    alpha_new = 1.0
    b = copy(amp.newton_equation.b)
    while acs.theta > acs.theta_acc && acs.alpha > 1e-10
        acs.alpha = copy(alpha_new)
        X_alpha = acs.alpha * ams.X
        M = get_manifold(amp)
        retract!(M, ams.p_trial, ams.p, X_alpha, ams.retraction_method)

        rhs_next = amp.newton_equation(M, get_vectorbundle(amp), ams.p, ams.p_trial)
        rhs_simplified = rhs_next - (1.0 - acs.alpha)*b
        amp.newton_equation.b .= rhs_simplified

        simplified_newton = ams.sub_problem(amp, ams)
        acs.theta = norm(simplified_newton)/norm(ams.X)
        alpha_new = min(1.0, ((acs.alpha*acs.theta_des)/(acs.theta)))
        if acs.alpha < 1e-15
            println("Newton's method failed")
            return
        end
    end
    amp.newton_equation.b .= b
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


raw"""
    VectorbundleManoptProblem{
    TM<:AbstractManifold,TV<:AbstractManifold,O
}

Model a vector bundle problem, that consists of the domain manifold ``\mathcal M`` that is a AbstractManifold, the range vector bundle ``\mathcal E`` and the Newton equation ``Q_{F(x)}\circ F'(x) \delta x + F(x) = 0_{p(F(x))}``. The Newton equation should be implemented as a functor that computes a representation of the Newton matrix and the right hand side. It needs to have a field ``A`` to store a representation of the Newton matrix ``Q_{F(x)}\circ F'(x) `` and a field ``b`` to store a representation of the right hand side ``F(x)``.
"""
struct VectorbundleManoptProblem{
    TM<:AbstractManifold,TV<:AbstractManifold,O
} <: AbstractManoptProblem{TM}
    manifold::TM
    vectorbundle::TV
    newton_equation::O
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

return the Newton equation [`newton_equation`](@ref) stored within an [`VectorbundleManoptProblem`](@ref).
"""

function get_newton_equation(vbp::VectorbundleManoptProblem)
    return vbp.newton_equation
end


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
    vectorbundle_newton!(M, E, NE, p; kwargs...)

Perform Newton's method for finding a zero of a mapping ``F:\mathcal M \to \mathcal E`` where ``\mathcal M`` is a manifold and ``\mathcal E`` is a vector bundle.
In each iteration the Newton equation
`` Q_{F(p)} \circ F'(p) X + F(p) = 0``
is solved to compute a Newton direction ``X``. The next iterate is then computed by applying a retraction. 
``NE`` is a functor representing the Newton equation. It has at least fields ``A`` and ``b`` to store a representation of the Newton matrix ``Q_{F(p)}\circ F'(p)`` (covariant derivative of ``F``) and the right hand side ``F(p)`` at a point ``p\in\mathcal M``. The point ``p`` denotes the starting point. The algorithm can be run in-place of ``p``.

"""
vectorbundle_newton(M::AbstractManifold, E::AbstractManifold, args...; kwargs...) #replace type of E with VectorBundle once this is available in ManifoldsBase

function vectorbundle_newton(
    M::AbstractManifold, E::AbstractManifold, NE, p; kwargs...
)
    q = copy(M, p)
    return vectorbundle_newton!(M, E, NE, q; kwargs...)
end


function vectorbundle_newton!(
    M::AbstractManifold,
    E::AbstractManifold,
    NE::O,
    p::P;
    evaluation=AllocatingEvaluation(),
    sub_problem::Pr=nothing,
    sub_state::Op=nothing, 
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
    isnothing(sub_problem) && error("Please provide a sub_problem")
    isnothing(sub_state) && error("Please provide a sub_state")

    vbp = VectorbundleManoptProblem(M, E, NE)
    
    vbs = VectorbundleNewtonState(
        M,
        E,
        p,
        sub_problem,
        sub_state;
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


# function step_solver!(mp::VectorbundleManoptProblem, s::VectorbundleNewtonState, k)

#     E = get_vectorbundle(mp) # vector bundle (codomain of F)

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
    mp.newton_equation(M, E, s.p)

    # compute Newton direction
    s.X = s.sub_problem(mp, s)
    
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
    mp.newton_equation(M, E, s.p)

    # compute Newton direction
    s.sub_problem(mp, s.X, s)

    step = s.stepsize(mp, s, k)

    # retract
    retract!(M, s.p, s.p, s.X, step, s.retraction_method)
    s.p_trial = copy(M, s.p)

    return s
end
