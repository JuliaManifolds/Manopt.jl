# Questions
# Stopping Criterion is not suitable it min outside C?
# where des the backtracking come from in this form?

"""
    ProjectedGradientMethodState <: AbstractManoptSolverState

# Fields

$(_var(:Field, :stepsize, "backtracking")) to determine a step size from ``p_k`` to the candidate ``q_k``
$(_var(:Field, :inverse_retraction_method))
$(_var(:Field, :p; add=[:as_Iterate]))
* `q` an interims point for the projected gradient step
$(_var(:Field, :stepsize)) ``α_k`` to determine the ``q_k`` candidate
$(_var(:Field, :stopping_criterion, "stop"))
$(_var(:Field, :retraction_method))
$(_var(:Field, :X))
* `Y::T` a temporary memory for a tangent vector to store the no. Used within the backtracking

# Constructor

    ProjectedGradientMethodState(M, p=rand(M); kwargs...)

## Keyword arguments

$(_var(:Keyword, :stepsize, "backtracking"; default="[`ArmijoLinesearchStepsize`](@ref)`(M)`")) ``p_k`` to the candidate ``q_k``
$(_var(:Keyword, :inverse_retraction_method))
$(_var(:Keyword, :stepsize; default="[`ConstantStepsize`](@ref)`(M)`")) ``α_k`` to determine the ``q_k`` candidate
$(_var(:Keyword, :stopping_criterion, "stop"; default="[`StopAfterIteration`](@ref)`(300)`"))
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :X))
"""
struct ProjectedGradientMethodState{P,T,S,S2,SC,RM,IRM} <: AbstractManoptSolverState
    backtrack::S2
    p::P
    q::P # for doing a step (y_k) and projection (z_k) inplace
    Y::T
    inverse_retraction_method::IRM
    stop::SC
    retraction_method::RM
    stepsize::S # α_k
    X::T
    old_p::P # last iterate – temporarily here to check an idea
end
function ProjectedGradientMethodState(
    M::AbstractManifold,
    p=rand(M);
    backtrack::Stepsize=ArmijoLinesearchStepsize(M),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M, typeof(p)
    ),
    stepsize::Stepsize=ConstantStepsize(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(300), # TODO: Improve?
    X=zero_vector(M, p),
)
    return ProjectedGradientMethodState(
        backtrack,
        p,
        copy(M, p),
        copy(M, p, X),
        inverse_retraction_method,
        stopping_criterion,
        retraction_method,
        stepsize,
        X,
        copy(M, p),
    )
end
get_iterate(pgms::ProjectedGradientMethodState) = pgms.p
get_gradient(pgms::ProjectedGradientMethodState) = pgms.X
# TODO: Write a show method.

#
#
# New Stopping Criterion
"""
    StopWhenProjectedGradienStationary <: StoppingCriterion

Stop when the step taken by the projection is  (before linesearch)
exactly the opposite of the

"""
mutable struct StopWhenProjectedGradienStationary{F,TSSA<:StoreStateAction} <:
               StoppingCriterion
    threshold::F
    last_change::F
    storage::TSSA
    at_iteration::Int
end
function StopWhenProjectedGradienStationary(
    M::AbstractManifold,
    ε::F;
    storage::StoreStateAction=StoreStateAction(M; store_points=Tuple{:Iterate}),
) where {F<:Real}
    return StopWhenProjectedGradienStationary{F,typeof(storage)}(ε, zero(ε), storage, -1)
end
function (c::StopWhenProjectedGradienStationary)(
    mp::AbstractManoptProblem, pgms::ProjectedGradientMethodState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
        c.last_change = Inf
    end
    if has_storage(c.storage, PointStorageKey(:Iterate))
        M = get_manifold(mp)
        p_old = get_storage(c.storage, PointStorageKey(:Iterate))
        c.last_change = distance(M, p_old, pgms.q)
        if c.last_change < c.threshold && k > 0
            c.at_iteration = k
            c.storage(mp, pgms, k)
            return true
        end
    end
    c.storage(mp, pgms, k)
    return false
end
function get_reason(c::StopWhenProjectedGradienStationary)
    if (c.at_iteration >= 0)
        return "At iteration $(c.at_iteration) algorithm has reached a stationary point, since the distance from the last iterate to the projected gradient ($(c.last_change)) less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenProjectedGradienStationary)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "stopped after $(c.threshold):\t$s"
end

#
#
# The solver
_doc_pgm = """
    projected_gradient_method(M, f, grad_f, proj, p=rand(M); kwargs...)
    projected_gradient_method(M, obj::ConstrainedSetObjective, p=rand(M); kwargs...)
    projected_gradient_method!(M, f, grad_f, proj, p; kwargs...)
    projected_gradient_method!(M, obj::ConstrainedSetObjective, p; kwargs...)

Compute the projected gradient method for the constrained problem

$(_problem(:SetConstrained))

by performing the following steps

1. Using the `stepsize` ``α_k`` compute a candidate ``q_k = $(_tex(:proj))_{$(_tex(:Cal, "C"))}$(_tex(:Bigl))($(_tex(:retr))_{p_k}$(_tex(:bigl))(-α_k $(_tex(:grad)) f(p_k)$(_tex(:bigr)))$(_tex(:Bigr)))``
2. Compute a backtracking stepsize ``β_k ≤ 1`` along ``Y_k = $(_tex(:retr))_{p_k}^{-1}q_k``
3. Compute the new iterate ``p_{k+1} = $(_tex(:retr))_{p_k}( β_k $(_tex(:retr))_{p_k}^{-1}q_k )``

until the `stopping_criterion=` is fulfilled

# Input

$(_var(:Argument, :M; type=true))
$(_var(:Argument, :f))
$(_var(:Argument, :grad_f))
* `proj` the function that projects onto the set ``$(_tex(:Cal, "C"))``
  as a function `(M, p) -> q` or a function `(M, q, p) -> q` computing the projection in-place of `q`.
$(_var(:Argument, :p))


# Keyword arguments

$(_var(:Keyword, :stepsize, "backtrack"; default="A")) to perform the backtracking
$(_var(:Keyword, :evaluation))
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stepsize; default="[`ConstantStepsize`](@ref)`(injectivity_radius(M)/2)`")) to perform the candidate projected step.
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1.0e-6)`)"))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_pgm)"
function projected_gradient_method(M, f, grad_f, proj; kwargs...)
    return projected_gradient_method(M, f, grad_f, proj, rand(M); kwargs...)
end
function projected_gradient_method(
    M, f, grad_f, proj, p; indicator=nothing, evaluation=AllocatingEvaluation(), kwargs...
)
    cs_obj = ConstrainedSetObjective(
        f, grad_f, proj; evaluation=evaluation, indicator=indicator
    )
    return projected_gradient_method(M, cs_obj, p; kwargs...)
end
function projected_gradient_method(M, obj::ConstrainedSetObjective, p; kwargs...)
    q = copy(M, p)
    return projected_gradient_method!(M, obj, q; kwargs...)
end

@doc "$(_doc_pgm)"
function projected_gradient_method!(
    M, f, grad_f, proj, p; indicator=nothing, evaluation=AllocatingEvaluation(), kwargs...
)
    cs_obj = ConstrainedSetObjective(
        f, grad_f, proj; evaluation=evaluation, indicator=indicator
    )
    return projected_gradient_method!(M, cs_obj, p; kwargs...)
end
function projected_gradient_method!(
    M,
    obj::ConstrainedSetObjective,
    p;
    backtrack::Stepsize=ArmijoLinesearchStepsize(M; stop_increasing_at_step=0),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M, typeof(p)
    ),
    stepsize::Stepsize=ConstantStepsize(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) |
                                          StopWhenProjectedGradienStationary(M, 1e-7),
    X=zero_vector(M, p),
    kwargs...,
)
    dobj = decorate_objective!(M, obj; kwargs...)
    dmp = DefaultManoptProblem(M, dobj)
    pgms = ProjectedGradientMethodState(
        M,
        p;
        backtrack=backtrack,
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
        stepsize=stepsize,
        stopping_criterion=stopping_criterion,
        X=X,
    )
    dpgms = decorate_state!(pgms; kwargs...)
    solve!(dmp, dpgms)
    return get_solver_return(get_objective(dmp), dpgms)
end

function initialize_solver!(amp::AbstractManoptProblem, pgms::ProjectedGradientMethodState)
    get_gradient!(amp, pgms.X, pgms.p)
    return pgms
end

function step_solver!(amp::AbstractManoptProblem, pgms::ProjectedGradientMethodState, k)
    M = get_manifold(amp)
    # Step 1: gradient step
    get_gradient!(amp, pgms.X, pgms.p)
    copyto!(M, pgms.old_p, pgms.p)
    # Gradient step in q
    retract!(
        M, pgms.q, pgms.p, -get_stepsize(amp, pgms, k) * pgms.X, pgms.retraction_method
    )
    get_projected_point!(amp, pgms.q, pgms.q)
    # Determine search direction
    inverse_retract!(M, pgms.Y, pgms.p, pgms.q, pgms.inverse_retraction_method)
    τ = pgms.backtrack(amp, pgms, k, pgms.Y)
    # println("τ:", τ)
    # Compute new iterate
    retract!(M, pgms.p, pgms.p, τ * pgms.Y, pgms.retraction_method)
    # now we have
    return pgms
end
