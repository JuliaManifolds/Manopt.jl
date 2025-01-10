"""
    ProjectedGradientMethodState <: AbstractManoptSolverState

# Fields

$(_var(:Field, :stepsize, "backtracking")) to determine a step size from ``p_k`` to the candidate ``q_k``
$(_var(:Field, :p; add=[:as_Iterate]))
$(_var(:Field, :stepsize)) ``α_k`` to determine the ``q_k`` candidate
$(_var(:Field, :stopping_criterion, "stop"))
$(_var(:Field, :retraction_method))
$(_var(:Field, :X))
* `η::T` a temporary memory for a tangent vector. Used within the backtracking
"""
struct ProjectedGradientMethodState{P,T,S,S2,A,SC,RM,IRM} <: AbstractManoptSolverState
    backtrack::S2
    p::P
    q::P # for doing a step (y_k) and projection (z_k) inplace
    η::T
    inverse_retraction_method::IRM
    stop::SC
    retraction_method::RM
    stepsize::S # α_k
    X::T
end
# Ansatz: Default stepsize is armijo along the log_p proj

get_iterate(pgms::ProjectedGradientMethodState) = pgms.p
get_gradient(pgms::ProjectedGradientMethodState) = pgms.X

_doc_pgm = """
    projected_gradient_method(M, f, grad_f, proj, p=rand(M); kwargs...)
    projected_gradient_method!(M, f, grad_f, proj, p; kwargs...)

Compute the projected gradient method for the constrained problem

$(_problem(:SetConstrained))

by performing the following steps

1. Using the `stepsize` ``α_k`` compute a candidate ``q_k = $(_tex(:proj))_{$(_tex(:Cal, "C"))}$(_tex(:Bigl))($(_tex(:retr))_{p_k}$(_tex(:bigl))(-α_k $(_tex(:grad)) f(p_k)$(_tex(:bigr)))$(_tex(:Bigr)))``
2. Compute a backtracking stepsize ``β_k ≤ 1`` along ``η_k = $(_tex(:retr))_{p_k}^{-1}q_k``
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
"""

# TODO: Implement high level interface variants
@doc "$(_doc_pgm)"
function projected_gradient_method(M, f, grad_f, proj, p=rand(M); kwargs...) end

@doc "$(_doc_pgm)"
function projected_gradient_method!(M, f, grad_f, proj, p; kwargs...) end

function initialize_solver!(amp::AbstractManoptProblem, pgms::ProjectedGradientMethodState)
    get_gradient!(amp, pgms.X, pgms.p)
    return pgms
end

function step_solver!(amp::AbstractManoptProblem, pgms::ProjectedGradientMethodState, k)
    M = get_manifold(amp)
    # Step 1 candidate & project
    get_gradient!(amp, pgms.X, pgms.p)
    retract!(M, pgms.q, get_stepsize(amp, pgms, k) * pgms.X, pgms.retraction_method)
    project!(amp, pgms.q) # TODO: Implement in the ConstrainedSet plan
    # Determine search direction
    inverse_retract!(M, pgms.η, pgms.p, pgms.q, pgms.inverse_retraction_method)
    # Maybe currently a bit too fixed on Armijo
    # In the manuscript; β is the contraction factor, ρ is the sufficient decrease, θ?
    # Now this should also work for NM, WolfePowell, WPBinary, Constant (just not AWN I think)
    τ = pgms.backtrack(amp, pgms, k, pgms.η)
    # Compute new iterate
    retract!(M, pgms.p, pgms.p, τ * pgms.η, pgms.retraction_method)
    return pgms
end
