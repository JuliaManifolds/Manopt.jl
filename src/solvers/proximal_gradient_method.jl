#
#
# Solver
@doc """
    proximal_gradient_method(M, f, grad_g, prox_h, p=rand(M); kwargs...)
    proximal_gradient_method(M, mpgo::ManifoldProximalGradientObjective, p=rand(M); kwargs...)
    proximal_gradient_method!(M, f, grad_g, prox_h, p; kwargs...)
    proximal_gradient_method!(M, mpgo::ManifoldProximalGradientObjective, p; kwargs...)

Perform the proximal gradient method

Given the minimization problem

```math
$(_tex(:argmin))_{p∈$(_tex(:Cal, "M"))} f(p),
$(_tex(:quad)) $(_tex(:text, " where ")) $(_tex(:quad)) f(p) = g(p) + h(p).
```

this method performs the (intrinsic) proximal gradient method
alhgorithm.

Let ``λ_k ≥ 0`` be a sequence of (proximal) parameters, initialise
``p^{(0)} = p``,
and ``k=0``

Then perform as long as the stopping criterion is not fulfilled
```math
p^{(k+1)} = prox_{λ_kh}$(_tex(:Bigl))(
$(_tex(:retr))_{a^{(k)}}$(_tex(:bigl))(-λ_k $(_tex(:grad)) g(a^{(k)}$(_tex(:bigr)))
$(_tex(:Bigr))),
```
where ``a^{(k)}=p^{(k)}`` by default, but it allows to introduce some acceleration before
computing the gradient step.

# Input

$(_var(:Argument, :M; type=true))
$(_var(:Argument, :f))
* `grad_g`:           a gradient `(M,p) -> X` or `(M, X, p) -> X` of the smooth part ``g`` of the problem
* `prox_h`:           a proximal map `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the nonsmoooth part ``h`` of ``f``
$(_var(:Argument, :p))

# Keyword Arguments

* `acceleration=(p, s, k) -> (copyto!(get_manifold(M), s.a, s.p); s)`: a function `(problem, state, k) -> state` to compute an acceleration, that is performed before the gradient step
$(_var(:Keyword, :evaluation))
* `λ = k -> 0.25`: a function returning the sequence of proximal parameters ``λ_k``
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(100)`"))
$(_var(:Keyword, :sub_problem, "sub_problem", "Union{AbstractManoptProblem, F, Nothing}"; default="nothing", add="or nothing to take the proximal map from the [`ManifoldProximalGradientObjective`](@ref)"))
$(_var(:Keyword, :sub_state; default="evaluation", add="This field is ignored, if the `sub_problem` is `Nothing`"))
$(_var(:Keyword, :X; add=:as_Gradient))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""
function proximal_gradient_method(
    M::AbstractManifold,
    f,
    grad_g,
    prox_h,
    p=rand(M);
    evaluation=AllocatingEvaluation(),
    kwargs...,
)
    mpgo = ManifoldProximalGradientObjective(f, grad_g, prox_h; evaluation=evaluation)
    return proximal_gradient_method(M, mpgo, p; evaluation=evaluation, kwargs...)
end
function proximal_gradient_method(
    M::AbstractManifold, mpgo::O, p=rand(M); kwargs...
) where {O<:Union{ManifoldProximalGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return proximal_gradient_method!(M, mpgo, q; kwargs...)
end

function proximal_gradient_method!(
    M::AbstractManifold, f, grad_g, prox_h, p; evaluation=AllocatingEvaluation(), kwargs...
)
    mpgo = ManifoldProximalGradientObjective(f, grad_g, prox_h; evaluation=evaluation)
    return proximal_gradient_method!(M, mpgo, p; evaluation=evaluation, kwargs...)
end
function proximal_gradient_method!(
    M::AbstractManifold,
    mpgo::O,
    p;
    acceleration=function (pr, st, k)
        copyto!(get_manifold(pr), st.a, st.p)
        return st
    end,
    backtracking_plan=nothing,
    λ=i -> 1.0,
    stopping_criterion=StopWhenGradientMappingNormLess(1e-2) |
                       StopAfterIteration(5000) |
                       StopWhenChangeLess(M, 1e-9),
    X=zero_vector(M, p),
    retraction_method=default_retraction_method(M, typeof(p)),
    inverse_retraction_method=default_inverse_retraction_method(M, typeof(p)),
    sub_problem=nothing,
    sub_state=AllocatingEvaluation(),
    kwargs...,
) where {
    O<:Union{ManifoldProximalGradientObjective,AbstractDecoratedManifoldObjective},
    S<:StoppingCriterion,
}
    dmpgo = decorate_objective!(M, mpgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpgo)
    pgms = ProximalGradientMethodState(
        M;
        p=p,
        acceleration=acceleration,
        backtracking_plan=backtracking_plan,
        λ=λ,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        sub_problem=sub_problem,
        sub_state=sub_state,
        X=X,
    )
    dpgms = decorate_state!(pgms; kwargs...)
    solve!(dmp, dpgms)
    return get_solver_return(get_objective(dmp), dpgms)
end

function initialize_solver!(amp::AbstractManoptProblem, pgms::ProximalGradientMethodState)
    M = get_manifold(amp)
    copyto!(M, pgms.q, pgms.p)
    zero_vector!(M, pgms.X, pgms.p)
    return pgms
end

function step_solver!(amp::AbstractManoptProblem, pgms::ProximalGradientMethodState, k)
    M = get_manifold(amp)
    # acceleration
    pgms.acceleration(amp, pgms, k)
    # evaluate the gradient at a
    get_gradient!(amp, pgms.X, pgms.a)

    if isnothing(pgms.backtracking_plan)
        # Regular step with fixed λ
        λ_k = pgms.λ(k)
    else
        # Backtracking case
        # # Initial gradient step with s
        # retract!(M, pgms.q, pgms.a, -pgms.backtracking_plan.s * pgms.X, pgms.retraction_method)
        # Compute step size using backtracking
        λ_k = backtracking_step_size(
            M,
            amp,
            pgms.a,
            pgms.X,
            pgms.retraction_method,
            pgms.inverse_retraction_method;
            pgmb=ProximalGradientMethodBacktracking(
                M,
                pgms.q;
                s=pgms.backtracking_plan.s,
                η=pgms.backtracking_plan.η,
                γ=pgms.backtracking_plan.γ,
            ),
        )
    end

    # Gradient step with chosen λ_k
    retract!(M, pgms.q, pgms.a, -λ_k * pgms.X, pgms.retraction_method)
    # Proximal step
    _pgm_proximal_step(amp, pgms, λ_k)

    return pgms
end

# (I) Problem is nothing -> use prox from objective
function _pgm_proximal_step(
    amp::AbstractManoptProblem, pgms::ProximalGradientMethodState{P,T,Nothing}, λ::Real
) where {P,T}
    get_proximal_map!(amp, pgms.p, λ, pgms.q)
    return pgms
end

# (II) Problem is a subsolver -> solve
function _pgm_proximal_step(
    amp::AbstractManoptProblem,
    pgms::ProximalGradientMethodState{
        P,T,<:AbstractManoptProblem,<:AbstractManoptSolverState
    },
    λ::Real,
) where {P,T}
    M = get_manifold(amp)
    # set lambda
    set_parameter!(pgms.sub_problem, :λ, λ)
    # set start value to q
    set_iterate!(pgms.sub_state, M, copy(M, pgms.q))
    solve!(pgms.sub_problem, pgms.sub_state)
    copyto!(M, pgms.p, get_solver_result(pgms.sub_state))
    return pgms
end

# (III) Solver for the backtracking case
function _pgm_proximal_step_backtracking!(amp::AbstractManoptProblem, p, q, λ)
    get_proximal_map!(amp, p, λ, q)
    return p
end
