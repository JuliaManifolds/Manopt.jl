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
# (I) Problem is nothing -> use prox from
function step_solver!(amp::AbstractManoptProblem, pgms::ProximalGradientMethodState, k)
    M = get_manifold(amp)
    # acceleration?
    pgms.acceleration(amp, pgms, k)
    # evaluate the gradient at a
    get_gradient!(amp, pgms.X, pgms.a)
    # (a) gradient step
    retract!(M, pgms.q, pgms.a, -pgms.λ(k) * pgms.X, pgms.retraction_method)
    # (b) proximal step
    _pgm_proximal_step(amp, pgms, k)
    return pgms
end
# (I) Problem is nothing -> use prox from
function _pgm_proximal_step(
    amp::AbstractManoptProblem, pgms::ProximalGradientMethodState{P,T,Nothing}, k
) where {P,T}
    get_proximal_map!(amp, pgms.p, pgms.λ(k), pgms.q)
    return pgms
end
# (II) Problem is a subsolver -> solve
function _pgm_proximal_step(
    amp::AbstractManoptProblem,
    pgms::ProximalGradientMethodState{
        P,T,<:AbstractManoptProblem,<:AbstractManoptSolverState
    },
    k,
) where {P,T}
    M = get_manifold(amp)
    # set lambda
    set_parameter!(pgms.sub_problem, :λ, dcps.λ(k))
    # aet start value to a
    set_iterate!(pgms.sub_state, M, copy(M, dcps.q))
    solve!(pgms.sub_problem, dcps.sub_state)
    copyto!(M, pgms.p, get_solver_result(dcps.sub_state))
    get_proximal_map!(amp, pgms.p, pgms.λ(k), pgms.q)
    return pgms
end

"""
    StopWhenGradientMappingNormLess <: StoppingCriterion

A stopping criterion based on the current gradient norm.

# Fields

* `threshold`: the threshold to indicate to stop when the distance is below this value

# Internal fields

* `last_change` store the last change
* `at_iteration` store the iteration at which the stop indication happened

# Constructor

    StopWhenGradientMappingNormLess(ε)

Create a stopping criterion with threshold `ε` for the gradient mapping for the [`proximal_gradient_method`](@ref).
That is, this criterion indicates to stop when [`get_gradient`](@ref) returns a gradient vector of norm less than `ε`,
where the norm to use can be specified in the `norm=` keyword.
"""
mutable struct StopWhenGradientMappingNormLess{TF} <: StoppingCriterion
    threshold::TF
    last_change::TF
    at_iteration::Int
    function StopWhenGradientMappingNormLess(ε::TF) where {TF}
        return new{TF}(ε, zero(ε), -1)
    end
end

function (sc::StopWhenGradientMappingNormLess)(
    mp::AbstractManoptProblem, s::ProximalGradientMethodState, i::Int
)
    M = get_manifold(mp)
    if i == 0 # reset on init
        sc.at_iteration = -1
    end
    if (i > 0)
        sc.last_change =
            1 / s.λ(i) * norm(
                M, s.q, inverse_retract(M, s.q, get_iterate(s), s.inverse_retraction_method)
            )
        if sc.last_change < sc.threshold
            sc.at_iteration = i
            return true
        end
    end
    return false
end
function get_reason(c::StopWhenGradientMappingNormLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the gradient mapping norm ($(c.last_change)) is less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenGradientMappingNormLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|G| < $(c.threshold): $s"
end
indicates_convergence(c::StopWhenGradientMappingNormLess) = true
function show(io::IO, c::StopWhenGradientMappingNormLess)
    return print(
        io, "StopWhenGradientMappingNormLess($(c.threshold))\n    $(status_summary(c))"
    )
end

"""
    update_stopping_criterion!(c::StopWhenGradientMappingNormLess, :MinGradNorm, v::Float64)

Update the minimal gradient norm when an algorithm shall stop
"""
function update_stopping_criterion!(
    c::StopWhenGradientMappingNormLess, ::Val{:MinGradNorm}, v::Float64
)
    c.threshold = v
    return c
end
