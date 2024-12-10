@doc raw"""
    ProximalGradientMethodState <: AbstractManoptSolverState

stores options for the [`proximal_gradient_method`](@ref) solver

# Fields

* `p`:                         the current iterate
* `q`:                         an inner point storages
* `X`                          the interims gradient evaluation
* `stopping_criterion`:        a [`StoppingCriterion`](@ref)
* `λ`:                         a function for the values of ``λ_i`` per iteration ``ì``
* `retraction_method`:         a retraction to use
* `inverse_retraction_method`: an inverse retraction to use

# Constructor

    ProximalGradientMethodState(M, p=rand(M); kwargs...)

Generate the state for a given manifold `M` with initial iterate `p`.

# Keyword arguments

All fields from above are keyword arguments with the following defaults.

* `λ = i -> 0.5`
* `stopping_criterion = `[`StopAfterIteration`](@ref)`(100)` a stopping criterion
* `X = zero_vector(M, p)`
* `retraction_method = `[`default_retraction_method`](@ref)`(M, typeof(p))`
* `inverse_retraction_method = `[`default_inverse_retraction_method`](@ref)`(M, typeof(p))`
"""
mutable struct ProximalGradientMethodState{
    P,
    T,
    S<:StoppingCriterion,
    F,
    RM<:AbstractRetractionMethod,
    IRM<:AbstractInverseRetractionMethod,
} <: AbstractManoptSolverState
    λ::F
    p::P
    q::P
    stop::S
    X::T
    retraction_method::RM
    inverse_retraction_method::IRM
end
function ProximalGradientMethodState(
    M::AbstractManifold,
    p::P;
    stopping_criterion::S=StopAfterIteration(100),
    λ::F=i -> 0.25,
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    inverse_retraction_method::IRM=default_inverse_retraction_method(M, typeof(p)),
) where {P,T,S,F,RM<:AbstractRetractionMethod,IRM<:AbstractInverseRetractionMethod}
    return ProximalGradientMethodState{P,T,S,F,RM,IRM}(
        λ,
        p,
        copy(M, p),
        stopping_criterion,
        X,
        retraction_method,
        inverse_retraction_method,
    )
end
get_iterate(pgms::ProximalGradientMethodState) = pgms.p
function set_iterate!(pgms::ProximalGradientMethodState, p)
    pgms.p = p
    return p
end
function show(io::IO, pgms::ProximalGradientMethodState)
    i = get_count(pgms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pgms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Proximal Gradient Method
    $Iter

    ## Parameters

    * λ:                              $(pgms.λ)

    * retraction_method:              $(pgms.retraction_method)

    ## Stopping criterion

    $(status_summary(pgms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

#
#
# Solver
@doc raw"""
    proximal_gradient_method(M, f, grad_g, prox_h, p=rand(M); kwargs...)
    proximal_gradient_method(M, mpgo::ManifoldProximalGradientObjective, p=rand(M); kwargs...)
    proximal_gradient_method!(M, f, grad_g, prox_h, p; kwargs...)
    proximal_gradient_method!(M, mpgo::ManifoldProximalGradientObjective, p; kwargs...)

Perform (an intrinsic idea of the) proximal gradient method

Given the minimization problem

```math
\operatorname*{arg\,min}_{p∈\mathcal M} f(p),
\qquad \text{where}\quad f(p) = g(p) + h(p)
```

this method performs the (until now more like intuitive) intrinsic proximal gradient method
alhgorithm.
Let ``λ_k \geq 0`` be a sequence of (proximal) parameters, initialise
``p^{(0)} = p``,
and ``k=0``
Then perform as long as the stopping criterion is not fulfilled
```math
p^{(k+1)} = prox_{λ_kh}\Bigl(
\operatorname{retr}_{p^{(k)}}\bigl(-λ_k \operatorname{grad} g(p^{(k)}\bigr)
\Bigr)
```

# Input

* `M`:                a manifold ``\mathcal M``
* `f`:                a cost function ``f:\mathcal M→ℝ`` to minimize
* `grad_g`:           a gradient `(M,λ,p) -> X` or `(M, X, λ, p) -> X` of the smooth part ``g`` of the problem
* `prox_h`:           a proximal map `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the nonsmoooth part ``h`` of ``f``
* `p`:                an initial value ``p ∈ \mathcal M``

# Keyword Arguments

* `evaluation = `[`AllocatingEvaluation`](@ref)) specify whether the proximal maps work by allocation (default) form `prox(M, λ, x)`
  or [`InplaceEvaluation`](@ref) in place of form `prox!(M, y, λ, x)`.
* `λ = `i -> 0.25` ) a function returning the sequence of proximal parameters ``λ_k``
* `stopping_criterion = `[`StopWhenGradientMappingNormLess(1e-2)`](@ref)` | `[`StopAfterIteration`](@ref)`(5000) | `[`StopWhenChangeLess`](@ref)`(M, 1e.9)` a stopping criterion
* `X = zero_vector(M, p)`
* `retraction_method = `[`default_retraction_method`](@ref)`(M, typeof(p))` the retraction ``\operatorname{retr}``
* `inverse_retraction_method = `[`default_inverse_retraction_method`](@ref)`(M, typeof(p))` the inverse retraction ``\operatorname{retr}^{-1}``

All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.
If you provide the [`ManifoldProximalGradientObjective`](@ref) directly, these decorations can still be specified
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
    λ=i -> 1.0,
    stopping_criterion=StopWhenGradientMappingNormLess(1e-2) |
                       StopAfterIteration(5000) |
                       StopWhenChangeLess(M, 1e-9),
    X=zero_vector(M, p),
    retraction_method=default_retraction_method(M, typeof(p)),
    inverse_retraction_method=default_inverse_retraction_method(M, typeof(p)),
    kwargs...,
) where {O<:Union{ManifoldProximalGradientObjective,AbstractDecoratedManifoldObjective}}
    dmpgo = decorate_objective!(M, mpgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpgo)
    pgms = ProximalGradientMethodState(
        M,
        p;
        stopping_criterion=stopping_criterion,
        λ=λ,
        X=X,
        retraction_method=retraction_method,
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
function step_solver!(amp::AbstractManoptProblem, pgms::ProximalGradientMethodState, i)
    M = get_manifold(amp)
    get_gradient!(amp, pgms.X, pgms.p)
    copyto!(M, pgms.q, pgms.p)
    # (a) + (b) gradient and prox steps
    get_proximal_map!(
        amp,
        pgms.p,
        pgms.λ(i),
        retract!(M, pgms.p, pgms.p, -pgms.λ(i) * pgms.X, pgms.retraction_method),
    )
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
