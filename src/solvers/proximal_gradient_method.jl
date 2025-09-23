# Solver
_doc_prox_grad_method = """
    proximal_gradient_method(M, f, g, grad_g, p=rand(M); prox_nonsmooth=nothing, kwargs...)
    proximal_gradient_method(M, mpgo::ManifoldProximalGradientObjective, p=rand(M); kwargs...)
    proximal_gradient_method!(M, f, g, grad_g, p; prox_nonsmooth=nothing, kwargs...)
    proximal_gradient_method!(M, mpgo::ManifoldProximalGradientObjective, p; kwargs...)

Perform the proximal gradient method as introduced in [BergmannJasaJohnPfeffer:2025:1](@cite) and [BergmannJasaJohnPfeffer:2025:2](@cite).
See also [FengHuangSongYingZeng:2021](@cite) for a similar approach.

Given the minimization problem

```math
$(_tex(:argmin))_{p∈$(_tex(:Cal, "M"))} f(p),
$(_tex(:quad)) $(_tex(:text, " where ")) $(_tex(:quad)) f(p) = g(p) + h(p).
```

This method performs the (intrinsic) proximal gradient method algorithm.

Let ``λ_k ≥ 0`` be a sequence of (proximal) parameters, initialize
``p^{(0)} = p``, and ``k=0``.

Then perform as long as the stopping criterion is not fulfilled
```math
p^{(k+1)} = prox_{λ_kh}$(_tex(:Bigl))(
$(_tex(:retr))_{a^{(k)}}$(_tex(:bigl))(-λ_k $(_tex(:grad)) g(a^{(k)}$(_tex(:bigr)))
$(_tex(:Bigr))),
```
where ``a^{(k)}=p^{(k)}`` by default, but it allows to introduce some acceleration before
computing the gradient step.

# Input

$(_var(:Argument, :M; type = true))
$(_var(:Argument, :f; add = "total cost function `f = g + h`"))
* `g`:              the smooth part of the cost function
* `grad_g`:           a gradient `(M,p) -> X` or `(M, X, p) -> X` of the smooth part ``g`` of the problem

# Keyword Arguments

* `acceleration=(p, s, k) -> (copyto!(get_manifold(M), s.a, s.p); s)`: a function `(problem, state, k) -> state` to compute an acceleration, that is performed before the gradient step - the default is to copy the current point to the acceleration point, i.e. no acceleration is performed
$(_var(:Keyword, :evaluation))
* `prox_nonsmooth`:          a proximal map `(M,λ,p) -> q` or `(M, q, λ, p) -> q` for the (possibly) nonsmoooth part ``h`` of ``f``
$(_var(:Argument, :p))
$(_var(:Keyword, :stepsize; default = "[`default_stepsize`](@ref)`(M, ProximalGradientMethodState)`")) that by default uses a [`ProximalGradientMethodBacktracking`](@ref).
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(100)`"))
$(_var(:Keyword, :sub_problem, "sub_problem", "Union{AbstractManoptProblem, F, Nothing}"; default = "nothing", add = "or nothing to take the proximal map from the [`ManifoldProximalGradientObjective`](@ref)"))
$(_var(:Keyword, :sub_state; default = "evaluation", add = "This field is ignored, if the `sub_problem` is `Nothing`"))
$(_var(:Keyword, :X; add = :as_Gradient))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_prox_grad_method)"
function proximal_gradient_method(
        M::AbstractManifold,
        f,
        g,
        grad_g,
        p = rand(M);
        prox_nonsmooth = nothing,
        evaluation = AllocatingEvaluation(),
        kwargs...,
    )
    mpgo = ManifoldProximalGradientObjective(
        f, g, grad_g, prox_nonsmooth; evaluation = evaluation
    )
    return proximal_gradient_method(M, mpgo, p; evaluation = evaluation, kwargs...)
end

function proximal_gradient_method(
        M::AbstractManifold, mpgo::O, p = rand(M); kwargs...
    ) where {O <: Union{ManifoldProximalGradientObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(proximal_gradient_method; kwargs...)
    q = copy(M, p)
    return proximal_gradient_method!(M, mpgo, q; kwargs...)
end
calls_with_kwargs(::typeof(proximal_gradient_method)) = (proximal_gradient_method!,)

@doc "$(_doc_prox_grad_method)"
function proximal_gradient_method!(
        M::AbstractManifold,
        f,
        g,
        grad_g,
        p;
        prox_nonsmooth = nothing,
        evaluation = AllocatingEvaluation(),
        kwargs...,
    )
    mpgo = ManifoldProximalGradientObjective(
        f, g, grad_g, prox_nonsmooth; evaluation = evaluation
    )
    return proximal_gradient_method!(M, mpgo, p; evaluation = evaluation, kwargs...)
end
function proximal_gradient_method!(
        M::AbstractManifold,
        mpgo::O,
        p;
        acceleration = function (pr, st, k)
            copyto!(get_manifold(pr), st.a, st.p)
            return st
        end,
        debug = [DebugWarnIfStepsizeCollapsed()],
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, ProximalGradientMethodState
        ),
        cost_nonsmooth::Union{Nothing, Function} = nothing,
        subgradient_nonsmooth::Union{Nothing, Function} = nothing,
        stopping_criterion::S = StopWhenGradientMappingNormLess(1.0e-7) |
            StopAfterIteration(5000) |
            StopWhenChangeLess(M, 1.0e-9),
        X = zero_vector(M, p),
        retraction_method = default_retraction_method(M, typeof(p)),
        inverse_retraction_method = default_inverse_retraction_method(M, typeof(p)),
        sub_problem = if isnothing(mpgo.proximal_map_h!!)
            DefaultManoptProblem(
                M,
                ManifoldSubgradientObjective(
                    ProximalGradientNonsmoothCost(cost_nonsmooth, 0.1, p),
                    ProximalGradientNonsmoothSubgradient(subgradient_nonsmooth, 0.1, p),
                ),
            )
        else
            nothing
        end,
        sub_state = if !isnothing(mpgo.proximal_map_h!!)
            # AllocatingEvaluation()
            nothing
        else
            SubGradientMethodState(
                M;
                p = p,
                stepsize = Manopt.DecreasingStepsize(
                    M; exponent = 1, factor = 1, subtrahend = 0, length = 1, shift = 0, type = :absolute
                ),
                stopping_criterion = StopAfterIteration(2500) | StopWhenSubgradientNormLess(1.0e-8),
            )
        end,
        kwargs...,
    ) where {
        O <: Union{ManifoldProximalGradientObjective, AbstractDecoratedManifoldObjective},
        S <: StoppingCriterion,
    }
    keywords_accepted(proximal_gradient_method!; kwargs...)
    # Check whether either the right defaults were provided or a `sub_problem`.
    if isnothing(mpgo.proximal_map_h!!) && isnothing(cost_nonsmooth)
        error(
            """
            The `sub_problem` is not correctly initialized. Provide _one of_ the following setups
            * `prox_nonsmooth` keyword argument as a closed form solution,
            * `cost_nonsmooth` keyword argument for the (possibly nonsmooth) part of the cost function whose proximal map is to be computed,
            """,
        )
    end
    dmpgo = decorate_objective!(M, mpgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpgo)
    pgms = ProximalGradientMethodState(
        M;
        p = p,
        acceleration = acceleration,
        stepsize = _produce_type(stepsize, M),
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
        stopping_criterion = stopping_criterion,
        sub_problem = sub_problem,
        sub_state = sub_state,
        X = X,
    )
    dpgms = decorate_state!(pgms; kwargs...)
    solve!(dmp, dpgms)
    return get_solver_return(get_objective(dmp), dpgms)
end
calls_with_kwargs(::typeof(proximal_gradient_method!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(amp::AbstractManoptProblem, pgms::ProximalGradientMethodState)
    M = get_manifold(amp)
    zero_vector!(M, pgms.X, pgms.p)
    copyto!(M, pgms.a, pgms.p)
    return pgms
end

function step_solver!(amp::AbstractManoptProblem, pgms::ProximalGradientMethodState, k)
    M = get_manifold(amp)
    # Store previous iterate
    copyto!(M, pgms.q, pgms.p)

    # (Possible) Acceleration
    pgms.acceleration(amp, pgms, k)

    # Evaluate the gradient at (possibly) accelerated point
    get_gradient!(amp, pgms.X, pgms.a)

    # Compute stepsize using the provided stepsize object
    pgms.last_stepsize = get_stepsize(amp, pgms, k)

    # Gradient step with chosen stepsize
    retract!(M, pgms.a, pgms.a, -pgms.last_stepsize * pgms.X, pgms.retraction_method)

    # Proximal step with chosen stepsize
    _pgm_proximal_step(amp, pgms, pgms.last_stepsize)

    return pgms
end

# (I) Problem is nothing -> use prox from objective
function _pgm_proximal_step(
        amp::AbstractManoptProblem, pgms::ProximalGradientMethodState{P, T, Nothing}, λ::Real
    ) where {P, T}
    get_proximal_map!(amp, pgms.p, λ, pgms.a)
    return pgms
end

# (II) Problem is a subsolver -> solve
function _pgm_proximal_step(
        amp::AbstractManoptProblem,
        pgms::ProximalGradientMethodState{
            P, T, <:AbstractManoptProblem, <:AbstractManoptSolverState,
        },
        λ::Real,
    ) where {P, T}
    M = get_manifold(amp)
    # set lambda
    set_parameter!(pgms.sub_problem, :Objective, :Cost, :λ, λ)
    set_parameter!(pgms.sub_problem, :Objective, :SubGradient, :λ, λ)
    # set the proximity point of the subproblem
    set_parameter!(pgms.sub_problem, :Objective, :Cost, :proximity_point, pgms.a)
    set_parameter!(pgms.sub_problem, :Objective, :SubGradient, :proximity_point, pgms.a)
    # set start value to a
    set_iterate!(pgms.sub_state, M, copy(M, pgms.a))
    solve!(pgms.sub_problem, pgms.sub_state)
    copyto!(M, pgms.p, get_solver_result(pgms.sub_state))
    return pgms
end
