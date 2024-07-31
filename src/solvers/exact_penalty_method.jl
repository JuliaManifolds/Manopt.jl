@doc raw"""
    ExactPenaltyMethodState{P,T} <: AbstractManoptSolverState

Describes the exact penalty method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `p`:                   a set point on a manifold as starting point
* `sub_problem`:         an [`AbstractManoptProblem`](@ref) problem for the subsolver
* `sub_state`:           an [`AbstractManoptSolverState`](@ref) for the subsolver
* `ϵ`:                   (`1e–3`) the accuracy tolerance
* `ϵ_min`:               (`1e-6`) the lower bound for the accuracy tolerance
* `u`:                   (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_min`:               (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ`:                   (`1.0`) the penalty parameter
* `θ_ρ`:                 (`0.3`) the scaling factor of the penalty parameter
* `stopping_criterion`:  ([`StopAfterIteration`](@ref)`(300) | (`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min) & `[`StopWhenChangeLess`](@ref)`(min_stepsize))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

# Constructor

    ExactPenaltyMethodState(M::AbstractManifold, p, sub_problem, sub_state; kwargs...)

construct an exact penalty options with the remaining previously mentioned fields as keywords using their provided defaults.

# See also

[`exact_penalty_method`](@ref)
"""
mutable struct ExactPenaltyMethodState{
    P,
    Pr<:Union{F,AbstractManoptProblem} where {F},
    St<:AbstractManoptSolverState,
    R<:Real,
    TStopping<:StoppingCriterion,
} <: AbstractSubProblemSolverState
    p::P
    sub_problem::Pr
    sub_state::St
    ϵ::R
    ϵ_min::R
    u::R
    u_min::R
    ρ::R
    θ_ρ::R
    θ_u::R
    θ_ϵ::R
    stop::TStopping
    function ExactPenaltyMethodState(
        ::AbstractManifold,
        p::P,
        sub_problem::Pr,
        sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState};
        ϵ::R=1e-3,
        ϵ_min::R=1e-6,
        ϵ_exponent=1 / 100,
        θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
        u::R=1e-1,
        u_min::R=1e-6,
        u_exponent=1 / 100,
        θ_u=(u_min / u)^(u_exponent),
        ρ::R=1.0,
        θ_ρ::R=0.3,
        stopping_criterion::SC=StopAfterIteration(300) | (
            StopWhenSmallerOrEqual(:ϵ, ϵ_min) | StopWhenChangeLess(1e-10)
        ),
    ) where {P,Pr<:Union{F,AbstractManoptProblem} where {F},R<:Real,SC<:StoppingCriterion}
        sub_state_storage = maybe_wrap_allocation_type(sub_state)
        epms = new{P,Pr,typeof(sub_state_storage),R,SC}()
        epms.p = p
        epms.sub_problem = sub_problem
        epms.sub_state = sub_state_storage
        epms.ϵ = ϵ
        epms.ϵ_min = ϵ_min
        epms.u = u
        epms.u_min = u_min
        epms.ρ = ρ
        epms.θ_ρ = θ_ρ
        epms.θ_u = θ_u
        epms.θ_ϵ = θ_ϵ
        epms.stop = stopping_criterion
        return epms
    end
end
get_iterate(epms::ExactPenaltyMethodState) = epms.p
function get_message(epms::ExactPenaltyMethodState)
    # for now only the sub solver might have messages
    return get_message(epms.sub_state)
end
function set_iterate!(epms::ExactPenaltyMethodState, M, p)
    epms.p = p
    return epms
end
function show(io::IO, epms::ExactPenaltyMethodState)
    i = get_count(epms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(epms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Exact Penalty Method
    $Iter
    ## Parameters
    * ϵ: $(epms.ϵ) (ϵ_min: $(epms.ϵ_min), θ_ϵ: $(epms.θ_ϵ))
    * u: $(epms.u) (ϵ_min: $(epms.u_min), θ_u: $(epms.θ_u))
    * ρ: $(epms.ρ) (θ_ρ: $(epms.θ_ρ))

    ## Stopping criterion

    $(status_summary(epms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    exact_penalty_method(M, F, gradF, p=rand(M); kwargs...)
    exact_penalty_method(M, cmo::ConstrainedManifoldObjective, p=rand(M); kwargs...)

perform the exact penalty method (EPM) [LiuBoumal:2019](@cite)
The aim of the EPM is to find a solution of the constrained optimisation task

```math
\begin{aligned}
\min_{p ∈\mathcal{M}} &f(p)\\
\text{subject to } &g_i(p)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(p)=0 \quad  \text{ for } j=1,…,n,
\end{aligned}
```

where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^n``
are twice continuously differentiable functions from `M` to ℝ.
For that a weighted ``L_1``-penalty term for the violation of the constraints is added to the objective

```math
f(x) + ρ\biggl( \sum_{i=1}^m \max\bigl\{0, g_i(x)\bigr\} + \sum_{j=1}^n \vert h_j(x)\vert\biggr),
```

where ``ρ>0`` is the penalty parameter.
Since this is non-smooth, a [`SmoothingTechnique`](@ref) with parameter `u` is applied,
see the [`ExactPenaltyCost`](@ref).

In every step ``k`` of the exact penalty method, the smoothed objective is then minimized over all
``x ∈\mathcal{M}``.
Then, the accuracy tolerance ``ϵ`` and the smoothing parameter ``u`` are updated by setting

```math
ϵ^{(k)}=\max\{ϵ_{\min}, θ_ϵ ϵ^{(k-1)}\},
```

where ``ϵ_{\min}`` is the lowest value ``ϵ`` is allowed to become and ``θ_ϵ ∈ (0,1)`` is constant scaling factor, and

```math
u^{(k)} = \max \{u_{\min}, \theta_u u^{(k-1)} \},
```

where ``u_{\min}`` is the lowest value ``u`` is allowed to become and ``θ_u ∈ (0,1)`` is constant scaling factor.

Finally, the penalty parameter ``ρ`` is updated as

```math
ρ^{(k)} = \begin{cases}
ρ^{(k-1)}/θ_ρ,  & \text{if } \displaystyle \max_{j ∈ \mathcal{E},i ∈ \mathcal{I}} \Bigl\{ \vert h_j(x^{(k)}) \vert, g_i(x^{(k)})\Bigr\} \geq u^{(k-1)} \Bigr) ,\\
ρ^{(k-1)}, & \text{else,}
\end{cases}
```

where ``θ_ρ ∈ (0,1)`` is a constant scaling factor.

# Input

* `M`      a manifold ``\mathcal M``
* `f`      a cost function ``f:\mathcal M→ℝ`` to minimize
* `grad_f` the gradient of the cost function

# Optional (if not called with the [`ConstrainedManifoldObjective`](@ref) `cmo`)

* `g`:      (`nothing`) the inequality constraints
* `h`:      (`nothing`) the equality constraints
* `grad_g`: (`nothing`) the gradient of the inequality constraints
* `grad_h`: (`nothing`) the gradient of the equality constraints

Note that one of the pairs (`g`, `grad_g`) or (`h`, `grad_h`) has to be provided.
Otherwise the problem is not constrained and you should consider using unconstrained solvers like [`quasi_Newton`](@ref).

# Optional

* `smoothing`:                 ([`LogarithmicSumOfExponentials`](@ref)) [`SmoothingTechnique`](@ref) to use
* `ϵ`:                         (`1e–3`) the accuracy tolerance
* `ϵ_exponent`:                (`1/100`) exponent of the ϵ update factor;
* `ϵ_min`:                     (`1e-6`) the lower bound for the accuracy tolerance
* `u`:                         (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_exponent`:                (`1/100`) exponent of the u update factor;
* `u_min`:                     (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ`:                         (`1.0`) the penalty parameter
* `equality_constraints`:      (`nothing`) the number ``n`` of equality constraints.
* `gradient_range`             (`nothing`, equivalent to [`NestedPowerRepresentation`](@extref) specify how gradients are represented
* `gradient_equality_range`:   (`gradient_range`) specify how the gradients of the equality constraints are represented
* `gradient_inequality_range`: (`gradient_range`) specify how the gradients of the inequality constraints are represented
* `inequality_constraints`:    (`nothing`) the number ``m`` of inequality constraints.
* `min_stepsize`:              (`1e-10`) the minimal step size
* `sub_cost`:                  ([`ExactPenaltyCost`](@ref)`(problem, ρ, u; smoothing=smoothing)`) use this exact penalty cost, especially with the same numbers `ρ,u` as in the options for the sub problem
* `sub_grad`:                  ([`ExactPenaltyGrad`](@ref)`(problem, ρ, u; smoothing=smoothing)`) use this exact penalty gradient, especially with the same numbers `ρ,u` as in the options for the sub problem
* `sub_kwargs`:                (`(;)`) keyword arguments to decorate the sub options, for example debug, that automatically respects the main solvers debug options (like sub-sampling) as well
* `sub_stopping_criterion`:    ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenGradientNormLess`](@ref)`(ϵ) | `[`StopWhenStepsizeLess`](@ref)`(1e-10)`) specify a stopping criterion for the subsolver.
* `sub_problem`:               ([`DefaultManoptProblem`](@ref)`(M, `[`ManifoldGradientObjective`](@ref)`(sub_cost, sub_grad; evaluation=evaluation)`, provide a problem for the subsolver
* `sub_state`:                 ([`QuasiNewtonState`](@ref)) using [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) with [`InverseBFGS`](@ref) and `sub_stopping_criterion` as a stopping criterion. See also `sub_kwargs`.
* `stopping_criterion`:        ([`StopAfterIteration`](@ref)`(300)` | ([`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min)` & [`StopWhenChangeLess`](@ref)`(1e-10)`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

For the `range`s of the constraints' gradient, other power manifold tangent space representations,
mainly the [`ArrayPowerRepresentation`](@extref Manifolds :jl:type:`Manifolds.ArrayPowerRepresentation`) can be used if the gradients can be computed more efficiently in that representation.

With `equality_constraints` and `inequality_constraints` you have to provide the dimension
of the ranges of `h` and `g`, respectively. If not provided, together with `M` and the start point `p0`,
a call to either of these is performed to try to infer these.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
exact_penalty_method(M::AbstractManifold, args...; kwargs...)
function exact_penalty_method(M::AbstractManifold, f, grad_f; kwargs...)
    return exact_penalty_method(M, f, grad_f, rand(M); kwargs...)
end
function exact_penalty_method(
    M::AbstractManifold,
    f::TF,
    grad_f::TGF,
    p;
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inequality_constrains::Union{Integer,Nothing}=nothing,
    equality_constrains::Union{Nothing,Integer}=nothing,
    kwargs...,
) where {TF,TGF}
    num_eq = if isnothing(equality_constrains)
        _number_of_constraints(h, grad_h; M=M, p=p)
    else
        inequality_constrains
    end
    num_ineq = if isnothing(inequality_constrains)
        _number_of_constraints(g, grad_g; M=M, p=p)
    else
        inequality_constrains
    end
    cmo = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        evaluation=evaluation,
        equality_constrains=num_eq,
        inequality_constrains=num_ineq,
        M=M,
        p=p,
    )
    return exact_penalty_method(
        M,
        cmo,
        p;
        evaluation=evaluation,
        equality_constrains=equality_constrains,
        inequality_constrains=inequality_constrains,
        kwargs...,
    )
end
function exact_penalty_method(
    M::AbstractManifold,
    f,
    grad_f,
    p::Number;
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    q = [p]
    f_(M, p) = f(M, p[])
    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    g_ = isnothing(g) ? nothing : (M, p) -> g(M, p[])
    grad_g_ = isnothing(grad_g) ? nothing : _to_mutating_gradient(grad_g, evaluation)
    h_ = isnothing(h) ? nothing : (M, p) -> h(M, p[])
    grad_h_ = isnothing(grad_h) ? nothing : _to_mutating_gradient(grad_h, evaluation)
    cmo = ConstrainedManifoldObjective(
        f_, grad_f_, g_, grad_g_, h_, grad_h_; evaluation=evaluation, M=M, p=p
    )
    rs = exact_penalty_method(M, cmo, q; evaluation=evaluation, kwargs...)
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function exact_penalty_method(
    M::AbstractManifold, cmo::O, p=rand(M); kwargs...
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return exact_penalty_method!(M, cmo, q; kwargs...)
end

@doc raw"""
    exact_penalty_method!(M, f, grad_f, p; kwargs...)
    exact_penalty_method!(M, cmo::ConstrainedManifoldObjective, p; kwargs...)

perform the exact penalty method (EPM) performed in place of `p`.

For all options, see [`exact_penalty_method`](@ref).
"""
exact_penalty_method!(M::AbstractManifold, args...; kwargs...)
function exact_penalty_method!(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inequality_constrains=nothing,
    equality_constrains=nothing,
    kwargs...,
)
    if isnothing(inequality_constrains)
        inequality_constrains = _number_of_constraints(g, grad_g; M=M, p=p)
    end
    if isnothing(equality_constrains)
        equality_constrains = _number_of_constraints(h, grad_h; M=M, p=p)
    end
    cmo = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        evaluation=evaluation,
        equality_constrains=equality_constrains,
        inequality_constrains=inequality_constrains,
        M=M,
        p=p,
    )
    return exact_penalty_method!(
        M,
        cmo,
        p;
        evaluation=evaluation,
        equality_constrains=equality_constrains,
        inequality_constrains=inequality_constrains,
        kwargs...,
    )
end
function exact_penalty_method!(
    M::AbstractManifold,
    cmo::O,
    p;
    evaluation=AllocatingEvaluation(),
    ϵ::Real=1e-3,
    ϵ_min::Real=1e-6,
    ϵ_exponent=1 / 100,
    θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
    u::Real=1e-1,
    u_min::Real=1e-6,
    u_exponent=1 / 100,
    ρ::Real=1.0,
    objective_type=:Riemannian,
    θ_ρ::Real=0.3,
    θ_u=(u_min / u)^(u_exponent),
    gradient_range=nothing,
    gradient_equality_range=gradient_range,
    gradient_inequality_range=gradient_range,
    smoothing=LogarithmicSumOfExponentials(),
    sub_cost=ExactPenaltyCost(cmo, ρ, u; smoothing=smoothing),
    sub_grad=ExactPenaltyGrad(cmo, ρ, u; smoothing=smoothing),
    sub_kwargs=(;),
    sub_problem::Pr=DefaultManoptProblem(
        M,
        decorate_objective!(
            M,
            ManifoldGradientObjective(sub_cost, sub_grad; evaluation=evaluation);
            objective_type=objective_type,
            sub_kwargs...,
        ),
    ),
    sub_stopping_criterion=StopAfterIteration(300) |
                           StopWhenGradientNormLess(ϵ) |
                           StopWhenStepsizeLess(1e-8),
    sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState}=decorate_state!(
        QuasiNewtonState(
            M,
            copy(M, p);
            initial_vector=zero_vector(M, p),
            direction_update=QuasiNewtonLimitedMemoryDirectionUpdate(
                M, copy(M, p), InverseBFGS(), 30
            ),
            stopping_criterion=sub_stopping_criterion,
            stepsize=default_stepsize(M, QuasiNewtonState),
            sub_kwargs...,
        );
        sub_kwargs...,
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
        StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(1e-10)
    ),
    kwargs...,
) where {
    O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective},
    Pr<:Union{F,AbstractManoptProblem} where {F},
}
    sub_state_storage = maybe_wrap_allocation_type(sub_state)
    emps = ExactPenaltyMethodState(
        M,
        p,
        sub_problem,
        sub_state_storage;
        ϵ=ϵ,
        ϵ_min=ϵ_min,
        u=u,
        u_min=u_min,
        ρ=ρ,
        θ_ρ=θ_ρ,
        θ_ϵ=θ_ϵ,
        θ_u=θ_u,
        stopping_criterion=stopping_criterion,
    )
    dcmo = decorate_objective!(M, cmo; objective_type=objective_type, kwargs...)
    mp = if isnothing(gradient_equality_range) && isnothing(gradient_inequality_range)
        DefaultManoptProblem(M, dcmo)
    else
        ConstrainedManoptProblem(
            M,
            dcmo;
            gradient_equality_range=gradient_equality_range,
            gradient_inequality_range=gradient_inequality_range,
        )
    end
    epms = decorate_state!(emps; kwargs...)
    solve!(mp, epms)
    return get_solver_return(get_objective(mp), epms)
end
#
# Solver functions
#
function initialize_solver!(::AbstractManoptProblem, epms::ExactPenaltyMethodState)
    return epms
end
function step_solver!(
    amp::AbstractManoptProblem, epms::ExactPenaltyMethodState{P,<:AbstractManoptProblem}, i
) where {P}
    M = get_manifold(amp)
    # use subsolver to minimize the smoothed penalized function
    set_manopt_parameter!(epms.sub_problem, :Objective, :Cost, :ρ, epms.ρ)
    set_manopt_parameter!(epms.sub_problem, :Objective, :Cost, :u, epms.u)
    set_manopt_parameter!(epms.sub_problem, :Objective, :Gradient, :ρ, epms.ρ)
    set_manopt_parameter!(epms.sub_problem, :Objective, :Gradient, :u, epms.u)
    set_iterate!(epms.sub_state, M, copy(M, epms.p))
    update_stopping_criterion!(epms, :MinIterateChange, epms.ϵ)

    epms.p = get_solver_result(solve!(epms.sub_problem, epms.sub_state))

    # get new evaluation of penalty
    cost_ineq = get_inequality_constraint(amp, epms.p, :)
    cost_eq = get_equality_constraint(amp, epms.p, :)
    max_violation = max(max(maximum(cost_ineq; init=0), 0), maximum(abs.(cost_eq); init=0))
    # update ρ if necessary
    (max_violation > epms.u) && (epms.ρ = epms.ρ / epms.θ_ρ)
    # update u and ϵ
    epms.u = max(epms.u_min, epms.u * epms.θ_u)
    epms.ϵ = max(epms.ϵ_min, epms.ϵ * epms.θ_ϵ)
    return epms
end
get_solver_result(epms::ExactPenaltyMethodState) = epms.p
