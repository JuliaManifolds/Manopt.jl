@doc """
    ExactPenaltyMethodState{P,T} <: AbstractManoptSolverState

Describes the exact penalty method, with

# Fields

* `ϵ`: the accuracy tolerance
* `ϵ_min`: the lower bound for the accuracy tolerance
$(_var(:Field, :p; add = [:as_Iterate]))
* `ρ`: the penalty parameter
$(_var(:Field, :sub_problem))
$(_var(:Field, :sub_state))
$(_var(:Field, :stopping_criterion, "stop"))
* `u`: the smoothing parameter and threshold for violation of the constraints
* `u_min`: the lower bound for the smoothing parameter and threshold for violation of the constraints
* `θ_ϵ`: the scaling factor of the tolerance parameter
* `θ_ρ`: the scaling factor of the penalty parameter
* `θ_u`: the scaling factor of the smoothing parameter

# Constructor

    ExactPenaltyMethodState(M::AbstractManifold, sub_problem, sub_state; kwargs...)

construct the exact penalty state.

    ExactPenaltyMethodState(M::AbstractManifold, sub_problem;
        evaluation=AllocatingEvaluation(), kwargs...
)

construct the exact penalty state, where `sub_problem` is a closed form solution with `evaluation` as type of evaluation.

# Keyword arguments

* `ϵ=1e-3`
* `ϵ_min=1e-6`
* `ϵ_exponent=1 / 100`: a shortcut for the scaling factor ``θ_ϵ``
* `θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent)`
* `u=1e-1`
* `u_min=1e-6`
* `u_exponent=1 / 100`:  a shortcut for the scaling factor ``θ_u``.
* `θ_u=(u_min / u)^(u_exponent)`
$(_var(:Keyword, :p; add = :as_Initial))
* `ρ=1.0`
* `θ_ρ=0.3`
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))` (`"))
  [`StopWhenSmallerOrEqual`](@ref)`(:ϵ, ϵ_min)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-10) )`

# See also

[`exact_penalty_method`](@ref)
"""
mutable struct ExactPenaltyMethodState{
        P,
        Pr <: Union{F, AbstractManoptProblem} where {F},
        St <: AbstractManoptSolverState,
        R <: Real,
        TStopping <: StoppingCriterion,
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
            M::AbstractManifold,
            sub_problem::Pr,
            sub_state::St;
            p::P = rand(M),
            ϵ::R = 1.0e-3,
            ϵ_min::R = 1.0e-6,
            ϵ_exponent = 1 / 100,
            θ_ϵ = (ϵ_min / ϵ)^(ϵ_exponent),
            u::R = 1.0e-1,
            u_min::R = 1.0e-6,
            u_exponent = 1 / 100,
            θ_u = (u_min / u)^(u_exponent),
            ρ::R = 1.0,
            θ_ρ::R = 0.3,
            stopping_criterion::SC = StopAfterIteration(300) | (
                StopWhenSmallerOrEqual(:ϵ, ϵ_min) | StopWhenChangeLess(M, 1.0e-10)
            ),
        ) where {
            P,
            Pr <: Union{F, AbstractManoptProblem} where {F},
            St <: AbstractManoptSolverState,
            R <: Real,
            SC <: StoppingCriterion,
        }
        sub_state_storage = maybe_wrap_evaluation_type(sub_state)
        epms = new{P, Pr, typeof(sub_state_storage), R, SC}()
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
function ExactPenaltyMethodState(
        M::AbstractManifold, sub_problem; evaluation::E = AllocatingEvaluation(), kwargs...
    ) where {E <: AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation = evaluation)
    return ExactPenaltyMethodState(M, sub_problem, cfs; kwargs...)
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

_doc_EPM_penalty = raw"""
```math
f(x) + ρ\biggl( \sum_{i=1}^m \max\bigl\{0, g_i(x)\bigr\} + \sum_{j=1}^n \vert h_j(x)\vert\biggr),
```
where ``ρ>0`` is the penalty parameter.
"""

_doc_EMP_ϵ_update = raw"""
```math
ϵ^{(k)}=\max\{ϵ_{\min}, θ_ϵ ϵ^{(k-1)}\},
```

where ``ϵ_{\min}`` is the lowest value ``ϵ`` is allowed to become and ``θ_ϵ ∈ (0,1)`` is constant scaling factor, and
"""

_doc_EMP_ρ_update = raw"""
```math
ρ^{(k)} = \begin{cases}
ρ^{(k-1)}/θ_ρ,  & \text{if } \displaystyle \max_{j ∈ \mathcal{E},i ∈ \mathcal{I}} \Bigl\{ \vert h_j(x^{(k)}) \vert, g_i(x^{(k)})\Bigr\} \geq u^{(k-1)} \Bigr) ,\\
ρ^{(k-1)}, & \text{ else,}
\end{cases}
```

where ``θ_ρ ∈ (0,1)`` is a constant scaling factor.
"""
_doc_EMP_u_update = raw"""
```math
u^{(k)} = \max \{u_{\min}, \theta_u u^{(k-1)} \},
```

where ``u_{\min}`` is the lowest value ``u`` is allowed to become and ``θ_u ∈ (0,1)`` is constant scaling factor.
"""

_doc_EPM = """
    exact_penalty_method(M, f, grad_f, p=rand(M); kwargs...)
    exact_penalty_method(M, cmo::ConstrainedManifoldObjective, p=rand(M); kwargs...)
    exact_penalty_method!(M, f, grad_f, p; kwargs...)
    exact_penalty_method!(M, cmo::ConstrainedManifoldObjective, p; kwargs...)

perform the exact penalty method (EPM) [LiuBoumal:2019](@cite)
The aim of the EPM is to find a solution of the constrained optimisation task

$(_problem(:Constrained))

where `M` is a Riemannian manifold, and ``f``, ``$(_math(:Sequence, "g", "i", "1", "n"))`` and ``$(_math(:Sequence, "h", "j", "1", "m"))``
are twice continuously differentiable functions from `M` to ℝ.
For that a weighted ``L_1``-penalty term for the violation of the constraints is added to the objective

$(_doc_EPM_penalty)

Since this is non-smooth, a [`SmoothingTechnique`](@ref) with parameter `u` is applied,
see the [`ExactPenaltyCost`](@ref).

In every step ``k`` of the exact penalty method, the smoothed objective is then minimized over all ``p ∈$(_math(:M))``.
Then, the accuracy tolerance ``ϵ`` and the smoothing parameter ``u`` are updated by setting

$(_doc_EMP_ϵ_update)

$(_doc_EMP_u_update)

Finally, the penalty parameter ``ρ`` is updated as

$(_doc_EMP_ρ_update)

# Input

$(_var(:Argument, :M; type = true))
$(_var(:Argument, :f))
$(_var(:Argument, :grad_f))
$(_var(:Argument, :p))

# Keyword arguments
 if not called with the [`ConstrainedManifoldObjective`](@ref) `cmo`

* `g=nothing`: the inequality constraints
* `h=nothing`: the equality constraints
* `grad_g=nothing`: the gradient of the inequality constraints
* `grad_h=nothing`: the gradient of the equality constraints

Note that one of the pairs (`g`, `grad_g`) or (`h`, `grad_h`) has to be provided.
Otherwise the problem is not constrained and a better solver would be for example [`quasi_Newton`](@ref).

# Further keyword arguments

* `ϵ=1e–3`: the accuracy tolerance
* `ϵ_exponent=1/100`: exponent of the ϵ update factor;
* `ϵ_min=1e-6`: the lower bound for the accuracy tolerance
* `u=1e–1`: the smoothing parameter and threshold for violation of the constraints
* `u_exponent=1/100`: exponent of the u update factor;
* `u_min=1e-6`: the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ=1.0`: the penalty parameter
* `equality_constraints=nothing`: the number ``n`` of equality constraints.
  If not provided, a call to the gradient of `g` is performed to estimate these.
* `gradient_range=nothing`: specify how both gradients of the constraints are represented
* `gradient_equality_range=gradient_range`:
   specify how gradients of the equality constraints are represented, see [`VectorGradientFunction`](@ref).
* `gradient_inequality_range=gradient_range`:
   specify how gradients of the inequality constraints are represented, see [`VectorGradientFunction`](@ref).
* `inequality_constraints=nothing`: the number ``m`` of inequality constraints.
   If not provided, a call to the gradient of `g` is performed to estimate these.
* `min_stepsize=1e-10`: the minimal step size
* `smoothing=`[`LogarithmicSumOfExponentials`](@ref): a [`SmoothingTechnique`](@ref) to use
* `sub_cost=`[`ExactPenaltyCost`](@ref)`(problem, ρ, u; smoothing=smoothing)`: cost to use in the sub solver
  $(_note(:KeywordUsedIn, "sub_problem"))
* `sub_grad=`[`ExactPenaltyGrad`](@ref)`(problem, ρ, u; smoothing=smoothing)`: gradient to use in the sub solver
  $(_note(:KeywordUsedIn, "sub_problem"))
$(_var(:Keyword, :sub_kwargs))
* `sub_stopping_criterion=`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(ϵ)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-10)`: a stopping cirterion for the sub solver
  $(_note(:KeywordUsedIn, "sub_state"))
$(_var(:Keyword, :sub_state; default = "[`DefaultManoptProblem`](@ref)`(M, `[`ManifoldGradientObjective`](@ref)`(sub_cost, sub_grad; evaluation=evaluation)"))
$(_var(:Keyword, :sub_state; default = "[`QuasiNewtonState`](@ref)", add = " where [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) with [`InverseBFGS`](@ref) is used"))
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))` ( `[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min)`$(_sc(:All))[`StopWhenChangeLess`](@ref)`(1e-10) )`"))

For the `range`s of the constraints' gradient, other power manifold tangent space representations,
mainly the [`ArrayPowerRepresentation`](@extref Manifolds :jl:type:`Manifolds.ArrayPowerRepresentation`) can be used if the gradients can be computed more efficiently in that representation.

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_EPM)"
exact_penalty_method(M::AbstractManifold, args...; kwargs...)
function exact_penalty_method(M::AbstractManifold, f, grad_f; kwargs...)
    return exact_penalty_method(M, f, grad_f, rand(M); kwargs...)
end
function exact_penalty_method(
        M::AbstractManifold,
        f::TF,
        grad_f::TGF,
        p;
        g = nothing,
        h = nothing,
        grad_g = nothing,
        grad_h = nothing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        inequality_constraints::Union{Integer, Nothing} = nothing,
        equality_constraints::Union{Nothing, Integer} = nothing,
        kwargs...,
    ) where {TF, TGF}
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    g_ = _ensure_mutating_cost(g, p)
    grad_g_ = _ensure_mutating_gradient(grad_g, p, evaluation)
    h_ = _ensure_mutating_cost(h, p)
    grad_h_ = _ensure_mutating_gradient(grad_h, p, evaluation)
    cmo = ConstrainedManifoldObjective(
        f_,
        grad_f_,
        g_,
        grad_g_,
        h_,
        grad_h_;
        evaluation = evaluation,
        equality_constraints = equality_constraints,
        inequality_constraints = equality_constraints,
        M = M,
        p = p_,
    )
    rs = exact_penalty_method(
        M,
        cmo,
        p_;
        evaluation = evaluation,
        equality_constraints = equality_constraints,
        inequality_constraints = inequality_constraints,
        kwargs...,
    )
    return _ensure_matching_output(p, rs)
end
function exact_penalty_method(
        M::AbstractManifold, cmo::O, p = rand(M); kwargs...
    ) where {O <: Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(exact_penalty_method; kwargs...)
    q = copy(M, p)
    return exact_penalty_method!(M, cmo, q; kwargs...)
end
calls_with_kwargs(::typeof(exact_penalty_method)) = (exact_penalty_method!,)

@doc "$(_doc_EPM)"
exact_penalty_method!(M::AbstractManifold, args...; kwargs...)
function exact_penalty_method!(
        M::AbstractManifold,
        f,
        grad_f,
        p;
        g = nothing,
        h = nothing,
        grad_g = nothing,
        grad_h = nothing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        inequality_constraints = nothing,
        equality_constraints = nothing,
        kwargs...,
    )
    if isnothing(inequality_constraints)
        inequality_constraints = _number_of_constraints(g, grad_g; M = M, p = p)
    end
    if isnothing(equality_constraints)
        equality_constraints = _number_of_constraints(h, grad_h; M = M, p = p)
    end
    cmo = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        evaluation = evaluation,
        equality_constraints = equality_constraints,
        inequality_constraints = inequality_constraints,
        M = M,
        p = p,
    )
    return exact_penalty_method!(
        M,
        cmo,
        p;
        evaluation = evaluation,
        equality_constraints = equality_constraints,
        inequality_constraints = inequality_constraints,
        kwargs...,
    )
end
function exact_penalty_method!(
        M::AbstractManifold,
        cmo::O,
        p;
        evaluation = AllocatingEvaluation(),
        ϵ::Real = 1.0e-3,
        ϵ_min::Real = 1.0e-6,
        ϵ_exponent = 1 / 100,
        θ_ϵ = (ϵ_min / ϵ)^(ϵ_exponent),
        u::Real = 1.0e-1,
        u_min::Real = 1.0e-6,
        u_exponent = 1 / 100,
        ρ::Real = 1.0,
        objective_type = :Riemannian,
        θ_ρ::Real = 0.3,
        θ_u = (u_min / u)^(u_exponent),
        gradient_range = nothing,
        gradient_equality_range = gradient_range,
        gradient_inequality_range = gradient_range,
        smoothing = LogarithmicSumOfExponentials(),
        sub_cost = ExactPenaltyCost(cmo, ρ, u; smoothing = smoothing),
        sub_grad = ExactPenaltyGrad(cmo, ρ, u; smoothing = smoothing),
        sub_kwargs = (;),
        sub_problem::Pr = DefaultManoptProblem(
            M,
            decorate_objective!(
                M,
                ManifoldGradientObjective(sub_cost, sub_grad; evaluation = evaluation);
                objective_type = objective_type,
                sub_kwargs...,
            ),
        ),
        sub_stopping_criterion = StopAfterIteration(300) |
            StopWhenGradientNormLess(ϵ) |
            StopWhenStepsizeLess(1.0e-8),
        sub_state::Union{AbstractEvaluationType, AbstractManoptSolverState} = decorate_state!(
            QuasiNewtonState(
                M;
                p = copy(M, p),
                initial_vector = zero_vector(M, p),
                direction_update = QuasiNewtonLimitedMemoryDirectionUpdate(
                    M, copy(M, p), InverseBFGS(), 30
                ),
                stopping_criterion = sub_stopping_criterion,
                stepsize = default_stepsize(M, QuasiNewtonState),
                sub_kwargs...,
            );
            sub_kwargs...,
        ),
        stopping_criterion::StoppingCriterion = StopAfterIteration(300) | (
            StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(M, 1.0e-10)
        ),
        kwargs...,
    ) where {
        O <: Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective},
        Pr <: Union{F, AbstractManoptProblem} where {F},
    }
    keywords_accepted(exact_penalty_method!; kwargs...)
    sub_state_storage = maybe_wrap_evaluation_type(sub_state)
    emps = ExactPenaltyMethodState(
        M,
        sub_problem,
        sub_state_storage;
        p = p,
        ϵ = ϵ,
        ϵ_min = ϵ_min,
        u = u,
        u_min = u_min,
        ρ = ρ,
        θ_ρ = θ_ρ,
        θ_ϵ = θ_ϵ,
        θ_u = θ_u,
        stopping_criterion = stopping_criterion,
    )
    dcmo = decorate_objective!(M, cmo; objective_type = objective_type, kwargs...)
    mp = if isnothing(gradient_equality_range) && isnothing(gradient_inequality_range)
        DefaultManoptProblem(M, dcmo)
    else
        ConstrainedManoptProblem(
            M,
            dcmo;
            gradient_equality_range = gradient_equality_range,
            gradient_inequality_range = gradient_inequality_range,
        )
    end
    epms = decorate_state!(emps; kwargs...)
    solve!(mp, epms)
    return get_solver_return(get_objective(mp), epms)
end
calls_with_kwargs(::typeof(exact_penalty_method!)) = (decorate_objective!, decorate_state!)

#
# Solver functions
#
function initialize_solver!(::AbstractManoptProblem, epms::ExactPenaltyMethodState)
    return epms
end
function step_solver!(
        amp::AbstractManoptProblem, epms::ExactPenaltyMethodState{P, <:AbstractManoptProblem}, i
    ) where {P}
    M = get_manifold(amp)
    # use subsolver to minimize the smoothed penalized function
    set_parameter!(epms.sub_problem, :Objective, :Cost, :ρ, epms.ρ)
    set_parameter!(epms.sub_problem, :Objective, :Cost, :u, epms.u)
    set_parameter!(epms.sub_problem, :Objective, :Gradient, :ρ, epms.ρ)
    set_parameter!(epms.sub_problem, :Objective, :Gradient, :u, epms.u)
    set_iterate!(epms.sub_state, M, copy(M, epms.p))
    set_parameter!(epms, :StoppingCriterion, :MinIterateChange, epms.ϵ)

    new_p = get_solver_result(solve!(epms.sub_problem, epms.sub_state))
    copyto!(M, epms.p, new_p)

    # get new evaluation of penalty
    cost_ineq = get_inequality_constraint(amp, epms.p, :)
    cost_eq = get_equality_constraint(amp, epms.p, :)
    max_violation = max(max(maximum(cost_ineq; init = 0), 0), maximum(abs.(cost_eq); init = 0))
    # update ρ if necessary
    (max_violation > epms.u) && (epms.ρ = epms.ρ / epms.θ_ρ)
    # update u and ϵ
    epms.u = max(epms.u_min, epms.u * epms.θ_u)
    epms.ϵ = max(epms.ϵ_min, epms.ϵ * epms.θ_ϵ)
    return epms
end
get_solver_result(epms::ExactPenaltyMethodState) = epms.p
