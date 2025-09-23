#
# State
#

_sc_alm_default = "[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))([`StopWhenSmallerOrEqual`](@ref)`(:ϵ, ϵ_min)`$(_sc(:All))[`StopWhenChangeLess`](@ref)`(1e-10) )`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`"
@doc """
    AugmentedLagrangianMethodState{P,T} <: AbstractManoptSolverState

Describes the augmented Lagrangian method, with

# Fields

a default value is given in brackets if a parameter can be left out in initialization.

* `ϵ`:     the accuracy tolerance
* `ϵ_min`: the lower bound for the accuracy tolerance
* `λ`:     the Lagrange multiplier with respect to the equality constraints
* `λ_max`: an upper bound for the Lagrange multiplier belonging to the equality constraints
* `λ_min`: a lower bound for the Lagrange multiplier belonging to the equality constraints
$(_var(:Field, :p; add = [:as_Iterate]))
* `penalty`: evaluation of the current penalty term, initialized to `Inf`.
* `μ`:     the Lagrange multiplier with respect to the inequality constraints
* `μ_max`: an upper bound for the Lagrange multiplier belonging to the inequality constraints
* `ρ`:     the penalty parameter
$(_var(:Field, :sub_problem))
$(_var(:Field, :sub_state))
* `τ`:     factor for the improvement of the evaluation of the penalty parameter
* `θ_ρ`:   the scaling factor of the penalty parameter
* `θ_ϵ`:   the scaling factor of the accuracy tolerance
$(_var(:Field, :stopping_criterion, "stop"))

# Constructor

    AugmentedLagrangianMethodState(M::AbstractManifold, co::ConstrainedManifoldObjective,
        sub_problem, sub_state; kwargs...
    )

construct an augmented Lagrangian method options, where the manifold `M` and the [`ConstrainedManifoldObjective`](@ref) `co` are used for
manifold- or objective specific defaults.

    AugmentedLagrangianMethodState(M::AbstractManifold, co::ConstrainedManifoldObjective,
        sub_problem; evaluation=AllocatingEvaluation(), kwargs...
    )

construct an augmented Lagrangian method options, where the manifold `M` and the [`ConstrainedManifoldObjective`](@ref) `co` are used for
manifold- or objective specific defaults, and `sub_problem` is a closed form solution with `evaluation` as type of evaluation.

## Keyword arguments

the following keyword arguments are available to initialise the corresponding fields

* `ϵ=1e–3`
* `ϵ_min=1e-6`
* `λ=ones(n)`: `n` is the number of equality constraints in the [`ConstrainedManifoldObjective`](@ref) `co`.
* `λ_max=20.0`
* `λ_min=- λ_max`
* `μ=ones(m)`: `m` is the number of inequality constraints in the [`ConstrainedManifoldObjective`](@ref) `co`.
* `μ_max=20.0`
$(_var(:Keyword, :p; add = :as_Initial))
* `ρ=1.0`
* `τ=0.8`
* `θ_ρ=0.3`
* `θ_ϵ=(ϵ_min/ϵ)^(ϵ_exponent)`
* stopping_criterion=$_sc_alm_default.

# See also

[`augmented_Lagrangian_method`](@ref)
"""
mutable struct AugmentedLagrangianMethodState{
        P,
        Pr <: Union{F, AbstractManoptProblem} where {F},
        St <: AbstractManoptSolverState,
        R <: Real,
        V <: AbstractVector{<:R},
        TStopping <: StoppingCriterion,
    } <: AbstractSubProblemSolverState
    p::P
    sub_problem::Pr
    sub_state::St
    ϵ::R
    ϵ_min::R
    λ_max::R
    λ_min::R
    μ_max::R
    μ::V
    λ::V
    ρ::R
    τ::R
    θ_ρ::R
    θ_ϵ::R
    penalty::R
    stop::TStopping
    last_stepsize::R
    function AugmentedLagrangianMethodState(
            M::AbstractManifold,
            co::ConstrainedManifoldObjective,
            sub_problem::Pr,
            sub_state::St;
            p::P = rand(M),
            ϵ::R = 1.0e-3,
            ϵ_min::R = 1.0e-6,
            λ::V = ones(length(get_equality_constraint(M, co, p, :))),
            λ_max::R = 20.0,
            λ_min::R = (-λ_max),
            μ::V = ones(length(get_inequality_constraint(M, co, p, :))),
            μ_max::R = 20.0,
            ρ::R = 1.0,
            τ::R = 0.8,
            θ_ρ::R = 0.3,
            ϵ_exponent = 1 / 100,
            θ_ϵ = (ϵ_min / ϵ)^(ϵ_exponent),
            stopping_criterion::SC = StopAfterIteration(300) |
                (
                StopWhenSmallerOrEqual(:ϵ, ϵ_min) &
                    StopWhenChangeLess(M, 1.0e-10)
            ) |
                StopWhenChangeLess(M, 1.0e-10),
            kwargs...,
        ) where {
            P,
            Pr <: Union{F, AbstractManoptProblem} where {F},
            St <: AbstractManoptSolverState,
            R <: Real,
            V,
            SC <: StoppingCriterion,
        }
        alms = new{P, Pr, St, R, V, SC}()
        alms.p = p
        alms.sub_problem = sub_problem
        alms.sub_state = sub_state
        alms.ϵ = ϵ
        alms.ϵ_min = ϵ_min
        alms.λ_max = λ_max
        alms.λ_min = λ_min
        alms.μ_max = μ_max
        alms.μ = μ
        alms.λ = λ
        alms.ρ = ρ
        alms.τ = τ
        alms.θ_ρ = θ_ρ
        alms.θ_ϵ = θ_ϵ
        alms.penalty = Inf
        alms.stop = stopping_criterion
        alms.last_stepsize = Inf
        return alms
    end
end
function AugmentedLagrangianMethodState(
        M::AbstractManifold,
        co::ConstrainedManifoldObjective,
        sub_problem;
        evaluation::E = AllocatingEvaluation(),
        kwargs...,
    ) where {E <: AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation = evaluation)
    return AugmentedLagrangianMethodState(M, co, sub_problem, cfs; kwargs...)
end

get_iterate(alms::AugmentedLagrangianMethodState) = alms.p
function set_iterate!(alms::AugmentedLagrangianMethodState, M, p)
    alms.p = p
    return alms
end
function get_message(alms::AugmentedLagrangianMethodState)
    # for now only the sub solver might have messages
    return get_message(alms.sub_state)
end

function show(io::IO, alms::AugmentedLagrangianMethodState)
    i = get_count(alms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(alms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Augmented Lagrangian Method
    $Iter
    ## Parameters
    * ϵ: $(alms.ϵ) (ϵ_min: $(alms.ϵ_min), θ_ϵ: $(alms.θ_ϵ))
    * λ: $(alms.λ) (λ_min: $(alms.λ_min), λ_max: $(alms.λ_max))
    * μ: $(alms.μ) (μ_max: $(alms.μ_max))
    * ρ: $(alms.ρ) (θ_ρ: $(alms.θ_ρ))
    * τ: $(alms.τ)
    * current penalty: $(alms.penalty)

    ## Stopping criterion

    $(status_summary(alms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

_doc_alm_λ_update = raw"""
```math
λ_j^{(k+1)} =\operatorname{clip}_{[λ_{\min},λ_{\max}]} (λ_j^{(k)} + ρ^{(k)} h_j(p^{(k+1)})) \text{for all} j=1,…,p,
```
"""
_doc_alm_μ_update = raw"""
```math
μ_i^{(k+1)} =\operatorname{clip}_{[0,μ_{\max}]} (μ_i^{(k)} + ρ^{(k)} g_i(p^{(k+1)})) \text{ for all } i=1,…,m,
```
"""
_doc_alm_ε_update = raw"""
```math
ϵ^{(k)}=\max\{ϵ_{\min}, θ_ϵ ϵ^{(k-1)}\},
```
"""

_doc_alm_σ = raw"""
```math
σ^{(k)}=\max_{j=1,…,p, i=1,…,m} \{\|h_j(p^{(k)})\|, \|\max_{i=1,…,m}\{g_i(p^{(k)}), -\frac{μ_i^{(k-1)}}{ρ^{(k-1)}} \}\| \}.
```
"""

_doc_alm_ρ_update = raw"""
```math
ρ^{(k)} = \begin{cases}
ρ^{(k-1)}/θ_ρ,  & \text{if } σ^{(k)}\leq θ_ρ σ^{(k-1)} ,\\
ρ^{(k-1)}, & \text{else,}
\end{cases}
```
"""

_doc_alm = """
    augmented_Lagrangian_method(M, f, grad_f, p=rand(M); kwargs...)
    augmented_Lagrangian_method(M, cmo::ConstrainedManifoldObjective, p=rand(M); kwargs...)
    augmented_Lagrangian_method!(M, f, grad_f, p; kwargs...)
    augmented_Lagrangian_method!(M, cmo::ConstrainedManifoldObjective, p; kwargs...)

perform the augmented Lagrangian method (ALM) [LiuBoumal:2019](@cite).
This method can work in-place of `p`.

The aim of the ALM is to find the solution of the constrained optimisation task

$(_problem(:Constrained))

where `M` is a Riemannian manifold, and ``f``, ``$(_math(:Sequence, "g", "i", "1", "n"))`` and ``$(_math(:Sequence, "h", "j", "1", "m"))``
are twice continuously differentiable functions from `M` to ℝ.
In every step ``k`` of the algorithm, the [`AugmentedLagrangianCost`](@ref)
 ``$(_doc_AL_Cost("k"))`` is minimized on $(_tex(:Cal, "M")),
  where ``μ^{(k)} ∈ ℝ^n`` and ``λ^{(k)} ∈ ℝ^m`` are the current iterates of the Lagrange multipliers and ``ρ^{(k)}`` is the current penalty parameter.

The Lagrange multipliers are then updated by

$_doc_alm_λ_update

and

$_doc_alm_μ_update

   where ``λ_{$(_tex(:text, "min"))} ≤ λ_{$(_tex(:text, "max"))}`` and ``μ_{$(_tex(:text, "max"))}`` are the multiplier boundaries.

Next, the accuracy tolerance ``ϵ`` is updated as

$_doc_alm_ε_update

 where ``ϵ_{$(_tex(:text, "min"))}`` is the lowest value ``ϵ`` is allowed to become and ``θ_ϵ ∈ (0,1)`` is constant scaling factor.

Last, the penalty parameter ``ρ`` is updated as follows: with

$_doc_alm_σ

`ρ` is updated as

$_doc_alm_ρ_update

where ``θ_ρ ∈ (0,1)`` is a constant scaling factor.

# Input

$(_var(:Argument, :M; type = true))
$(_var(:Argument, :f))
$(_var(:Argument, :grad_f))

# Optional (if not called with the [`ConstrainedManifoldObjective`](@ref) `cmo`)

* `g=nothing`: the inequality constraints
* `h=nothing`: the equality constraints
* `grad_g=nothing`: the gradient of the inequality constraints
* `grad_h=nothing`: the gradient of the equality constraints

Note that one of the pairs (`g`, `grad_g`) or (`h`, `grad_h`) has to be provided.
Otherwise the problem is not constrained and a better solver would be for example [`quasi_Newton`](@ref).

# Keyword Arguments

$(_var(:Keyword, :evaluation))
* `ϵ=1e-3`:           the accuracy tolerance
* `ϵ_min=1e-6`:       the lower bound for the accuracy tolerance
* `ϵ_exponent=1/100`: exponent of the ϵ update factor;
  also 1/number of iterations until maximal accuracy is needed to end algorithm naturally

  * `equality_constraints=nothing`: the number ``n`` of equality constraints.
  If not provided, a call to the gradient of `g` is performed to estimate these.

* `gradient_range=nothing`: specify how both gradients of the constraints are represented

* `gradient_equality_range=gradient_range`:
   specify how gradients of the equality constraints are represented, see [`VectorGradientFunction`](@ref).

* `gradient_inequality_range=gradient_range`:
   specify how gradients of the inequality constraints are represented, see [`VectorGradientFunction`](@ref).

* `inequality_constraints=nothing`: the number ``m`` of inequality constraints.
   If not provided, a call to the gradient of `g` is performed to estimate these.

* `λ=ones(size(h(M,x),1))`: the Lagrange multiplier with respect to the equality constraints
* `λ_max=20.0`:       an upper bound for the Lagrange multiplier belonging to the equality constraints
* `λ_min=- λ_max`:    a lower bound for the Lagrange multiplier belonging to the equality constraints

* `μ=ones(size(h(M,x),1))`: the Lagrange multiplier with respect to the inequality constraints
* `μ_max=20.0`: an upper bound for the Lagrange multiplier belonging to the inequality constraints

* `ρ=1.0`:            the penalty parameter
* `τ=0.8`:            factor for the improvement of the evaluation of the penalty parameter
* `θ_ρ=0.3`:          the scaling factor of the penalty parameter
* `θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent)`: the scaling factor of the exactness

* `sub_cost=[`AugmentedLagrangianCost± (@ref)`(cmo, ρ, μ, λ):` use augmented Lagrangian cost, based on the [`ConstrainedManifoldObjective`](@ref) build from the functions provided.
   $(_note(:KeywordUsedIn, "sub_problem"))

* `sub_grad=[`AugmentedLagrangianGrad`](@ref)`(cmo, ρ, μ, λ)`: use augmented Lagrangian gradient, based on the [`ConstrainedManifoldObjective`](@ref) build from the functions provided.
  $(_note(:KeywordUsedIn, "sub_problem"))

$(_var(:Keyword, :sub_kwargs))

$(_var(:Keyword, :stopping_criterion; default = _sc_alm_default))
$(_var(:Keyword, :sub_problem; default = "[`DefaultManoptProblem`](@ref)`(M, sub_objective)`"))
$(_var(:Keyword, :sub_state; default = "[`QuasiNewtonState`](@ref)", add = "as the quasi newton method, the [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) with [`InverseBFGS`](@ref) is used."))
* `sub_stopping_criterion::StoppingCriterion=`[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(ϵ)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-8)`,


For the `range`s of the constraints' gradient, other power manifold tangent space representations,
mainly the [`ArrayPowerRepresentation`](@extref Manifolds :jl:type:`Manifolds.ArrayPowerRepresentation`) can be used if the gradients can be computed more efficiently in that representation.

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_alm)"
function augmented_Lagrangian_method(
        M::AbstractManifold,
        f,
        grad_f,
        p = rand(M);
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        g = nothing,
        h = nothing,
        grad_g = nothing,
        grad_h = nothing,
        inequality_constraints::Union{Integer, Nothing} = nothing,
        equality_constraints::Union{Nothing, Integer} = nothing,
        kwargs...,
    )
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
        inequality_constraints = inequality_constraints,
        equality_constraints = equality_constraints,
        M = M,
        p = p,
    )
    rs = augmented_Lagrangian_method(M, cmo, p_; evaluation = evaluation, kwargs...)
    return _ensure_matching_output(p, rs)
end
function augmented_Lagrangian_method(
        M::AbstractManifold, cmo::O, p = rand(M); kwargs...
    ) where {O <: Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    keywords_accepted(augmented_Lagrangian_method; kwargs...)
    return augmented_Lagrangian_method!(M, cmo, q; kwargs...)
end
calls_with_kwargs(::typeof(augmented_Lagrangian_method)) = (augmented_Lagrangian_method!,)

@doc "$(_doc_alm)"
function augmented_Lagrangian_method!(
        M::AbstractManifold,
        f::TF,
        grad_f::TGF,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        g = nothing,
        h = nothing,
        grad_g = nothing,
        grad_h = nothing,
        inequality_constraints = nothing,
        equality_constraints = nothing,
        kwargs...,
    ) where {TF, TGF}
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
    dcmo = decorate_objective!(M, cmo; kwargs...)
    return augmented_Lagrangian_method!(
        M,
        dcmo,
        p;
        evaluation = evaluation,
        equality_constraints = equality_constraints,
        inequality_constraints = inequality_constraints,
        kwargs...,
    )
end
function augmented_Lagrangian_method!(
        M::AbstractManifold,
        cmo::O,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        ϵ::Real = 1.0e-3,
        ϵ_min::Real = 1.0e-6,
        ϵ_exponent::Real = 1 / 100,
        θ_ϵ::Real = (ϵ_min / ϵ)^(ϵ_exponent),
        μ::Vector = ones(length(get_inequality_constraint(M, cmo, p, :))),
        μ_max::Real = 20.0,
        λ::Vector = ones(length(get_equality_constraint(M, cmo, p, :))),
        λ_max::Real = 20.0,
        λ_min::Real = (-λ_max),
        τ::Real = 0.8,
        ρ::Real = 1.0,
        θ_ρ::Real = 0.3,
        gradient_range = nothing,
        gradient_equality_range = gradient_range,
        gradient_inequality_range = gradient_range,
        objective_type = :Riemannian,
        sub_cost = AugmentedLagrangianCost(cmo, ρ, μ, λ),
        sub_grad = AugmentedLagrangianGrad(cmo, ρ, μ, λ),
        sub_kwargs = (;),
        sub_stopping_criterion::StoppingCriterion = StopAfterIteration(300) |
            StopWhenGradientNormLess(ϵ) |
            StopWhenStepsizeLess(1.0e-8),
        sub_state::AbstractManoptSolverState = decorate_state!(
            QuasiNewtonState(
                M;
                p = copy(M, p),
                X = zero_vector(M, p),
                direction_update = QuasiNewtonLimitedMemoryDirectionUpdate(
                    M, copy(M, p), InverseBFGS(), min(manifold_dimension(M), 30)
                ),
                stopping_criterion = sub_stopping_criterion,
                stepsize = default_stepsize(M, QuasiNewtonState),
                sub_kwargs...,
            );
            sub_kwargs...,
        ),
        sub_problem::AbstractManoptProblem = DefaultManoptProblem(
            M,
            # pass down objective type to sub solvers
            decorate_objective!(
                M,
                ManifoldGradientObjective(sub_cost, sub_grad; evaluation = evaluation);
                objective_type = objective_type,
                sub_kwargs...,
            ),
        ),
        stopping_criterion::StoppingCriterion = StopAfterIteration(300) |
            (
            StopWhenSmallerOrEqual(:ϵ, ϵ_min) &
                StopWhenChangeLess(M, 1.0e-10)
        ) |
            StopWhenStepsizeLess(1.0e-10),
        kwargs...,
    ) where {O <: Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(augmented_Lagrangian_method!; kwargs...)
    sub_state_storage = maybe_wrap_evaluation_type(sub_state)
    alms = AugmentedLagrangianMethodState(
        M,
        cmo,
        sub_problem,
        sub_state_storage;
        p = p,
        ϵ = ϵ,
        ϵ_min = ϵ_min,
        λ_max = λ_max,
        λ_min = λ_min,
        μ_max = μ_max,
        μ = μ,
        λ = λ,
        ρ = ρ,
        τ = τ,
        θ_ρ = θ_ρ,
        θ_ϵ = θ_ϵ,
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
    alms = decorate_state!(alms; kwargs...)
    solve!(mp, alms)
    return get_solver_return(get_objective(mp), alms)
end
calls_with_kwargs(::typeof(augmented_Lagrangian_method!)) = (decorate_objective!, decorate_state!)
#
# Solver functions
#
function initialize_solver!(::AbstractManoptProblem, alms::AugmentedLagrangianMethodState)
    alms.penalty = Inf
    return alms
end
function step_solver!(mp::AbstractManoptProblem, alms::AugmentedLagrangianMethodState, iter)
    M = get_manifold(mp)
    # use subsolver to minimize the augmented Lagrangian
    set_parameter!(alms.sub_problem, :Objective, :Cost, :ρ, alms.ρ)
    set_parameter!(alms.sub_problem, :Objective, :Cost, :μ, alms.μ)
    set_parameter!(alms.sub_problem, :Objective, :Cost, :λ, alms.λ)
    set_parameter!(alms.sub_problem, :Objective, :Gradient, :ρ, alms.ρ)
    set_parameter!(alms.sub_problem, :Objective, :Gradient, :μ, alms.μ)
    set_parameter!(alms.sub_problem, :Objective, :Gradient, :λ, alms.λ)
    set_iterate!(alms.sub_state, M, copy(M, alms.p))

    set_parameter!(alms, :StoppingCriterion, :MinIterateChange, alms.ϵ)

    new_p = get_solver_result(solve!(alms.sub_problem, alms.sub_state))
    alms.last_stepsize = distance(M, alms.p, new_p, default_inverse_retraction_method(M))
    copyto!(M, alms.p, new_p)

    # update multipliers
    cost_ineq = get_inequality_constraint(mp, alms.p, :)
    n_ineq_constraint = length(cost_ineq)
    alms.μ .=
        min.(
        ones(n_ineq_constraint) .* alms.μ_max,
        max.(alms.μ .+ alms.ρ .* cost_ineq, zeros(n_ineq_constraint)),
    )
    cost_eq = get_equality_constraint(mp, alms.p, :)
    n_eq_constraint = length(cost_eq)
    alms.λ =
        min.(
        ones(n_eq_constraint) .* alms.λ_max,
        max.(ones(n_eq_constraint) .* alms.λ_min, alms.λ + alms.ρ .* cost_eq),
    )
    # get new evaluation of penalty
    penalty = maximum(
        [abs.(max.(-alms.μ ./ alms.ρ, cost_ineq))..., abs.(cost_eq)...]; init = 0
    )
    # update ρ if necessary
    (penalty > alms.τ * alms.penalty) && (alms.ρ = alms.ρ / alms.θ_ρ)
    alms.penalty = penalty

    # update the tolerance ϵ
    alms.ϵ = max(alms.ϵ_min, alms.ϵ * alms.θ_ϵ)
    return alms
end
get_solver_result(alms::AugmentedLagrangianMethodState) = alms.p

function get_last_stepsize(::AbstractManoptProblem, s::AugmentedLagrangianMethodState, k)
    return s.last_stepsize
end
