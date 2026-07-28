"""
    StepsizeState{P,T} <: AbstractManoptSolverState

A state to store a point and a descent direction used within a linesearch,
if these are different from the iterate and search direction of the main solver.

# Fields

* `p::P`: a point on a manifold
* `X::T`: a tangent vector at `p`.

# Constructor

    StepsizeState(p,X)
    StepsizeState(M::AbstractManifold; p=rand(M), x=zero_vector(M,p)

# See also

[`interior_point_Newton`](@ref)
"""
struct StepsizeState{P, T} <: AbstractManoptSolverState
    p::P
    X::T
    StepsizeState(; p::P, X::T) where {P, T} = new{P, T}(p, X)
end
StepsizeState(M::AbstractManifold; p = rand(M), X = zero_vector(M, p)) = StepsizeState(; p = p, X = X)
get_iterate(s::StepsizeState) = s.p
get_gradient(s::StepsizeState) = s.X
set_iterate!(s::StepsizeState, M, p) = copyto!(M, s.p, p)
set_gradient!(s::StepsizeState, M, p, X) = copyto!(M, s.X, p, X)
Base.show(io::IO, sss::StepsizeState) = print(io, "StepsizeState(; p = ", sss.p, ", X = ", sss.X, ")")
function status_summary(sss::StepsizeState{P, T}; context::Symbol = :default) where {P, T}
    (context === :short) && return repr(sss)
    return "A state for a stepsize problem."
end

function interior_point_initial_guess(
        mp::AbstractManoptProblem, ips::StepsizeState, ::Int, l::R, η; kwargs...
    ) where {R <: Real}
    N = get_manifold(mp)
    Y = get_gradient(N, get_objective(mp), ips.p)
    grad_norm = norm(N, ips.p, Y)
    max_step = max_stepsize(N, ips.p)
    return ifelse(isfinite(max_step), min(l, max_step / grad_norm), l)
end

@doc """
    InteriorPointCentralityCondition{CO,R}

A functor to check the centrality condition.

In order to obtain a step in the linesearch performed within the [`interior_point_Newton`](@ref),
Section 6 of [LaiYoshise:2024](@cite) propose the following additional conditions to hold
inspired by the Euclidean case described in Section 6 [El-BakryTapiaTsuchiyaZhang:1996](@cite):

For a given [`ConstrainedManifoldObjective`](@ref) assume consider the [`KKTVectorField`](@ref) ``F``,
that is we are at a point ``q = (p, λ, μ, s)``  on ``$(_math(:Manifold)) × ℝ^m × ℝ^n × ℝ^m``and a search direction ``V = (X, Y, Z, W)``.

Then, let

```math
τ_1 = $(_tex(:frac, "m$(_tex(:min))$(_tex(:set, "μ ⊙ s"))", "μ^{$(_tex(:rm, "T"))}s"))
$(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad))
τ_2 = $(_tex(:frac, "μ^{$(_tex(:rm, "T"))}s", "$(_tex(:norm, "F(q)"))")),
```
where ``⊙`` denotes the Hadamard (or elementwise) product.

For a new candidate ``q(α) = $(_tex(:bigl))(p(α), λ(α), μ(α), s(α)$(_tex(:bigr)) := ($(_tex(:retr))_p(αX), λ+αY, μ+αZ, s+αW)``,
we then define two functions

```math
c_1(α) = $(_tex(:min))$(_tex(:set, "μ(α) ⊙ s(α)")) - $(_tex(:frac, "γτ_1 μ(α)^{$(_tex(:rm, "T"))}s(α)", "m"))
$(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad))
c_2(α) = μ(α)^{$(_tex(:rm, "T"))}s(α) – γτ_2 $(_tex(:norm, "F(q(α))")).
```

While the paper now states that the (Armijo) line search starts at a point
``$(_tex(:tilde)) α``, it is easier to include the condition that ``c_1(α) ≥ 0`` and ``c_2(α) ≥ 0``
into the line search as well.

The functor `InteriorPointCentralityCondition(cmo, γ, μ, s, normKKT)(N,qα)`
defined here evaluates this condition and returns true if both ``c_1`` and ``c_2`` are non-negative.

# Fields

* `cmo`: a [`ConstrainedManifoldObjective`](@ref)
* `γ`: a constant
* `τ1`, `τ2`: the constants given in the formula.

# Constructor

    InteriorPointCentralityCondition(cmo, γ)
    InteriorPointCentralityCondition(cmo, γ, τ1, τ2)

Initialise the centrality conditions.
The parameters `τ1`, `τ2` are initialise to zero if not provided.

!!! note

    Besides [`get_parameter`](@ref) for all three constants,
    and [`set_parameter!`](@ref) for ``γ``,
    to update ``τ_1`` and ``τ_2``, call `set_parameter(ipcc, :τ, N, q)` to update
    both ``τ_1`` and ``τ_2`` according to the formulae above.
"""
mutable struct InteriorPointCentralityCondition{CO, R}
    cmo::CO
    γ::R
    τ1::R
    τ2::R
end
function InteriorPointCentralityCondition(cmo::CO, γ::R) where {CO, R}
    return InteriorPointCentralityCondition{CO, R}(cmo, γ, zero(γ), zero(γ))
end
function (ipcc::InteriorPointCentralityCondition)(N, qα)
    μα = qα[N, 2]
    sα = qα[N, 4]
    m = length(μα)
    # f1 false
    (minimum(μα .* sα) - ipcc.γ * ipcc.τ1 * sum(μα .* sα) / m < 0) && return false
    normKKTqα = sqrt(KKTVectorFieldNormSq(ipcc.cmo)(N, qα))
    # f2 false
    (sum(μα .* sα) - ipcc.γ * ipcc.τ2 * normKKTqα < 0) && return false
    return true
end
function get_parameter(ipcc::InteriorPointCentralityCondition, ::Val{:γ})
    return ipcc.γ
end
function set_parameter!(ipcc::InteriorPointCentralityCondition, ::Val{:γ}, γ)
    ipcc.γ = γ
    return ipcc
end
function get_parameter(ipcc::InteriorPointCentralityCondition, ::Val{:τ1})
    return ipcc.τ1
end
function get_parameter(ipcc::InteriorPointCentralityCondition, ::Val{:τ2})
    return ipcc.τ2
end
function set_parameter!(ipcc::InteriorPointCentralityCondition, ::Val{:τ}, N, q)
    μ = q[N, 2]
    s = q[N, 4]
    m = length(μ)
    normKKTq = sqrt(KKTVectorFieldNormSq(ipcc.cmo)(N, q))
    ipcc.τ1 = m * minimum(μ .* s) / sum(μ .* s)
    ipcc.τ2 = sum(μ .* s) / normKKTq
    return ipcc
end

@doc """
    InteriorPointNewtonState{P,T} <: AbstractHessianSolverState

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
* `λ`:           the Lagrange multiplier with respect to the equality constraints
* `μ`:           the Lagrange multiplier with respect to the inequality constraints
$(_fields(:p; add_properties = [:as_Iterate]))
* `s`:           the current slack variable
$(_fields(:sub_problem))
$(_fields(:sub_state))
* `X`:           the current gradient with respect to `p`
* `Y`:           the current gradient with respect to `μ`
* `Z`:           the current gradient with respect to `λ`
* `W`:           the current gradient with respect to `s`
* `ρ`:           store the orthogonality `μ's/m` to compute the barrier parameter `β` in the sub problem
* `σ`:           scaling factor for the barrier parameter `β` in the sub problem
$(_fields(:stopping_criterion; name = "stop"))
$(_fields([:retraction_method, :stepsize]))
* `step_problem`: an [`AbstractManoptProblem`](@ref) storing the manifold and objective for the line search
* `step_state`: storing iterate and search direction in a state for the line search, see [`StepsizeState`](@ref)

# Constructor

    InteriorPointNewtonState(
        M::AbstractManifold, cmo::ConstrainedManifoldObjective, sub_problem::Pr, sub_state::St; kwargs...
    )
    InteriorPointNewtonState(
        M::AbstractManifold, cmo::ConstrainedManifoldObjective, sub_problem::Pr; kwargs...
    )
    InteriorPointNewtonState(sub_problem::Pr, sub_state::St; kwargs...)

Initialize the state, where both the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) and the [`ConstrainedManifoldObjective`](@ref)
are used to fill in reasonable defaults for the keywords.
For a closed form solution of the sub solver, you can provide the evaluation either as `St` in the first
constructor or as a keyword like in the second.
The third constructor is considered an internal constructor accepting the same keywords,
but those that are filled by defaults based on `M` or `cmo` become mandatory

# Input

$(_args(:M))
* `cmo`:         a [`ConstrainedManifoldObjective`](@ref)
$(_args([:sub_problem, :sub_state]))

# Keyword arguments

Let `m` and `n` denote the number of inequality and equality constraints, respectively

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `is_feasible_error=:error`: specify how to handle infeasible starting points, see [`is_feasible`](@ref) for options.
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:retraction_method))
* `s=ones(m)` slack variables for the inequality constraints
* `step_objective=`[`ManifoldGradientObjective`](@ref)`(`[`KKTVectorFieldNormSq`](@ref)`(cmo)`, [`KKTVectorFieldNormSqGradient`](@ref)`(cmo)`; evaluation=[`InplaceEvaluation`](@ref)`())`
* `step_problem`: wrap the manifold ``$(_math(:Manifold)) × ℝ^m × ℝ^n × ℝ^m``
* `step_state`: the [`StepsizeState`](@ref) with point and search direction
$(_kwargs(:stepsize; default = " `[`ArmijoLinesearch`](@ref)`()"))
  with the [`InteriorPointCentralityCondition`](@ref) as additional condition to accept a step"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`[` | `](@ref StopWhenAny)[`StopWhenChangeLess`](@ref)`(1e-8)"))
* `vector_space=`[`Rn`](@ref Manopt.Rn): a function that, given an integer, returns the manifold to be used for the vector space components ``ℝ^m,ℝ^n``
* `W=zero(s)` tangent vector (gradient) for the slack variables
* `X=`[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`
* `Y=zero(μ)` tangent vector (gradient) for the inequality constraints
* `Z=zero(λ)` tangent vector (gradient) for the equality constraints
* `λ=zeros(n)` Lagrange multipliers for the equality constraints
* `μ=ones(m)` Lagrange multipliers for the inequality constraints
* `ρ=μ's/m`  storage for the orthogonality check
* `σ=`[`calculate_σ`](@ref)`(M, cmo, p, μ, λ, s)`

and internally `_step_M` and `_step_p` for the manifold and point in the stepsize.
"""
mutable struct InteriorPointNewtonState{
        P, T, Pr <: Union{AbstractManoptProblem, F} where {F}, St <: AbstractManoptSolverState,
        C <: AbstractDict{Symbol},
        V, R <: Real,
        SC <: StoppingCriterion, TRTM <: AbstractRetractionMethod, TStepsize <: Stepsize,
        TStepPr <: AbstractManoptProblem, TStepSt <: AbstractManoptSolverState,
    } <: AbstractHessianSolverState
    callbacks::C
    is_feasible_error::Symbol
    p::P
    retraction_method::TRTM
    s::V
    step_problem::TStepPr
    step_state::TStepSt
    stepsize::TStepsize
    stop::SC
    sub_problem::Pr
    sub_state::St
    W::V
    X::T
    Y::V
    Z::V
    λ::V
    μ::V
    ρ::R
    σ::R
    function InteriorPointNewtonState(
            sub_problem::Pr, sub_state::St;
            callbacks::C = Dict{Symbol, Function}(),
            is_feasible_error::Symbol = :error,
            p::P, retraction_method::RTM, s::V,
            step_problem::StepPr, step_state::StepSt, stepsize::S,
            stopping_criterion::SC = StopAfterIteration(200) | StopWhenChangeLess(1.0e-8),
            λ::V, μ::V,
            W::V = zero(s), X::T, Y::V = zero(μ), Z::V = zero(λ),
            ρ::R, σ::R, kwargs...
        ) where {
            P, T, V, R,
            Pr <: Union{AbstractManoptProblem, F} where {F}, St <: AbstractManoptSolverState,
            C <: AbstractDict{Symbol},
            StepPr <: AbstractManoptProblem, StepSt <: AbstractManoptSolverState,
            SC <: StoppingCriterion, RTM <: AbstractRetractionMethod, S <: Stepsize,
        }
        ips = new{P, T, Pr, St, C, V, R, SC, RTM, S, StepPr, StepSt}()
        ips.callbacks = callbacks
        ips.is_feasible_error = is_feasible_error
        ips.p = p
        ips.retraction_method = retraction_method
        ips.s = s
        ips.step_problem = step_problem; ips.step_state = step_state
        ips.stepsize = stepsize
        ips.stop = stopping_criterion
        ips.sub_problem = sub_problem; ips.sub_state = sub_state
        ips.W = W
        ips.X = X
        ips.Y = Y; ips.Z = Z
        ips.λ = λ; ips.μ = μ
        ips.ρ = ρ; ips.σ = σ
        return ips
    end
    function InteriorPointNewtonState(
            M::AbstractManifold, cmo::ConstrainedManifoldObjective, sub_problem::Pr, sub_state::St;
            callbacks::C = Dict{Symbol, Function}(),
            p = rand(M), X = zero_vector(M, p),
            μ = ones(length(get_inequality_constraint(M, cmo, p, :))),
            λ = zeros(length(get_equality_constraint(M, cmo, p, :))),
            s = ones(length(get_inequality_constraint(M, cmo, p, :))),
            ρ = μ's / length(get_inequality_constraint(M, cmo, p, :)),
            σ = calculate_σ(M, cmo, p, μ, λ, s),
            retraction_method::RTM = default_retraction_method(M),
            step_objective = ManifoldGradientObjective(
                KKTVectorFieldNormSq(cmo), KKTVectorFieldNormSqGradient(cmo);
                evaluation = InplaceEvaluation(),
            ),
            vector_space = Rn,
            _step_M = M × vector_space(length(μ)) × vector_space(length(λ)) × vector_space(length(s)),
            step_problem::StepPr = DefaultManoptProblem(_step_M, step_objective),
            _step_p = rand(_step_M),
            step_state::StepSt = StepsizeState(; p = _step_p, X = zero_vector(_step_M, _step_p)),
            centrality_condition = (N, p) -> true,
            stepsize::S = ArmijoLinesearchStepsize(
                get_manifold(step_problem);
                retraction_method = default_retraction_method(get_manifold(step_problem)),
                initial_stepsize = 1.0, additional_decrease_condition = centrality_condition,
            ),
            kwargs...,
        ) where {
            Pr <: Union{AbstractManoptProblem, F} where {F}, St <: AbstractManoptSolverState,
            C <: AbstractDict{Symbol},
            RTM <: AbstractRetractionMethod, S <: Stepsize,
            StepPr <: AbstractManoptProblem, StepSt <: AbstractManoptSolverState,
        }
        return InteriorPointNewtonState(
            sub_problem, sub_state;
            callbacks = callbacks, p = p, retraction_method = retraction_method, s = s,
            step_problem = step_problem, step_state = step_state, stepsize = stepsize,
            λ = λ, μ = μ, X = X, ρ = ρ, σ = σ,
            kwargs...
        )
    end
end
function InteriorPointNewtonState(
        M::AbstractManifold, cmo::ConstrainedManifoldObjective, sub_problem;
        evaluation::E = AllocatingEvaluation(), kwargs...,
    ) where {E <: AbstractEvaluationType}
    # TODO: wrap the closed form solution `sub_problem` if eval is allocating
    cfs = ClosedFormSubSolverState(; evaluation = evaluation)
    return InteriorPointNewtonState(M, cmo, sub_problem, cfs; kwargs...)
end
# get & set iterate
get_iterate(ips::InteriorPointNewtonState) = ips.p
function set_iterate!(ips::InteriorPointNewtonState, ::AbstractManifold, p)
    ips.p = p
    return ips
end
# get & set gradient (not sure if needed?)
get_gradient(ips::InteriorPointNewtonState) = ips.X
function set_gradient!(ips::InteriorPointNewtonState, ::AbstractManifold, X)
    ips.X = X
    return ips
end
# only message on stepsize for now
function get_message(ips::InteriorPointNewtonState)
    return get_message(ips.stepsize)
end
provided_callbacks(::Type{InteriorPointNewtonState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:BeforeSubsolver, :Stepsize, :Subsolver])
get_callbacks(ips::InteriorPointNewtonState) = ips.callbacks
# pretty print state info
function status_summary(ips::InteriorPointNewtonState; context::Symbol = :default)
    i = get_count(ips, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ips.stop) ? "Yes" : "No"
    _is_inline(context) && (return "$(repr(ips)) – $(Iter) $(has_converged(ips) ? "(converged)" : "")")
    as = _callbacks_summary(ips)
    s = """
    # Solver state for `Manopt.jl`s Interior Point Newton Method
    $Iter
    ## Parameters$(as)
    * ρ: $(ips.ρ)
    * σ: $(ips.σ)
    * retraction method: $(ips.retraction_method)

    ## Stepsize
    $(_in_str(status_summary(ips.stepsize; context = context); indent = 1, headers = 1))

    ## Stopping criterion
    $(_in_str(status_summary(ips.stop; context = context); indent = 1, headers = 1))    This indicates convergence: $Conv"""
    return s
end
function Base.show(io::IO, ipns::InteriorPointNewtonState)
    print(io, "InteriorPointNewtonState(", ipns.sub_problem, ", ", ipns.sub_state, ";")
    print(io, " callbacks = ", ipns.callbacks, ", is_feasibility_error = ", ipns.is_feasible_error, ", retraction_method = ", ipns.retraction_method)
    print(io, ", p = ", ipns.p, ", X = ", ipns.X, ", μ = ", ipns.μ, ", Y = ", ipns.Y)
    print(io, ", λ = ", ipns.λ, ", Z = ", ipns.Z, ", s = ", ipns.s, ", W = ", ipns.W)
    print(io, ", ρ = ", ipns.ρ, ", σ = ", ipns.σ, ", step_problem = ", ipns.step_problem)
    print(io, ", step_state = ", ipns.step_state)
    return print(io, ")")
end

@doc """
    StopWhenKKTResidualLess <: StoppingCriterion

Stop when the KKT residual

```
r^2
= $(_tex(:norm, "$(_tex(:grad))_p $(_tex(:Cal, "L"))(p, μ, λ) "))^2
+ $(_tex(:sum, "i=1", "m")) [μ_i]_{-}^2 + [g_i(p)]_+^2 + $(_tex(:abs, "μ_i g_i(p)"))^2
+ $(_tex(:sum, "j=1", "n")) $(_tex(:abs, "h_i(p)"))^2.
```

is less than a given threshold ``r < ε``.
We use ``[v]_+ = $(_tex(:max))$(_tex(:set, "0,v"))`` and ``[v]_- = $(_tex(:min))$(_tex(:set, "0,t"))``
for the positive and negative part of ``v``, respectively

## Fields

* `ε`: a threshold
* `residual`: store the last residual if the stopping criterion is hit.
$(_fields(:at_iteration))
"""
mutable struct StopWhenKKTResidualLess{R} <: StoppingCriterion
    ε::R
    residual::R
    at_iteration::Int
    function StopWhenKKTResidualLess(ε::R) where {R}
        return new{R}(ε, zero(ε), -1)
    end
end
function (c::StopWhenKKTResidualLess)(
        amp::AbstractManoptProblem, ipns::InteriorPointNewtonState, k::Int
    )
    M = get_manifold(amp)
    (k <= 0) && return false
    # now k > 0
    # Check residual
    μ, λ, s, p = ipns.μ, ipns.λ, ipns.s, ipns.p
    c.residual = 0.0
    m, n = length(ipns.μ), length(ipns.λ)
    # First component
    c.residual += norm(M, p, LagrangianGradient(get_objective(amp), μ, λ)(M, p))
    # ineq constr part
    for i in 1:m
        gi = get_inequality_constraint(amp, ipns.p, i)
        c.residual += min(0.0, μ[i])^2 + max(gi, 0)^2 + abs(μ[i] * gi)^2
    end
    # eq constr part
    for j in 1:n
        hj = get_equality_constraint(amp, ipns.p, j)
        c.residual += abs(hj)^2
    end
    c.residual = sqrt(c.residual)
    if c.residual < c.ε
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenKKTResidualLess)
    if (c.at_iteration >= 0)
        return "After iteration #$(c.at_iteration) the algorithm stopped with a KKT residual $(c.residual) < $(c.ε).\n"
    end
    return ""
end
function status_summary(swrr::StopWhenKKTResidualLess; context::Symbol = :default)
    has_stopped = (swrr.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "‖F(p, λ, μ)‖ < ε = $(swrr.ε):$(_MANOPT_INDENT)" : "Stop when the KKT residual is less than ε = $(swrr.ε)\n$(_MANOPT_INDENT)") * s
end
indicates_convergence(::StopWhenKKTResidualLess) = true
function Base.show(io::IO, c::StopWhenKKTResidualLess)
    return print(io, "StopWhenKKTResidualLess($(c.ε))")
end

@doc """
    calculate_σ(M, cmo, p, μ, λ, s; kwargs...)

Compute the new ``σ`` factor for the barrier parameter in [`interior_point_Newton`](@ref) as

```math
$(_tex(:min))$(_tex(:set, "$(_tex(:frac, "1", "2")), $(_tex(:norm, "F(p; μ, λ, s)"))^{$(_tex(:frac, "1", "2"))}")),
```
where ``F`` is the KKT vector field, hence the [`KKTVectorFieldNormSq`](@ref) is used.

# Keyword arguments

* `vector_space=`[`Rn`](@ref Manopt.Rn) a function that, given an integer, returns the manifold to be used for the vector space components ``ℝ^m,ℝ^n``
* `N` the manifold ``$(_math(:Manifold)) × ℝ^m × ℝ^n × ℝ^m`` the vector field lives on (generated using `vector_space`)
* `q` provide memory on `N` for interims evaluation of the vector field
"""
function calculate_σ(
        N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, p, μ, λ, s; kwargs...
    )
    return calculate_σ(N, get_objective(cmo, true), p, μ, λ, s; kwargs...)
end
function calculate_σ(
        M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s;
        vector_space = Rn,
        N = ProductManifold(
            M, vector_space(length(μ)), vector_space(length(λ)), vector_space(length(s)),
        ),
        q = allocate_result(N, rand),
    )
    q1, q2, q3, q4 = submanifold_components(N, q)
    copyto!(N[1], q1, p)
    q2 .= μ
    q3 .= λ
    q4 .= s
    return min(0.5, (KKTVectorFieldNormSq(cmo)(N, q))^(1 / 4))
end

_doc_IPN_subsystem = """
```math
  $(_tex(:operatorname, "J")) F(p, μ, λ, s)[X, Y, Z, W] = -F(p, μ, λ, s),
  $(_tex(:text, " where "))
  X ∈ $(_math(:TangentSpace)), Y,W ∈ ℝ^m, Z ∈ ℝ^n
```
"""
_doc_IPN = """
    interior_point_Newton(M, f, grad_f, Hess_f, p=rand(M); kwargs...)
    interior_point_Newton(M, cmo::ConstrainedManifoldObjective, p=rand(M); kwargs...)
    interior_point_Newton!(M, f, grad]_f, Hess_f, p; kwargs...)
    interior_point_Newton!(M, cmo::ConstrainedManifoldObjective, p; kwargs...)

perform the interior point Newton method following [LaiYoshise:2024](@cite).

In order to solve the constrained problem

$(_problem(:Constrained))

This algorithms iteratively solves the linear system based on extending the KKT system
by a slack variable `s`.

$(_doc_IPN_subsystem)

see [`CondensedKKTVectorFieldJacobian`](@ref) and [`CondensedKKTVectorField`](@ref), respectively,
for the reduced form, this is usually solved in.
From the resulting `X` and `Z` in the reduced form, the other two, ``Y``, ``W``, are then computed.

From the gradient ``(X,Y,Z,W)`` at the current iterate ``(p, μ, λ, s)``,
a line search is performed using the [`KKTVectorFieldNormSq`](@ref) norm of the KKT vector field (squared)
and its gradient [`KKTVectorFieldNormSqGradient`](@ref) together with the [`InteriorPointCentralityCondition`](@ref).

Note that since the vector field ``F`` includes the gradients of the constraint
functions ``g, h``, its gradient or Jacobian requires the Hessians of the constraints.

For that search direction a line search is performed, that additionally ensures that
the constraints are further fulfilled.

# Input

$(_args([:M, :f, :grad_f, :Hess_f, :p]))

or a [`ConstrainedManifoldObjective`](@ref) `cmo` containing `f`, `grad_f`, `Hess_f`, and the constraints

# Keyword arguments

The keyword arguments related to the constraints (the first eleven) are ignored if you
pass a [`ConstrainedManifoldObjective`](@ref) `cmo`

$(_kwargs(:callbacks; add_properties = [:process_note]))
* `centrality_condition=missing`; an additional condition when to accept a step size.
  This can be used to ensure that the resulting iterate is still an interior point if you provide a check `(N,q) -> true/false`,
  where `N` is the manifold of the `step_problem`.
* `equality_constraints=nothing`: the number ``n`` of equality constraints.
$(_kwargs(:evaluation))
* `g=nothing`: the inequality constraints
* `grad_g=nothing`: the gradient of the inequality constraints
* `grad_h=nothing`: the gradient of the equality constraints
* `gradient_range=nothing`: specify how gradients are represented, where `nothing` is equivalent to [`NestedPowerRepresentation`](@extref `ManifoldsBase.NestedPowerRepresentation`)
* `gradient_equality_range=gradient_range`: specify how the gradients of the equality constraints are represented
* `gradient_inequality_range=gradient_range`: specify how the gradients of the inequality constraints are represented
* `h=nothing`: the equality constraints
* `Hess_g=nothing`: the Hessian of the inequality constraints
* `Hess_h=nothing`: the Hessian of the equality constraints
* `inequality_constraints=nothing`: the number ``m`` of inequality constraints.
* `λ=ones(length(h(M, p)))`: the Lagrange multiplier with respect to the equality constraints ``h``
* `μ=ones(length(g(M, p)))`: the Lagrange multiplier with respect to the inequality constraints ``g``
$(_kwargs(:retraction_method))
* `ρ=μ's / length(μ)`:  store the orthogonality `μ's/m` to compute the barrier parameter `β` in the sub problem.
* `s=copy(μ)`: initial value for the slack variables
* `σ=`[`calculate_σ`](@ref)`(M, cmo, p, μ, λ, s)`:  scaling factor for the barrier parameter `β` in the sub problem, which is updated during the iterations
* `step_objective`: a [`ManifoldGradientObjective`](@ref) of the norm of the KKT vector field [`KKTVectorFieldNormSq`](@ref) and its gradient [`KKTVectorFieldNormSqGradient`](@ref)
* `step_problem`: the manifold ``$(_math(:Manifold)) × ℝ^m × ℝ^n × ℝ^m`` together with the `step_objective`
  as the problem the line search `stepsize=` employs for determining a step size
* `step_state`: the [`StepsizeState`](@ref) with point and search direction
$(_kwargs(:stepsize; default = "`[`ArmijoLinesearch`](@ref)`()"))
  with the `centrality_condition` keyword as additional criterion to accept a step, if this is provided"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`[` | `](@ref StopWhenAny)[`StopWhenKKTResidualLess`](@ref)`(1e-8)"))
  a stopping criterion, by default depending on the residual of the KKT vector field or a maximal number of steps, which ever hits first.
* `sub_kwargs=(;)`: keyword arguments to decorate the sub options, for example debug, that automatically respects the main solvers debug options (like sub-sampling) as well
* `sub_objective`: The [`SymmetricLinearSystemObjective`](@ref) modelling the system of equations to use in the sub solver,
  includes the [`CondensedKKTVectorFieldJacobian`](@ref) ``$(_tex(:Cal, "A"))(X)`` and the [`CondensedKKTVectorField`](@ref) ``b`` in ``$(_tex(:Cal, "A"))(X) + b = 0`` we aim to solve.
  $(_note(:KeywordUsedIn, "sub_problem"))
* `sub_stopping_criterion=`[`StopAfterIteration`](@ref)`(manifold_dimension(M))`[` | `](@ref StopWhenAny)[`StopWhenRelativeResidualLess`](@ref)`(c,1e-8)`, where ``c = $(_tex(:norm, "b"))`` from the system to solve.
  $(_note(:KeywordUsedIn, "sub_state"))
$(_kwargs(:sub_problem; default = "`[`DefaultManoptProblem`](@ref)`(M, sub_objective)"))
$(_kwargs(:sub_state; default = "`[`ConjugateResidualState`](@ref)` "))
* `vector_space=`[`Rn`](@ref Manopt.Rn) a function that, given an integer, returns the manifold to be used for the vector space components ``ℝ^m,ℝ^n``
* `X=`[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`:
  the initial gradient with respect to `p`.
* `Y=zero(μ)`: the initial gradient with respect to `μ`
* `Z=zero(λ)`: the initial gradient with respect to `λ`
* `W=zero(s)`: the initial gradient with respect to `s`
* `is_feasible_error=:error`: specify how to handle infeasible starting points, see [`is_feasible`](@ref) for options.

As well as internal keywords used to set up these given keywords like `_step_M`, `_step_p`, `_sub_M`, `_sub_p`, and `_sub_X`,
that should not be changed.

All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.

!!! note

    The `centrality_condition=missing` disables to check centrality during the line search,
    but you can pass [`InteriorPointCentralityCondition`](@ref)`(cmo, γ)`, where `γ` is a constant,
    to activate this check.

# Output

The obtained approximate constrained minimizer ``p^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details, especially the `return_state=` keyword.
"""

@doc "$(_doc_IPN)"
interior_point_Newton(M::AbstractManifold, args...; kwargs...)
function interior_point_Newton(
        M::AbstractManifold, f, grad_f, Hess_f, p = rand(M);
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        g = nothing, h = nothing,
        grad_g = nothing, grad_h = nothing,
        Hess_g = nothing, Hess_h = nothing,
        inequality_constraints::Union{Integer, Nothing} = nothing,
        equality_constraints::Union{Nothing, Integer} = nothing,
        kwargs...,
    )
    cmo = ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h;
        hess_f = Hess_f, hess_g = Hess_g, hess_h = Hess_h,
        evaluation = evaluation,
        inequality_constraints = inequality_constraints,
        equality_constraints = equality_constraints,
        M = M, p = p,
    )
    return interior_point_Newton(M, cmo, p; evaluation = evaluation, kwargs...)
end
function interior_point_Newton(
        M::AbstractManifold, cmo::O, p; kwargs...
    ) where {O <: Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(interior_point_Newton; kwargs...)
    q = copy(M, p)
    return interior_point_Newton!(M, cmo, q; kwargs...)
end
calls_with_kwargs(::typeof(interior_point_Newton)) = (interior_point_Newton!,)

@doc "$(_doc_IPN)"
interior_point_Newton!(M::AbstractManifold, args...; kwargs...)

function interior_point_Newton!(
        M::AbstractManifold, f, grad_f, Hess_f, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        g = nothing, h = nothing,
        grad_g = nothing, grad_h = nothing,
        Hess_g = nothing, Hess_h = nothing,
        inequality_constraints = nothing,
        equality_constraints = nothing,
        kwargs...,
    )
    cmo = ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h;
        hess_f = Hess_f, hess_g = Hess_g, hess_h = Hess_h,
        evaluation = evaluation,
        equality_constraints = equality_constraints,
        inequality_constraints = inequality_constraints,
        M = M, p = p,
    )
    dcmo = decorate_objective!(M, cmo; kwargs...)
    return interior_point_Newton!(M, dcmo, p; evaluation = evaluation, kwargs...)
end
function interior_point_Newton!(
        M::AbstractManifold, cmo::O, p;
        callbacks = Dict{Symbol, Function}(),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        X = get_gradient(M, cmo, p),
        μ::AbstractVector = ones(inequality_constraints_length(cmo)),
        Y::AbstractVector = zero(μ),
        λ::AbstractVector = zeros(equality_constraints_length(cmo)),
        Z::AbstractVector = zero(λ),
        s::AbstractVector = copy(μ),
        W::AbstractVector = zero(s),
        ρ::Real = μ's / length(μ),
        σ::Real = calculate_σ(M, cmo, p, μ, λ, s),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        sub_kwargs = (;),
        vector_space = Rn,
        #γ=0.9,
        centrality_condition = missing, #InteriorPointCentralityCondition(cmo, γ, zero(γ), zero(γ)),
        step_objective = ManifoldGradientObjective(
            KKTVectorFieldNormSq(cmo), KKTVectorFieldNormSqGradient(cmo); evaluation = evaluation
        ),
        _step_M::AbstractManifold = ProductManifold(
            M, vector_space(length(μ)), vector_space(length(λ)), vector_space(length(s)),
        ),
        step_problem = DefaultManoptProblem(_step_M, step_objective),
        _step_p = rand(_step_M),
        step_state = StepsizeState(; p = _step_p, X = zero_vector(_step_M, _step_p)),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = ArmijoLinesearch(
            _step_M;
            retraction_method = default_retraction_method(_step_M),
            initial_guess = interior_point_initial_guess,
            additional_decrease_condition = if ismissing(centrality_condition)
                (M, p) -> true
            else
                centrality_condition
            end,
        ),
        stopping_criterion::StoppingCriterion = StopAfterIteration(800) |
            StopWhenKKTResidualLess(1.0e-8),
        _sub_M = ProductManifold(M, vector_space(length(λ))),
        _sub_p = rand(_sub_M),
        _sub_X = rand(_sub_M; vector_at = _sub_p),
        sub_objective = decorate_objective!(
            TangentSpace(_sub_M, _sub_p),
            SymmetricLinearSystemObjective(
                CondensedKKTVectorFieldJacobian(cmo, μ, s, σ * ρ),
                CondensedKKTVectorField(cmo, μ, s, σ * ρ),
            ),
            sub_kwargs...,
        ),
        sub_stopping_criterion::StoppingCriterion = StopAfterIteration(manifold_dimension(M)) |
            StopWhenRelativeResidualLess(
            norm(_sub_M, _sub_p, get_vector_field(TangentSpace(_sub_M, _sub_p), sub_objective)), 1.0e-8
        ),
        sub_state::St = decorate_state!(
            ConjugateResidualState(
                TangentSpace(_sub_M, _sub_p),
                sub_objective;
                X = _sub_X,
                stop = sub_stopping_criterion,
                sub_kwargs...,
            );
            sub_kwargs...,
        ),
        sub_problem::Pr = DefaultManoptProblem(TangentSpace(_sub_M, _sub_p), sub_objective),
        is_feasible_error = :error,
        kwargs...,
    ) where {
        O <: Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective},
        St <: AbstractManoptSolverState,
        Pr <: Union{F, AbstractManoptProblem} where {F},
    }
    !is_feasible(M, cmo, p; error = is_feasible_error)
    keywords_accepted(interior_point_Newton!; kwargs...)
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = InteriorPointNewtonState(
        M, cmo, sub_problem, sub_state;
        callbacks = process_callbacks_arg(callbacks, InteriorPointNewtonState),
        p = p, X = X, Y = Y, Z = Z, W = W, μ = μ, λ = λ, s = s,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        step_problem = step_problem, step_state = step_state,
        stepsize = _produce_type(stepsize, _step_M, _step_p),
        is_feasible_error = is_feasible_error,
        kwargs...,
    )
    ips = decorate_state!(ips; kwargs...)
    solve!(dmp, ips)
    return get_solver_return(get_objective(dmp), ips)
end
calls_with_kwargs(::typeof(interior_point_Newton!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(amp::AbstractManoptProblem, ips::InteriorPointNewtonState)
    M = get_manifold(amp)
    cmo = get_objective(amp)
    !is_feasible(M, cmo, ips.p; error = ips.is_feasible_error)
    return ips
end

function step_solver!(amp::AbstractManoptProblem, ips::InteriorPointNewtonState, k)
    M = get_manifold(amp)
    cmo = get_objective(amp)
    N = base_manifold(get_manifold(ips.sub_problem))
    q = base_point(get_manifold(ips.sub_problem))
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.λ)
    set_iterate!(ips.sub_state, get_manifold(ips.sub_problem), zero_vector(N, q))

    set_parameter!(ips.sub_problem, :Manifold, :Basepoint, q)
    set_parameter!(ips.sub_problem, :Objective, :μ, ips.μ)
    set_parameter!(ips.sub_problem, :Objective, :λ, ips.λ)
    set_parameter!(ips.sub_problem, :Objective, :s, ips.s)
    set_parameter!(ips.sub_problem, :Objective, :β, ips.ρ * ips.σ)
    # product manifold on which to perform linesearch
    callback(:BeforeSubsolver, amp, ips, k)
    X2 = get_solver_result(solve!(ips.sub_problem, ips.sub_state))
    callback(:Subsolver, amp, ips, k)
    ips.X, ips.Z = submanifold_components(N, X2) #for p and λ

    # Compute the remaining part of the solution
    m, n = length(ips.μ), length(ips.λ)
    if m > 0
        g = get_inequality_constraint(amp, ips.p, :)
        grad_g = get_grad_inequality_constraint(amp, ips.p, :)
        β = ips.ρ * ips.σ
        # for s and μ
        ips.W .= [-inner(M, ips.p, grad_g[i], ips.X) for i in 1:m] .- g .- ips.s
        ips.Y .= (β .- ips.μ .* (ips.s + ips.W)) ./ ips.s
    end

    N = get_manifold(ips.step_problem)
    # generate current full iterate in step state
    q = get_iterate(ips.step_state)
    q1, q2, q3, q4 = submanifold_components(N, q)
    copyto!(N[1], q1, get_iterate(ips))
    q2 .= ips.μ
    q3 .= ips.λ
    q4 .= ips.s
    set_iterate!(ips.step_state, M, q)
    # generate current full gradient in step state
    X = get_gradient(ips.step_state)
    copyto!(N[1], X[N, 1], ips.X)
    (m > 0) && (copyto!(N[2], X[N, 2], ips.Z))
    (n > 0) && (copyto!(N[3], X[N, 3], ips.Y))
    (m > 0) && (copyto!(N[4], X[N, 4], ips.W))
    set_gradient!(ips.step_state, M, q, X)
    # Update centrality factor – Maybe do this as an update function?
    γ = get_parameter(ips.stepsize, :DecreaseCondition, :γ)
    if !isnothing(γ)
        set_parameter!(ips.stepsize, :DecreaseCondition, :γ, (γ + 0.5) / 2)
    end
    set_parameter!(ips.stepsize, :DecreaseCondition, :τ, N, q)
    # determine stepsize
    α = ips.stepsize(ips.step_problem, ips.step_state, k; gradient = X)
    callback(:Stepsize, amp, ips, k)
    # Update Parameters and slack
    retract!(M, ips.p, ips.p, α * ips.X, ips.retraction_method)
    if m > 0
        ips.μ .+= α .* ips.Y
        ips.s .+= α .* ips.W
        ips.ρ = ips.μ'ips.s / m
        # we can use the memory from above still
        ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s; N = N, q = q)
    end
    (n > 0) && (ips.λ .+= α .* ips.Z)
    return ips
end

get_solver_result(ips::InteriorPointNewtonState) = ips.p
