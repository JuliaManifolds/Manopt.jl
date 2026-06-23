#
#
# The solver state
@doc """
    LevenbergMarquardtState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields

* `damping_term`:                         current value of the damping term
* `damping_term_min`:                     lower bound for the damping term
* `damping_term_max`:                     upper bound for the damping term
* `damping_increase_factor`:              improvement quotient exceeds `damping_reduction_threshold`.
* `damping_reduction_threshold`:          threshold for the improvement quotient above which
  the damping term is reduced by multiplying it with `β_reduction`.
* `damping_increase_threshold` :          threshold for the improvement quotient below which
  the damping term is increased by multiplying it with `β`.
* `direction`:                            the current search direction, which is the solution of
  the linearized subproblem in each iteration.
* `candidate_acceptance_threshold`:       Scaling factor for the sufficient cost decrease threshold required
  to accept new proposal points. Allowed range: `0 < η < 1`.
* `jacobian_matrices`:                           the current Jacobian of ``F`` in matrix form per block, hence a vector of matrices.
   This is (by default) set to `nothing` if another representation is used.
* `minimum_acceptable_model_improvement`: the minimum improvement in the model function that
  is required to accept a new point; if this is not met, the new point is rejected and
  the damping term is increased.
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:retraction_method))
* `residual_values`:                       values of the residuals calculated in the solver setup or the previous iteration
$(_fields(:stopping_criterion; name = "stop"))
$(_fields(:sub_problem))
$(_fields(:sub_state))
$(_fields(:X))

# Constructor

    LevenbergMarquardtState(M, sub_problem, sub_state, initial_residual_values, initial_jacobian; kwargs...)

Generate the Levenberg-Marquardt solver state.

# Keyword arguments

The following fields are keyword arguments

* `candidate_acceptance_threshold = 0.2`,
* `damping_increase_factor = 5.0`
* `damping_reduction_factor = 0.5`
* `damping_term_min = 0.1`
* `damping_term_max = Inf`
* `damping_term = damping_term_min`
* `damping_reduction_threshold = Inf`
* `damping_increase_threshold = candidate_acceptance_threshold`
* `direction = copy(M, p, X)`
* `p = `$(_link(:rand))
* `X = `$(_link(:zero_vector))
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-12)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-12)"))
* `minimum_acceptable_model_improvement::Real = eps(number_eltype(p))`

# See also

[`gradient_descent`](@ref), [`LevenbergMarquardt`](@ref)
"""
mutable struct LevenbergMarquardtState{
        P, T, R <: Real, Pr, St, TStop <: StoppingCriterion, TRTM <: AbstractRetractionMethod, TRes, TJac,
    } <: AbstractGradientSolverState
    candidate_acceptance_threshold::R
    damping_increase_factor::R
    damping_increase_threshold::R
    damping_reduction_threshold::R
    damping_reduction_factor::R
    damping_term::R
    damping_term_min::R
    damping_term_max::R
    direction::T
    jacobian_matrices::TJac
    minimum_acceptable_model_improvement::R
    p::P
    q::P
    residual_values::TRes
    retraction_method::TRTM
    stop::TStop
    sub_problem::Pr
    sub_state::St
    X::T
    function LevenbergMarquardtState(
            sub_problem::Pr, sub_state::St;
            candidate_acceptance_threshold::R, damping_increase_factor::R, damping_increase_threshold::R,
            damping_reduction_threshold::R, damping_reduction_factor::R, damping_term::R,
            damping_term_min::R, damping_term_max::R,
            direction::T, jacobian_matrices::TJac, minimum_acceptable_model_improvement::R, p::P, q::P,
            residual_values::TRes, retraction_method::TRTM, stopping_criterion::SC, X::T
        ) where {P, T, R <: Real, Pr, St <: AbstractManoptSolverState, SC <: StoppingCriterion, TRTM <: AbstractRetractionMethod, TRes, TJac}
        return new{P, T, R, Pr, St, SC, TRTM, TRes, TJac}(
            candidate_acceptance_threshold, damping_increase_factor, damping_increase_threshold,
            damping_reduction_threshold, damping_reduction_factor, damping_term, damping_term_min, damping_term_max,
            direction, jacobian_matrices, minimum_acceptable_model_improvement, p, q, residual_values,
            retraction_method, stopping_criterion, sub_problem, sub_state, X
        )
    end
    function LevenbergMarquardtState(
            M::AbstractManifold, sub_problem, sub_state, initial_residual_values, initial_jacobian_matrices = nothing;
            p = rand(M), X = zero_vector(M, p), direction = copy(M, p, X),
            stopping_criterion::StoppingCriterion = StopAfterIteration(200) | StopWhenGradientNormLess(1.0e-12) | StopWhenStepsizeLess(1.0e-12),
            retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
            candidate_acceptance_threshold::Real = 0.2,
            damping_increase_factor::Real = 5.0,
            damping_increase_threshold::Real = candidate_acceptance_threshold,
            damping_reduction_threshold::Real = Inf,
            damping_reduction_factor::Real = 0.5,
            damping_term_min::Real = 0.1,
            damping_term_max::Real = Inf,
            damping_term::Real = damping_term_min,
            minimum_acceptable_model_improvement::Real = eps(number_eltype(p)),
        )
        (candidate_acceptance_threshold <= 0 || candidate_acceptance_threshold >= 1) && throw(ArgumentError("The value of `candidate_acceptance_threshold` must be strictly between 0 and 1, received $(candidate_acceptance_threshold)"))
        (damping_term_min <= 0) && throw(ArgumentError("The value of damping_term_min must be strictly above 0, received $damping_term_min"))
        (damping_increase_factor <= 1) && throw(ArgumentError("The value of `damping_increase_factor must be strictly above 1, received $damping_increase_factor"))
        (damping_reduction_factor >= 1) && throw(ArgumentError("The value of `damping_reduction_factor must be strictly below 1, received $β_reduction"))
        _sub_state = maybe_wrap_evaluation_type(sub_state)
        R = promote_type(
            typeof(candidate_acceptance_threshold), typeof(damping_term_min), typeof(damping_increase_factor), typeof(damping_increase_threshold),
            typeof(damping_reduction_threshold), typeof(damping_reduction_factor), typeof(damping_term_min),
            typeof(damping_term_max), typeof(damping_term), typeof(minimum_acceptable_model_improvement)
        )
        return LevenbergMarquardtState(
            sub_problem, _sub_state;
            candidate_acceptance_threshold = convert(R, candidate_acceptance_threshold),
            damping_increase_factor = convert(R, damping_increase_factor), damping_increase_threshold = convert(R, damping_increase_threshold),
            damping_reduction_threshold = convert(R, damping_reduction_threshold), damping_reduction_factor = convert(R, damping_reduction_factor),
            damping_term = convert(R, damping_term), damping_term_min = convert(R, damping_term_min), damping_term_max = convert(R, damping_term_max),
            direction = direction, jacobian_matrices = initial_jacobian_matrices, minimum_acceptable_model_improvement = convert(R, minimum_acceptable_model_improvement),
            p = p, q = copy(M, p), residual_values = initial_residual_values, retraction_method = retraction_method, stopping_criterion = stopping_criterion, X = X,
        )
    end
end
#
function status_summary(lms::LevenbergMarquardtState; context::Symbol = :default)
    (context === :short) && return repr(lms)
    i = get_count(lms, :Iterations)
    conv_inl = (i > 0) ? (indicates_convergence(lms.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the Levenberg–Marquardt algorithm$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(lms.stop) ? "Yes" : "No"
    return """
    # Solver state for `Manopt.jl`s Levenberg Marquardt Algorithm
    $Iter
    ## Parameters
    * candidate acceptance threshold:$(_MANOPT_INDENT)$(lms.candidate_acceptance_threshold)
    * damping reduction threshold:   $(_MANOPT_INDENT)$(lms.damping_reduction_threshold)
    * damping reduction factor:      $(_MANOPT_INDENT)$(lms.damping_reduction_factor)
    * damping increase threshold:    $(_MANOPT_INDENT)$(lms.damping_increase_threshold)
    * damping increase factor:       $(_MANOPT_INDENT)$(lms.damping_increase_factor)
    * damping term:                  $(_MANOPT_INDENT)$(lms.damping_term) (min: $(lms.damping_term_min) | max: $(lms.damping_term_max))
    * retraction method:             $(_MANOPT_INDENT)$(lms.retraction_method)

    ## Stopping criterion

    $(status_summary(lms.stop; context = context))
    This indicates convergence: $Conv"""
end
function show(io::IO, lms::LevenbergMarquardtState)
    print(io, "LevenbergMarquardtState(", lms.sub_problem, ", ", lms.sub_state, "; ")
    print(io, "candidate_acceptance_threshold = ", lms.candidate_acceptance_threshold)
    print(io, ", damping_increase_factor = ", lms.damping_increase_factor, ", damping_increase_threshold = ", lms.damping_increase_threshold)
    print(io, ", damping_reduction_threshold = ", lms.damping_reduction_threshold, ", damping_reduction_factor = ", lms.damping_reduction_factor)
    print(io, ", damping_term = ", lms.damping_term, ", damping_term_min = ", lms.damping_term_min, ", damping_term_max = ", lms.damping_term_max)
    print(io, ", direction = ", lms.direction, ", jacobian_matrices = ", lms.jacobian_matrices, ". minimum_acceptable_model_improvement = ", lms.minimum_acceptable_model_improvement)
    print(io, ", p= ", lms.p, ", q = ", lms.q, ", residual_values = ", lms.residual_values, ", retraction_method = ", lms.retraction_method, ", stopping_criterion = ", lms.stop, ", X = ", lms.X)
    return print(io, ")")
end

#
#
# --- Subproblems ----
"""
    AbstractLevenbergMarquardtLinearSurrogateObjective{E<:AbstractEvaluationType}

Abstract supertype for Levenberg-Marquardt surrogates like
[`LevenbergMarquardtLinearSurrogateObjective`](@ref) and
[`LevenbergMarquardtLinearSurrogateCoordinatesObjective`](@ref).
"""
abstract type AbstractLevenbergMarquardtLinearSurrogateObjective{E <: AbstractEvaluationType} <: AbstractLinearSurrogateObjective{E, ManifoldNonlinearLeastSquaresObjective{E}} end

@doc """
    LevenbergMarquardtLinearSurrogateObjective{E<:AbstractEvaluationType, VF<:AbstractManifoldFirstOrderObjective{E}, R} <: AbstractLevenbergMarquardtLinearSurrogateObjective{E}

Given an [`ManifoldNonlinearLeastSquaresObjective`](@ref) `objective` and a `penalty` ``λ``,
this objective represents the penalized objective for the sub-problem to solve within every step
of the Levenberg-Marquardt algorithm following the ideas of [TriggsMcLauchlanHartleyFitzgibbon:2000](@cite) given by

```math
μ_p(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, _tex(:Cal, "L") * "(X) + y"; index = "2"))^2
  + $(_tex(:frac, "λ", "2"))$(_tex(:norm, "X"; index = "p"))^2,
  $(_tex(:qquad))$(_tex(:text, " for "))X ∈ $(_math(:TangentSpace)), λ ≥ 0,
```

where ``X ∈ $(_math(:TangentSpace))``, ``λ ≥ 0`` is the damping or penalty term,
``$(_tex(:Cal, "L")): $(_math(:TangentSpace)) → ℝ^n`` is a linear operator,
and ``y = y(p) ∈ ℝ^n`` is a vector field.
For the derivation of the Riemannian case, see [BaranBergmann:2026](@cite).

In order to build a surrogate also for the robustified Levenberg-Marquardt, introduce
``α = 1 - $(_tex(:sqrt, "1 + 2 $(_tex(:frac, "ρ''(p)", "ρ'(p)"))$(_tex(:norm, "F(p)"; index = "2"))^2"))``
and set ``y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p)`` and ``$(_tex(:Cal, "L"))(X) = CJ_F^*(p)[F(p)]``
with

```math
C = $(_tex(:sqrt, "ρ'(p)"))(I-αP), $(_tex(:qquad)) P = $(_tex(:frac, "F(p)F(p)^" * _tex(:rm, "T"), _tex(:norm, "F(p)"; index = "2") * "^2")),
```

where ``F(p) ∈ ℝ^n`` is the vector of residuals at point ``p ∈ M`` and ``J_F^*(p): ℝ^n → $(_math(:TangentSpace))``
is the adjoint Jacobian.
These two can be accessed with [`get_vector_field`](@ref) for ``y`` and [`get_linear_operator`](@ref) for ``$(_tex(:Cal, "L"))``,
respectively.
For technical details on the scaling using ``α``, especially how the `threshold` and `mode`
act as safeguards, see [`get_LevenbergMarquardt_scaling`](@ref)

## Fields

* `objective`:     the [`ManifoldNonlinearLeastSquaresObjective`](@ref) to penalize
* `penalty::Real`: the damping term ``λ``
* `threshold::Real`: threshold ``ε`` for stabilization of ``α`` as ``α ≤ 1-ε``, see  [`get_LevenbergMarquardt_scaling`](@ref)
* `mode::Symbol`:  which ode to use to stabilize α, see the internal helper [`get_LevenbergMarquardt_scaling`](@ref)
* `value_cache`:   a vector to store the residuals ``F(p)`` at the current point `p` internally to avoid re-computations

## Constructor

    LevenbergMarquardtLinearSurrogateObjective(objective; penalty::Real = 1e-6, threshold::Real = 1e-4, mode::Symbol = :Default)
"""
mutable struct LevenbergMarquardtLinearSurrogateObjective{
        E <: AbstractEvaluationType, R <: Real, TO <: ManifoldNonlinearLeastSquaresObjective{E}, TVC <: AbstractVector{R},
    } <: AbstractLevenbergMarquardtLinearSurrogateObjective{E}
    objective::TO
    penalty::R
    threshold::R
    mode::Symbol
    value_cache::TVC
    function LevenbergMarquardtLinearSurrogateObjective(
            objective::ManifoldNonlinearLeastSquaresObjective{E};
            penalty::R = 1.0e-6, threshold::R = 1.0e-4, mode::Symbol = :Default,
            residuals::TVC = zeros(residuals_count(get_objective(objective))),
        ) where {E, R <: Real, TVC <: AbstractVector}
        return new{E, R, typeof(objective), TVC}(objective, penalty, threshold, mode, residuals)
    end
end

function show(io::IO, o::LevenbergMarquardtLinearSurrogateObjective)
    return print(io, "LevenbergMarquardtLinearSurrogateObjective($(o.objective); penalty=$(o.penalty), threshold=$(o.threshold), mode=:$(o.mode))")
end

function status_summary(lmlso::LevenbergMarquardtLinearSurrogateObjective; context::Symbol = :default)
    (context === :short) && (return repr(lmlso))
    (context === :inline) && (return "A linear surrogate objective for the Levenberg Marquardt algorithm based on $(status_summary(lmlso.objective; context = context)) with penalty $(lmlso.penalty)")
    return """
    A linear surrogate objective for the Levenberg Marquardt Algorithm

    ## Objective
    $(_in_str(status_summary(lmlso.objective, context = context); indent = 1))

    ## Parameters
    * penalty:   $(_MANOPT_INDENT)$(lmlso.penalty)
    * threshold: $(_MANOPT_INDENT)$(lmlso.threshold)
    * mode:      $(_MANOPT_INDENT)$(lmlso.mode)
    """
end


"""
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime::Real, ρ_double_prime::Real, FSq::Real, threshold::Real=1.0e-5, mode::Symbol=:Default)

Compute the scalings for the residual ``y`` and within the operator ``C`` that are required for the robust
rescaling within [`LevenbergMarquardt`](@ref)s [`get_vector_field`](@ref) and [`get_linear_operator`](@ref),
respectively.
Here `FSq` denotes ``s = $(_tex(:norm, "F(p)"; index = "2"))^2`` of the residual vector function ``F`` evaluated at some point ``p``,
and `ρ_prime``=ρ'(s)` and `ρ_double_prime``=ρ''(s)` denote the current [`AbstractRobustifierFunction`](@ref)s
first and second derivative evaluated at ``s``.

The value for ``α`` is given by

```math
    α = 1 - $(_tex(:sqrt, "1 + 2$(_tex(:frac, "ρ''(s)", "ρ'(s)"))s"))
```

and hence the scaling of the residual and the within the projection of the operator are

```math
$(_tex(:frac, _tex(:sqrt, "ρ'(s)"), "1-α"))
$(_tex(:qquad))$(_tex(:text, " and "))$(_tex(:qquad))
$(_tex(:cases, "$(_tex(:frac, "α", "s")) & $(_tex(:text, " if ")) s ≠ 0", "0 & $(_tex(:text, " else,"))"))
```

respectively.

## Numerical stability

For a unique solution that is a minimizer in a Levenberg-Marquardt step,
we require `α < 1` and [TriggsMcLauchlanHartleyFitzgibbon:2000](@cite) recommends to bound this even by ``1-ε``
for some `threshold` ``ε > 0``.

Furthermore if ``ρ'(s) + 2ρ''(s)⋅s ≤ 0`` the Hessian is also indefinite.
This can be caught by making sure the argument of the ``√`` is ensured to be non-negative.

The [Ceres solver](http://ceres-solver.org/nnls_modeling.html#theory) even omits the second term
in the square root already if ``ρ(s)'' < 0`` for stability reason, which means setting ``α = 0``.
In the case ``s = 0`` we also set the operator scaling ``α / s = 0``.

This function offers two `mode`s
- `:Default` keeps negative ``ρ''(s) < 0`` but makes sure the square root is well-defined.
- `:Strict` set ``α = 0`` when ``ρ''(s) < 0`` or when ``s = 0``
"""
function get_LevenbergMarquardt_scaling(
        ρ_prime::Real, ρ_double_prime::Real, FkSq::Real,
        threshold::Real = 1.0e-5, mode::Symbol = :Default
    )
    # second derivative existent and negative: In strict mode (motivated by ceres) -> return sqrt(ρ_prime), 0
    (ismissing(ρ_double_prime) || (ρ_double_prime < 0 && mode == :Strict)) && return (sqrt(ρ_prime), 0.0)
    (iszero(FkSq) && mode == :Strict) && return (sqrt(ρ_prime), 0.0)
    α = 1 - sqrt(max(1 + 2 * (ρ_double_prime / ρ_prime) * FkSq, 0.0))
    α = min(α, 1 - threshold)
    residual_scaling = sqrt(ρ_prime) / (1 - α)
    operator_scaling = ifelse(iszero(FkSq), 0.0, α / FkSq)
    return residual_scaling, operator_scaling
end
function set_parameter!(lmlso::LevenbergMarquardtLinearSurrogateObjective, ::Val{:Penalty}, penalty::Real)
    lmlso.penalty = penalty
    return lmlso
end

"""
    get_cost(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )

Compute the surrogate cost. Let ``F`` denote the vector of residuals (of a block),
``ρ, ρ'``, ``ρ''`` the value, first, and second derivative of the [`AbstractRobustifierFunction`](@ref)
of the inner [`ManifoldNonlinearLeastSquaresObjective`](@ref)

```math
σ_k(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, "y + $(_tex(:Cal, "L"))(X)"; index = "2"))^2, $(_tex(:qquad)) X ∈ $(_math(:TangentSpace))
```

where
* ``$(_tex(:Cal, "L"))(X) = CJ[X]`` see [`get_linear_operator`](@ref) with a `penalty` of zero.
* ``y`` the rescaled vector field, see [`get_vector_field`](@ref)
"""
function get_cost(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    cost = norm(get_linear_operator(M, lmsco, p, X) + get_vector_field(M, lmsco, p))^2 / 2
    # add the damping term
    cost += (lmsco.penalty / 2) * norm(M, p, X)^2
    return cost
end
function get_cost(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, ::ZeroTangentVector
    )
    cost = norm(get_vector_field(M, lmsco, p))^2 / 2
    return cost
end

_docs_grad_LMSurrogate_grad = """
    get_gradient(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X)
    get_gradient!(M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X)

Compute the gradient of the [`LevenbergMarquardtLinearSurrogateObjective`](@ref), which is given by

```math
$(
    _tex(
        :aligned,
        "$(_tex(:grad)) μ_p(X) &= $(_tex(:sum, "i=1", "m")) $(_tex(:Cal, "L"))_i^*$(_tex(:bigl))($(_tex(:Cal, "L"))_i(X) + y_i $(_tex(:bigr))) + λX",
        """&= $(_tex(:sum, "i=1", "m")) J_{F_i}^*(p)$(_tex(:Bigl))[
        ρ_i' $(_tex(:bigl))(I- b F_i(p)F_i(p)^{$(_tex(:transp))}$(_tex(:bigr)))^2 J_{F_i}(p)[X] + a$(_tex(:sqrt, "ρ_i'"))$(_tex(:bigl))(I- b F_i(p)F_i(p)^{$(_tex(:transp))}$(_tex(:bigr))) F_i(p) + λX
        $(_tex(:Bigr))]"""
    )
)
```
where ``ρ_i' = ρ_i'($(_tex(:norm, "F_i(p)"))_2^2)``, ``ρ_i'' = ρ_i''($(_tex(:norm, "F_i(p)"))_2^2)``
are the values from the [`AbstractRobustifierFunction`](@ref) `ρ` its first and second derivative, respectively,
and ``a,b`` are the [`get_LevenbergMarquardt_scaling`](@ref) values of scaling the residual and operator, respectively.
See also [`get_jacobian`](@ref) and [`get_adjoint_jacobian`](@ref).

This can be computed inplace of `Y`.
"""

@doc "$(_docs_grad_LMSurrogate_grad)"
function get_gradient(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    Y = zero_vector(M, p)
    return get_gradient!(M, Y, lmsco, p, X)
end
@doc "$(_docs_grad_LMSurrogate_grad)"
function get_gradient!(
        M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    value_cache = lmsco.value_cache
    nlso = lmsco.objective
    # For every block
    zero_vector!(M, Y, p)
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len_o = length(o)
        _add_gradient!(
            M, Y, o, r, p, X;
            value_cache = value_cache[(start + 1):(start + len_o)], threshold = lmsco.threshold, mode = lmsco.mode,
        )
        start += len_o
    end
    # add penalty term
    Y .+= lmsco.penalty .* X
    return Y
end
# For each single summand, we are on the level of a single vectorial function and a robustifier – and add it directly
function _add_gradient!(
        M::AbstractManifold, Y, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol, jacobian_cache = nothing
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared indirectly
    b = zero(a)
    get_jacobian!(M, b, o, p, X)
    # Compute C^TCb = C^2 b (inplace of b)
    b .= ρ_prime .* (I - operator_scaling * (a * a'))^2 * b
    # add C^T y = C^T (sqrt(ρ(p)) / (1 - α) F(p)) (which overall has a ρ_prime upfront)
    b .+= residual_scaling .* sqrt(ρ_prime) .* (I - operator_scaling * (a * a')) * a
    # apply the adjoint
    add_adjoint_jacobian!(M, Y, o, p, b)
    return Y
end
# Componentwise
function _add_gradient!(
        M::AbstractManifold, Y, o::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol, jacobian_cache = nothing
    )
    # per single component a for-loop similar to the one for the blocks
    r = cr.robustifier
    b = zero(value_cache)
    get_jacobian!(M, b, o, p, X)
    # Componentwise a few things decouple
    for (i, ai) in enumerate(value_cache)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, threshold, mode)
        # get the “Jacobian” of the ith component, i.e. its
        # Compute C^TCa = C^2 a (inplace of a)
        b[i] = ρ_prime * (1 - operator_scaling * ai_sq)^2 * b[i]
        # add C^T y = C^T (sqrt(ρ(p)) / (1 - α) F(p)) (which overall has a ρ_prime upfront)
        b[i] += residual_scaling * sqrt(ρ_prime) * (1 - operator_scaling * ai_sq) * ai
    end
    # apply the adjoint
    add_adjoint_jacobian!(M, Y, o, p, b)
    return Y
end


_docs_grad_LMSurrogate_Hess = """
    get_hessian(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X, Y)
    get_hessian!(M::AbstractManifold, Z, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X, Y)

Compute the Hessian of the [`LevenbergMarquardtLinearSurrogateObjective`](@ref), which is given by

```math
$(
    _tex(
        :aligned,
        "$(_tex(:Hess)) μ_p(X)[Y] &= $(_tex(:sum, "i=1", "m")) $(_tex(:Cal, "L"))_i^*$(_tex(:bigl))($(_tex(:Cal, "L"))_i(Y)$(_tex(:bigr))) + λY",
        """&= $(_tex(:sum, "i=1", "m")) J_{F_i}^*(p)$(_tex(:Bigl))[
        ρ_i' $(_tex(:bigl))(I- b F_i(p)F_i(p)^{$(_tex(:transp))}$(_tex(:bigr)))^2 J_{F_i}(p)[Y] + λY
        $(_tex(:Bigr))]"""
    )
)
```
where ``ρ_i' = ρ_i'($(_tex(:norm, "F_i(p)"))_2^2)``, ``ρ_i'' = ρ_i''($(_tex(:norm, "F_i(p)"))_2^2)``
are the values from the [`AbstractRobustifierFunction`](@ref) `ρ` its first and second derivative, respectively,
and ``b`` is the [`get_LevenbergMarquardt_scaling`](@ref) values of scaling the operator.
See also [`get_jacobian`](@ref) and [`get_adjoint_jacobian`](@ref).

This can be computed inplace of `Z`.
"""


@doc "$(_docs_grad_LMSurrogate_Hess)"
function get_hessian(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X, Y
    )
    Z = zero_vector(M, p)
    return get_hessian!(M, Z, lmsco, p, X, Y)
end
@doc "$(_docs_grad_LMSurrogate_Hess)"
function get_hessian!(
        M::AbstractManifold, Z, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X, Y
    )
    value_cache = lmsco.value_cache
    nlso = lmsco.objective
    # For every block
    zero_vector!(M, Z, p)
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len_o = length(o)
        _add_hessian!(
            M, Z, o, r, p, X, Y;
            value_cache = value_cache[(start + 1):(start + len_o)], threshold = lmsco.threshold, mode = lmsco.mode,
        )
        start += len_o
    end
    # add penalty term
    Z .+= lmsco.penalty .* Y
    return Z
end
# For each single summand, we are on the level of a single vectorial function and a robustifier.
function _add_hessian!(
        M::AbstractManifold, Z, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p, X, Y;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol,
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[Y]], but since C is symmetric, we can do that squared indirectly
    b = zero(a)
    get_jacobian!(M, b, o, p, Y)
    # Compute C^TCb = C^2 b (inplace of b)
    b .= ρ_prime .* (I - operator_scaling * (a * a'))^2 * b
    # apply the adjoint
    add_adjoint_jacobian!(M, Z, o, p, b)
    return Z
end
# Componentwise
function _add_hessian!(
        M::AbstractManifold, Z, o::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p, X, Y;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol,
    )
    # per single component a for-loop similar to the one for the blocks
    r = cr.robustifier
    b = zero(value_cache)
    get_jacobian!(M, b, o, p, X)
    # Componentwise a few things decouple
    for (i, ai) in enumerate(value_cache)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, threshold, mode)
        # get the “Jacobian” of the ith component, i.e. its
        # Compute C^TCa = C^2 a (inplace of a)
        b[i] = ρ_prime * (1 - operator_scaling * ai_sq)^2 * b[i]
    end
    # apply the adjoint
    add_adjoint_jacobian!(M, Z, o, p, b)
    return Z
end

## The name is not optimal but it is merely something internal with a small safeguard
"""
    default_lm_lin_solve!(sk, JJ::AbstractMatrix, grad_f_c)

Solve the linear system of equations of the normal equations `JJ \\ grad_f_c` where JJ is a symmetric positive
definite matrix and save the result to `sk`. In case of numerical errors the
`PosDefException` is caught and the default symmetric solver `(Symmetric(JJ) \\ grad_f_c)`
is used.

The function is intended to be used with [`LevenbergMarquardt`](@ref).
"""
function default_lm_lin_solve!(sk, JJ::AbstractMatrix, grad_f_c)
    try
        ldiv!(sk, cholesky(Symmetric(JJ)), grad_f_c)
    catch e
        e isa PosDefException ? (sk .= Symmetric(JJ) \ grad_f_c) : rethrow()
    end
    return sk
end

"""
    CoordinatesNormalSystemState <: AbstractManoptSolverState

A solver state indicating that we solve the [`LevenbergMarquardtLinearSurrogateObjective`](@ref)
using a linear system in coordinates of the tangent space at the current iterate

## Fields

* `A` an ``n×n`` matrix to store the normal equations linear from [`get_linear_operator`](@ref) in coordinates, where `n` is the number of coordinates
* `b` a ``n`` vector storing the right hand side of the normal equations in coordinates
* `basis::`[`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`)
* `linsolve` a functor `(A,b) -> c` to solve the linear system or `(c, A, b) -> c` depending on the evaluation type specified in `solve!`

## Constructor
    CoordinatesNormalSystemState(
        M::AbstractManifold, p = rand(M);
        evaluation = InplaceEvaluation(),
        linsolve = default_lm_lin_solve!,
        basis = DefaultOrthonormalBasis(),
        A = nothing
    )

Construct the state, where not providing a memory for `A` uses the `eltype` of `p` to
determine the element type of the matrix to store.
"""
mutable struct CoordinatesNormalSystemState{E <: AbstractEvaluationType, F, TA <: AbstractMatrix, TB <: AbstractVector, TBA <: AbstractBasis} <: AbstractManoptSolverState
    A::TA
    b::TB
    basis::TBA
    c::TB
    linsolve!!::F
end
function CoordinatesNormalSystemState(
        M::AbstractManifold, p = rand(M);
        evaluation::E = InplaceEvaluation(), linsolve::F = default_lm_lin_solve!,
        basis::B = DefaultOrthonormalBasis(), A = nothing
    ) where {E <: AbstractEvaluationType, F, B <: AbstractBasis}
    n = number_of_coordinates(M, basis)
    c = zeros(number_eltype(p), n)
    if isnothing(A)
        A = zeros(eltype(c), n, n)
    end
    b = zeros(eltype(c), n)
    return CoordinatesNormalSystemState{E, F, typeof(A), typeof(b), B}(A, b, basis, c, linsolve)
end

# The objective here should be a LevenbergMarquardtLinearSurrogateObjective, but might be decorated as well, so for now lets not type it (yet?)
function solve!(dmp::DefaultManoptProblem{<:TangentSpace, <:NormalEquationsObjective}, cnss::CoordinatesNormalSystemState{E}) where {E <: AbstractEvaluationType}
    # Update A and b
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    neo = get_objective(dmp)
    get_linear_operator!(M, cnss.A, neo, p, cnss.basis)
    get_vector_field!(M, cnss.b, neo, p, cnss.basis)
    cnss.b .*= -1
    (E === AllocatingEvaluation) ? (cnss.c = cnss.linsolve!!(cnss.A, cnss.b)) : (cnss.linsolve!!(cnss.c, cnss.A, cnss.b))
    return cnss
end

"""
    get_linear_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X)
    get_linear_operator!(M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X)

Evaluate the linear operator ``$(_tex(:Cal, "L"))`` corresponding to the Levenberg-Marquardt surrogate objective, i.e.,

```math
$(_tex(:Cal, "L"))(X) = C J_F(p)[X] $(_tex(:bigr))],
```

with

```math
C = $(_tex(:sqrt, "ρ'(p)"))(I-αP), $(_tex(:qquad)) P = $(_tex(:frac, "F(p)F(p)^" * _tex(:rm, "T"), _tex(:norm, "F(p)"; index = "2") * "^2")),
```

where ``α = 1 - $(_tex(:sqrt, "1 + 2 $(_tex(:frac, "ρ''(p)", "ρ'(p)"))$(_tex(:norm, "F(p)"; index = "2"))^2"))``.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`ManifoldNonlinearLeastSquaresObjective`](@ref) and summed up.

This can be computed in-place of `y`.

See also [`get_vector_field`](@ref) for evaluating the corresponding vector field
"""
function get_linear_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    nlso = get_objective(lmsco)
    n = residuals_count(nlso)
    y = zeros(eltype(p), n)
    return get_linear_operator!(M, y, lmsco, p, X)
end
function get_linear_operator!(
        M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    nlso = get_objective(lmsco)
    # Init to zero
    fill!(y, 0)
    start = 0
    Y_cache = zero_vector(M, p)
    c_cache = allocate_result(M, get_coordinates, p, X, DefaultOrthonormalBasis())
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        _get_linear_operator!(
            M, view(y, (start + 1):(start + len)), o, r, p, X, value_cache;
            threshold = lmsco.threshold, mode = lmsco.mode, Y_cache = Y_cache, c_cache = c_cache,
        )
        start += len
    end
    return y
end
# for a single block – the actual formula
function _get_linear_operator!(
        M::AbstractManifold, y, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p, X,
        value_cache = get_value(M, o, p); threshold::Real, mode::Symbol, Y_cache, c_cache
    )
    F_sq = sum(abs2, value_cache)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    get_jacobian!(M, y, o, p, X; Y_cache = Y_cache, c_cache = c_cache)
    # Compute C y
    α = sqrt(ρ_prime)
    t = dot(value_cache, y)
    @. y = α * (y - operator_scaling * t * value_cache)
    return y
end
# Componenwise: Decouple
function _get_linear_operator!(
        M::AbstractManifold, y, o::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p, X,
        value_cache = get_value(M, o, p); threshold::Real, mode::Symbol, Y_cache, c_cache
    )
    a = value_cache
    r = cr.robustifier
    get_jacobian!(M, y, o, p, X)
    for (i, ai) in enumerate(value_cache)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, threshold, mode)
        # get the “Jacobian” of the ith component, i.e. y[i]
        # C is justr a diagonal matrix here
        y[i] = sqrt(ρ_prime) * (1 - operator_scaling * ai_sq) * y[i]
    end
    return y
end

"""
    get_vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    get_vector_field!(M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)

Compute the vector field ``y`` corresponding to the Levenberg-Marquardt surrogate objective, i.e.,

```math
y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p)
```

where the scaling uses ``α = 1 - $(_tex(:sqrt, "1 + 2 $(_tex(:frac, "ρ''(p)", "ρ'(p)"))$(_tex(:norm, "F(p)"; index = "2"))^2"))``

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`ManifoldNonlinearLeastSquaresObjective`](@ref) and summed up.

See also
* [`get_LevenbergMarquardt_scaling`](@ref) for details on the scaling factor
* [`get_linear_operator`](@ref) for evaluating the corresponding linear operator of the linear system
"""
function get_vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    n = residuals_count(nlso)
    y = zeros(number_eltype(p), n)
    return get_vector_field!(M, y, lmsco, p)
end
function get_vector_field!(
        M::AbstractManifold, y, lmsco::AbstractLevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    # Init to zero
    fill!(y, 0)
    start = 0
    # For every block
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        _get_vector_field!(M, view(y, (start + 1):(start + length(o))), o, r, p; threshold = lmsco.threshold, mode = lmsco.mode)
        start += length(o)
    end
    return y
end
# for a single block – the actual formula
function _get_vector_field!(
        M::AbstractManifold, y, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p;
        threshold::Real, mode::Symbol,
    )
    get_value!(M, y, o, p) # evaluate residuals F(p)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, _ = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # Compute y = sqrt(ρ(p)) / (1-α) * F(p)
    y .*= residual_scaling
    return y
end
# Componentwise, it decouples, C is diagonal
function _get_vector_field!(
        M::AbstractManifold, y, o::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p;
        threshold::Real, mode::Symbol,
    )
    get_value!(M, y, o, p) # evaluate residuals F(p)
    r = cr.robustifier
    for (i, ai) in enumerate(y)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        residual_scaling, _ = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, threshold, mode)
        # Compute y = sqrt(ρ(p)) / (1-α) * F(p)
        y[i] *= residual_scaling
    end
    return y
end

#
#
# For the Normal Equations Objective

"""
    get_cost(TpM::TangentSpace, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, X)

Compute the surrogate cost when solving its normal equation, see also
[`get_cost(::AbstractManifold, ::LevenbergMarquardtLinearSurrogateObjective, p, X)`](@ref),
[`get_linear_operator`](@ref), and [`get_vector_field`](@ref) for more details.
"""
function get_cost(
        TpM::TangentSpace, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, X
    ) where {E <: AbstractEvaluationType}
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_cost(M, neo.objective, p, X)
end
# Maybe a bit too precise, but in this case we get a coefficient vector and we want a tangent vector
function get_solver_result(
        dmp::DefaultManoptProblem{<:TangentSpace, <:NormalEquationsObjective{<:AbstractEvaluationType, <:LevenbergMarquardtLinearSurrogateObjective}},
        cnss::CoordinatesNormalSystemState
    )
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_vector(M, p, cnss.c, cnss.basis)
end

"""
    get_normal_linear_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X; threshold = lmsco.threshold, mode = lmsco.mode, penalty = lmsco.penalty)
    get_normal_linear_operator!(M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X; threshold = lmsco.threshold, mode = lmsco.mode, penalty = lmsco.penalty)
    get_normal_linear_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p[, c], B::AbstractBasis; threshold = lmsco.threshold, mode = lmsco.mode, penalty = lmsco.penalty)
    get_normal_linear_operator!(M::AbstractManifold, [A | b], lmsco::LevenbergMarquardtLinearSurrogateObjective, p[, c], B::AbstractBasis; threshold = lmsco.threshold, mode = lmsco.mode, penalty = lmsco.penalty)

Compute the linear operator ``$(_tex(:Cal, "A"))`` corresponding to the optimality conditions of the
modified Levenberg-Marquardt surrogate objective, i.e. the normal conditions

```math
$(_tex(:Cal, "A"))(X) = $(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))(X) + λX
= J_F^*(p)$(_tex(:bigl))[ C^T C J_F(p)[X] $(_tex(:bigr))] + λX,
```

where ``λ = ```penalty` is a damping parameter and with
``α = 1 - $(_tex(:sqrt, "1 + 2 $(_tex(:frac, "ρ''(p)", "ρ'(p)"))$(_tex(:norm, "F(p)"; index = "2"))^2"))``
we have

```math
C = $(_tex(:sqrt, "ρ'(p)"))(I-αP), $(_tex(:qquad)) P = $(_tex(:frac, "F(p)F(p)^" * _tex(:rm, "T"), _tex(:norm, "F(p)"; index = "2") * "^2")),
```

See [`get_LevenbergMarquardt_scaling`](@ref) for details on the scaling and `α`.
and [`get_jacobian`](@ref) and [`get_adjoint_jacobian`](@ref) concerning ``J_F`` and ``J_F^*``, respectively.

There are three variants to use this function to use the corresponding linear operator
* if you provide a tangent vector `X`, then the linear operator is evaluated at `X`, the corresponding gradient `Y` is returned
* if you provide `X` in coordinates `c` with respect to a basis `B` the linear operator is evaluated and the coordinates `b` of the result are returned
* if you provide (just) a basis `B` of the tangent space, then the matrix `A` of the linear operator represented in this basis is returned. The relation to the second case is that ``b = Ac``.

See also [`get_normal_vector_field`](@ref) for evaluating the corresponding vector field.
"""
function get_normal_linear_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        penalty::Real = lmsco.penalty,
    )
    Y = zero_vector(M, p)
    return get_normal_linear_operator!(M, Y, lmsco, p, X; penalty = penalty)
end
function get_normal_linear_operator!(
        M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        penalty::Real = lmsco.penalty,
    )
    nlso = get_objective(lmsco)
    # For every block
    zero_vector!(M, Y, p)
    Y_cache = zero_vector(M, p)
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        add_normal_linear_operator!(M, Y, o, r, p, X; threshold = lmsco.threshold, mode = lmsco.mode, value_cache = value_cache, Y_cache = Y_cache)
        start += len
    end
    # Finally add the damping term
    (penalty != 0) && (Y .+= penalty .* X)
    return Y
end
# for a single block – the actual formula - but never with penalty
function add_normal_linear_operator!(
        M::AbstractManifold, Y, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol, Y_cache = zero_vector(M, p)
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared indirectly
    b = zero(a)
    get_jacobian!(M, b, o, p, X; Y_cache = Y_cache)
    # Compute C^TCb = C^2 b (inplace of a)

    # The code below is mathematically equivalent to the following, but avoids allocating
    # the outer product a * a' and the matrix-vector product (a * a') * b
    # b .= ρ_prime .* (I - operator_scaling * (a * a'))^2 * b
    t = dot(a, b)
    aa = dot(a, a)
    coef = operator_scaling * t * (operator_scaling * aa - 2)

    @. b = ρ_prime * (b + coef * a)

    # Now apply the adjoint
    add_adjoint_jacobian!(M, Y, o, p, b; Y_cache = Y_cache)
    # penalty is added once after summing up all blocks, so we do not add it here
    return Y
end
# Componentwise: A few things decouple
function add_normal_linear_operator!(
        M::AbstractManifold, Y, o::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol, Y_cache = nothing,
    )
    b = zero(value_cache)
    get_jacobian!(M, b, o, p, X)
    r = cr.robustifier
    for (i, ai) in enumerate(value_cache)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, threshold, mode)
        # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared indirectly
        # Compute C^TCb = C^2 b (inplace of a)
        b[i] = ρ_prime * (1 - operator_scaling * ai_sq)^2 * b[i]
    end
    # Now apply the adjoint
    zero_vector!(M, Y, p)
    add_adjoint_jacobian!(M, Y, o, p, b)
    return Y
end
#
# Basis case: (a) including coordinates
function get_normal_linear_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, c, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    d = zero(c)
    return get_normal_linear_operator!(M, d, lmsco, p, c, B; penalty = penalty)
end
function get_normal_linear_operator!(
        M::AbstractManifold, d, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, c, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(d, 0)
    e = zero(d)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        get_normal_linear_operator!(M, e, o, r, p, c, B; threshold = lmsco.threshold, mode = lmsco.mode)
        d .+= e
    end
    # Finally add the damping term
    (penalty != 0) && (d .+= penalty * c)
    return d
end
function get_normal_linear_operator!(M::AbstractManifold, d, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p, c, B::AbstractBasis; kwargs...)
    # Lazy fallback: Create matrix and perform Ac inplace of d
    dA = number_of_coordinates(M, B)
    A = zeros(eltype(d), dA, dA)
    A = add_normal_linear_operator!(M, A, o, r, p, B; kwargs...)
    d .= A * c
    return d
end
#
# Basis case: (b) no coordinates -> compute a matrix representation
function get_normal_linear_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    d = number_of_coordinates(M, B)
    A = zeros(number_eltype(p), d, d)
    return get_normal_linear_operator!(M, A, lmsco, p, B; penalty = penalty)
end
function get_normal_linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(A, 0)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        add_normal_linear_operator!(M, A, o, r, p, B; threshold = lmsco.threshold, mode = lmsco.mode)
    end
    # Finally add the damping term
    (penalty != 0) && (LinearAlgebra.diagview(A) .+= penalty)
    return A
end
"""
    add_normal_linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractFirstOrderVectorFunction,
        r::AbstractRobustifierFunction, p, basis::AbstractBasis;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol
    )

Add the contribution of a single block (vectorial function with its robustifier) to
the linear normal operator, i.e. compute ``A += J_F^*(p)[C^T C J_F(p)[X]]`` in-place of `A`
for the given block.
See [`get_normal_linear_operator`](@ref) for details
"""
function add_normal_linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractFirstOrderVectorFunction,
        r::AbstractRobustifierFunction, p, basis::AbstractBasis;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # to Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared indirectly
    # (a) J_F is n-by-d so we have to allocate – where could we maybe store something like that and pass it down?
    JF = get_jacobian(M, o, p; basis = basis)
    # (I - s*a*a')^2 = I + (-2s + s^2*||a||^2) * a*a'
    # so JF' * (ρ' * (I - s*a*a')^2) * JF
    #   = ρ' * (JF'JF) + ρ' * (-2s + s^2*||a||^2) * (JF'a) * (JF'a)'
    rank1_scaling = ρ_prime * (-2 * operator_scaling + operator_scaling^2 * F_sq)
    mul!(A, JF', JF, ρ_prime, true)
    if !iszero(rank1_scaling)
        JFa = JF' * a
        mul!(A, JFa, JFa', rank1_scaling, true)
    end
    # damping term is added once after summing up all blocks, so we do not add it here
    return A
end
# For the componentwise variant, the C^TC turns into a diagonal matrix
function add_normal_linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractFirstOrderVectorFunction,
        cr::ComponentwiseRobustifierFunction, p, basis::AbstractBasis;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol
    )
    a = value_cache # evaluate residuals F(p)
    b = zero(a)
    r = cr.robustifier
    for (i, ai) in enumerate(a)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, threshold, mode)
        # to Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared indirectly
        # (a) J_F is n-by-d so we have to allocate – where could we maybe store something like that and pass it down?
        b[i] = ρ_prime * (1 - operator_scaling * ai_sq)^2
    end
    JF = get_jacobian(M, o, p; basis = basis)
    # compute A' C^TC A (C^TC = C^2 here) inplace of A
    A .+= JF' * Diagonal(b) * JF
    # damping term is added once after summing up all blocks, so we do not add it here
    return A
end

_doc_get_normal_vector_field = """
    get_normal_vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    get_normal_vector_field!(M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    get_normal_vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis)
    get_normal_vector_field!(M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis)

Compute the normal linear operator tangent vector ``X`` corresponding to the normal equations (optimality conditions) of the
Levenberg-Marquardt surrogate objective, i.e.,

```math
X = J_F^*(p)[ C^T y], $(_tex(:quad)) y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p).
```

If you provide an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `B` ``=$(_tex(:set, "Z_1,…,Z_d"))`` additionally,
the result will be given in coordinates `c`, i.e. such that ``X = $(_tex(:sum, "i=1", "d")) c_iZ_i``.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`ManifoldNonlinearLeastSquaresObjective`](@ref) and summed up.
See also [`get_normal_linear_operator`](@ref) for evaluating the corresponding linear operator of the (normal) linear system,
and [`get_LevenbergMarquardt_scaling`](@ref) for details on the scaling and computation of ``C``.
"""

_doc_add_normal_vector_field = """
    add_normal_vector_field!(M::AbstractManifold, X, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p)
    add_normal_vector_field!(M::AbstractManifold, c, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p, B::AbstractBasis)

Add the contribution of `o` / `r` to the normal linear operator tangent vector in `X` or `c`.
See [`get_normal_vector_field`](@ref) for the mathematical details.
Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`ManifoldNonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`get_normal_linear_operator`](@ref) for evaluating the corresponding linear operator of the (normal) linear system,
and [`get_LevenbergMarquardt_scaling`](@ref) for details on the scaling and computation of ``C``.
"""
@doc "$(_doc_get_normal_vector_field)"
function get_normal_vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    X = zero_vector(M, p)
    return get_normal_vector_field!(M, X, lmsco, p)
end

@doc "$(_doc_get_normal_vector_field)"
function get_normal_vector_field!(
        M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    # For every block
    zero_vector!(M, X, p)
    Z = copy(M, p, X)
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    Y_cache = zero_vector(M, p)
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        _get_normal_vector_field!(
            M, Z, o, r, p;
            threshold = lmsco.threshold, mode = lmsco.mode, value_cache = value_cache, Y_cache = Y_cache,
        )
        start += len
        X .+= Z
    end
    return X
end
# for a single block – the actual formula
function _get_normal_vector_field!(
        M::AbstractManifold, X, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol, Y_cache = zero_vector(M, p),
    )
    y = copy(value_cache)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # Compute y = ( ρ'(p) / (1-α)) F(p)
    γ = residual_scaling * sqrt(ρ_prime) * (1 - operator_scaling * dot(y, y))
    @. y = γ * y
    # and apply the adjoint, i.e. compute J_F^*(p)[C^T y]
    zero_vector!(M, X, p)
    add_adjoint_jacobian!(M, X, o, p, y; Y_cache = Y_cache)
    return X
end
# Componenwise C again reduces to a diagonal
function _get_normal_vector_field!(
        M::AbstractManifold, X, o::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol, Y_cache = nothing,
    )
    y = copy(value_cache)
    r = cr.robustifier
    for (i, ai) in enumerate(y)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, threshold, mode)
        # Compute y = ( ρ'(p) / (1-α)) F(p)
        y[i] = residual_scaling * sqrt(ρ_prime) * (1 - operator_scaling * ai_sq) * y[i]
    end
    # and apply the adjoint, i.e. compute J_F^*(p)[C^T y]
    zero_vector!(M, X, p)
    add_adjoint_jacobian!(M, X, o, p, y)
    return X
end

#
# (b) in a basis
@doc "$(_doc_get_normal_vector_field)"
function get_normal_vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis,
    )
    c = get_coordinates(M, p, zero_vector(M, p), B)
    return get_normal_vector_field!(M, c, lmsco, p, B)
end

@doc "$(_doc_get_normal_vector_field)"
function get_normal_vector_field!(
        M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis,
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(c, 0)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        add_normal_vector_field!(M, c, o, r, p, B; threshold = lmsco.threshold, mode = lmsco.mode)
    end
    return c
end

# for a single block – the actual formula
@doc "$(_doc_add_normal_vector_field)"
function add_normal_vector_field!(
        M::AbstractManifold, c, o::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p, B::AbstractBasis;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol,
    )
    y = copy(value_cache) # evaluate residuals F(p)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # Compute y = ρ'(p) / (1-α)) F(p) and ...
    y .= residual_scaling .* sqrt(ρ_prime) * (I - operator_scaling * (y * y')) * y
    # ...apply the adjoint, i.e. compute  J_F^*(p)[C^T y] (inplace of y)
    add_adjoint_jacobian!(M, c, o, p, y, B)
    return c
end
# Compponentwise: decouple, C is a diagonalmatrix
function add_normal_vector_field!(
        M::AbstractManifold, c, o::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p, B::AbstractBasis;
        value_cache = get_value(M, o, p), threshold::Real, mode::Symbol, Y_cache = nothing,
    )
    y = copy(value_cache) # evaluate residuals F(p)
    r = cr.robustifier
    for (i, ai) in enumerate(y)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, threshold, mode)
        # Compute y = ρ'(p) / (1-α)) F(p) and ...
        y[i] = residual_scaling * sqrt(ρ_prime) * (1 - operator_scaling * ai_sq) * ai
    end
    # ...apply the adjoint, i.e. compute  J_F^*(p)[C^T y] (inplace of y)
    add_adjoint_jacobian!(M, c, o, p, y, B)
    return c
end

#
# The Symmetric Linear System (e.g. in CGRes) for the LM Surrogate is its normal equations and vector.
# (a) a vector X or a basis B
_doc_linOp_NEO = """
    get_linear_operator(M::AbstractManifold, neo::NormalEquationsObjective, p, X)
    get_linear_operator(M::AbstractManifold, neo::NormalEquationsObjective, p, c, B)
    get_linear_operator(M::AbstractManifold, neo::NormalEquationsObjective, p, B)
    get_linear_operator!(M::AbstractManifold, Y, neo::NormalEquationsObjective, p, X)
    get_linear_operator!(M::AbstractManifold, b, neo::NormalEquationsObjective, p, c, B)
    get_linear_operator!(M::AbstractManifold, A, neo::NormalEquationsObjective, p, B)

    Evaluate the linear operator related to the normal equations of the [`LevenbergMarquardtLinearSurrogateObjective`](@ref),
    see [`get_normal_linear_operator`](@ref) for details.
"""

@doc "$(_doc_linOp_NEO)"
function get_linear_operator(
        M::AbstractManifold, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, XB
    ) where {E <: AbstractEvaluationType}
    return get_normal_linear_operator(M, neo.objective, p, XB)
end
@doc "$(_doc_linOp_NEO)"
function get_linear_operator!(
        M::AbstractManifold, YA, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, XB
    ) where {E <: AbstractEvaluationType}
    return get_normal_linear_operator!(M, YA, neo.objective, p, XB)
end
# (b) coefficients in a basis
function get_linear_operator(
        M::AbstractManifold, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, c, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return get_normal_linear_operator(M, neo.objective, p, c, B)
end
function get_linear_operator!(
        M::AbstractManifold, Y, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, c, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return get_normal_linear_operator!(M, Y, neo.objective, p, c, B)
end

_doc_vecField_NEO = """
    get_vector_field(M::AbstractManifold, neo::NormalEquationsObjective, p)
    get_vector_field(M::AbstractManifold, neo::NormalEquationsObjective, p, B)
    get_vector_field!(M::AbstractManifold, Y, neo::NormalEquationsObjective, p)
    get_vector_field!(M::AbstractManifold, c, neo::NormalEquationsObjective, p, B)

    Evaluate the vector field related to the normal equations of the [`LevenbergMarquardtLinearSurrogateObjective`](@ref),
    see [`get_normal_vector_field`](@ref) for details,
    but note that for the [`NormalEquationsObjective`](@ref) the format is slightly different:
    For the variant with `_normal` the result is similar to the surrogate, namely we have
    ``$(_tex(:Cal, "L"))(X) + y`` for the surrogate and hence also the same form ``$(_tex(:Cal, "N"))(X) + z``,
    which has to be set to zero to find ``X``.

    For the objective here we consider ````$(_tex(:Cal, "N"))(X) = z'``, i.e. the `get_vector_field` `z' = -z``
    differs by a sign.
"""

# RHS as a tangent vector
@doc "$(_doc_vecField_NEO)"
function get_vector_field(
        M::AbstractManifold, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, p
    ) where {E <: AbstractEvaluationType}
    return -get_normal_vector_field(M, neo.objective, p)
end
@doc "$(_doc_vecField_NEO)"
function get_vector_field!(
        M::AbstractManifold, Y, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, p
    ) where {E <: AbstractEvaluationType}
    get_normal_vector_field!(M, Y, neo.objective, p)
    Y .*= -1
    return Y
end
# RHS in coordinates
function get_vector_field(
        M::AbstractManifold, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return -get_normal_vector_field(M, neo.objective, p, B)
end
function get_vector_field!(
        M::AbstractManifold, c, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    get_normal_vector_field!(M, c, neo.objective, p, B)
    c .*= -1
    return c
end
