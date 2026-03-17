@doc """
    NonlinearLeastSquaresObjective{E<:AbstractEvaluationType} <: AbstractManifoldObjective{E}

An objective to model the robustified nonlinear least squares problem

$(_problem(:NonLinearLeastSquares))

# Fields

* `objective`: a vector of [`AbstractVectorGradientFunction`](@ref)`{E}`s, one for each
  block component cost function ``F_i``, which might internally also be a vector of component costs ``(F_i)_j``,
  as well as their Jacobian ``J_{F_i}`` or a vector of gradients ``$(_tex(:grad)) (F_i)_j``
  depending on the specified [`AbstractVectorialType`](@ref)s.
* `robustifier`: a vector of [`AbstractRobustifierFunction`](@ref)`s`, one for each
  block component cost function ``F_i``.

# Constructors

    NonlinearLeastSquaresObjective(f, jacobian, range_dimension::Integer, robustifier=IdentityRobustifier(); kwargs...)
    NonlinearLeastSquaresObjective(vf::AbstractVectorGradientFunction, robustifier::AbstractRobustifierFunction=IdentityRobustifier())
    NonlinearLeastSquaresObjective(fs::Vector{<:AbstractVectorGradientFunction}, robustifiers::Vector{<:AbstractRobustifierFunction}=fill(IdentityRobustifier(), length(fs)))

# Arguments

* `f` the vectorial cost function ``f: $(_math(:Manifold)) → ℝ^m``
* `jacobian` the Jacobian, might also be a vector of gradients of the component functions of `f`
* `range_dimension::Integer` the number of dimensions `m` the function `f` maps into

These three can also be passed as a [`AbstractVectorGradientFunction`](@ref) `vf` already.

* `robustifier` the robustifier function(s) to use, by default the [`IdentityRobustifier`](@ref) (for each component),
    which corresponds to the classical nonlinear least squares problem.
    For a single [`AbstractVectorGradientFunction`](@ref) `vf` this is always treated/wrapped into a [`ComponentwiseRobustifierFunction`](@ref).
    To actually have one robustifier on the whole norm (squared) of `vf` use the third signature and provide `[vf,]` and  `[robustifier,]` there.

# Keyword arguments

$(_kwargs(:evaluation))

As well as for the first variant of having a single block

* `function_type::`[`AbstractVectorialType`](@ref)`=`[`FunctionVectorialType`](@ref)`()`: specify
  the format the residuals are given in. By default a function returning a vector.
* `jacobian_tangent_basis::AbstractBasis=DefaultOrthonormalBasis()`; shortcut to specify
  the basis the Jacobian matrix is build with.
* `jacobian_type::`[`AbstractVectorialType`](@ref)`=`[`CoefficientVectorialType`](@ref)`(jacobian_tangent_basis)`:
  specify the format the Jacobian is given in. By default a matrix of the differential with
  respect to a certain basis of the tangent space.

# See also

[`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref)
"""
struct NonlinearLeastSquaresObjective{
        E <: AbstractEvaluationType,
        VF <: AbstractVectorGradientFunction{E},
        RF <: AbstractRobustifierFunction,
        TVC <: AbstractVector,
    } <: AbstractManifoldFirstOrderObjective{E, Vector{VF}}
    objective::Vector{VF}
    robustifier::Vector{RF}
    value_cache::TVC
    # block components case constructor
    function NonlinearLeastSquaresObjective(
            fs::Vector{VF},
            robustifiers::Vector{RV} = fill(IdentityRobustifier(), length(fs)),
            value_cache::TVC = zeros(sum(length(f) for f in fs)),
        ) where {E <: AbstractEvaluationType, VF <: AbstractVectorGradientFunction{E}, RV <: AbstractRobustifierFunction, TVC <: AbstractVector}
        # we need to check that the lengths match
        (length(fs) != length(robustifiers)) && throw(
            ArgumentError(
                "Number of functions ($(length(fs))) does not match number of robustifiers ($(length(robustifiers)))",
            ),
        )
        return new{E, VF, RV, TVC}(fs, robustifiers, value_cache)
    end
    # single component case constructor
    function NonlinearLeastSquaresObjective(
            f::F,
            robustifier::R = IdentityRobustifier(),
            value_cache::TVC = zeros(length(f)),
        ) where {E <: AbstractEvaluationType, F <: AbstractVectorGradientFunction{E}, R <: AbstractRobustifierFunction, TVC <: AbstractVector}
        cr = ComponentwiseRobustifierFunction(robustifier)
        return new{E, F, typeof(cr), TVC}([f], [cr], value_cache)
    end
end

# the old single function constructor – TODO: carefully document that this is just one special case that has a nicer shortcut and calls the vectorial one
function NonlinearLeastSquaresObjective(
        f,
        jacobian,
        range_dimension::Integer,
        robustifier::AbstractRobustifierFunction = IdentityRobustifier();
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        jacobian_tangent_basis::AbstractBasis = DefaultOrthonormalBasis(),
        jacobian_type::AbstractVectorialType = CoefficientVectorialType(jacobian_tangent_basis),
        function_type::AbstractVectorialType = FunctionVectorialType(),
    )
    vgf = VectorGradientFunction(
        f,
        jacobian,
        range_dimension;
        evaluation = evaluation,
        jacobian_type = jacobian_type,
        function_type = function_type,
    )
    return NonlinearLeastSquaresObjective(vgf, robustifier)
end

"""
    residuals_count(nlso::NonlinearLeastSquaresObjective)

Return the total number of residuals in [`NonlinearLeastSquaresObjective`](@ref) `nlso`.
"""
function residuals_count(nlso::NonlinearLeastSquaresObjective)
    return sum(length(o) for o in nlso.objective)
end

# Cost
function get_cost(
        M::AbstractManifold,
        nlso::NonlinearLeastSquaresObjective,
        p;
        kwargs...,
    )
    v = 0.0
    start = 0
    get_residuals!(M, nlso.value_cache, nlso, p)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        value_cache = view(nlso.value_cache, (start + 1):(start + len))
        v += _get_cost(M, o, r, p; value_cache = value_cache)
        start += len
    end
    v /= 2
    return v
end
function _get_cost(
        M, vgf::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
        value_cache = get_value(M, vgf, p)
    )
    vi = sum(abs2, value_cache)
    (a, _, _) = get_robustifier_values(r, vi)
    return a
end
function _get_cost(
        M, vgf::AbstractVectorGradientFunction, cr::ComponentwiseRobustifierFunction, p;
        value_cache = get_value(M, vgf, p)
    )
    v = abs2.(value_cache)
    # componentwise robustify
    (a, _, _) = get_robustifier_values(cr, v)
    return sum(a)
end
#

_doc_get_gradient_nlso = """
    get_gradient(M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...)
    get_gradient!(M::AbstractManifold, X, nlso::NonlinearLeastSquaresObjective, p; kwargs...)

Compute the gradient for the [`NonlinearLeastSquaresObjective`](@ref) `nlso` at the point ``p ∈ M``,
i.e.

```math
$(_tex(:grad)) f(p) = $(_tex(:sum, "i=1", "m")) ρ'_i$(_tex(:bigl))($(_tex(:norm, "F_i(p)"; index = "2"))^2$(_tex(:bigr)))
$(_tex(:sum, "j=1", "n_i")) f_{i,j}(p) $(_tex(:grad)) f_{i,j}(p)
```

where ``F_i(p) ∈ ℝ^{n_i}`` is the vector of residuals for the `i`-th block component cost function
and ``f_{i,j}(p)`` its `j`-th component function.

# Keyword arguments
* `value_cache=nothing` : if provided, this vector is used to store the residuals ``F(p)``
  internally to avoid recomputations.
"""
@doc "$(_doc_get_gradient_nlso)"
function get_gradient(
        M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...,
    )
    X = zero_vector(M, p)
    return get_gradient!(M, X, nlso, p; kwargs...)
end

# @doc "$(_doc_get_gradient_nlso)"
function get_gradient!(
        M::AbstractManifold, X, nlso::NonlinearLeastSquaresObjective, p;
        value_cache = nothing, jacobian_cache = fill(nothing, length(nlso.objective)),
    )
    zero_vector!(M, X, p)
    start = 0
    Y = copy(M, p, X)
    for (o, r, jb) in zip(nlso.objective, nlso.robustifier, jacobian_cache) # for every block
        len = length(o)
        Fi = isnothing(value_cache) ? get_value(M, o, p) : view(value_cache, (start + 1):(start + len))
        _add_gradient!(M, X, o, r, p; value_cache = Fi, jacobian_cache = jb)
        start += len
    end
    return X
end
function _add_gradient!(
        M, X, vgf::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
        value_cache = get_value(M, vgf, p), jacobian_cache = nothing
    )
    # get gradients for every component
    len = length(vgf)

    # compute robustifier derivative
    (_, b, _) = get_robustifier_values(r, sum(abs2, value_cache))
    if isnothing(jacobian_cache)
        Y = allocate(M, X)
        for j in 1:len
            get_gradient!(M, Y, vgf, p, j) # gradient of f_{i,j}
            X .+= (b * value_cache[j]) .* Y
        end
    else
        Jc = jacobian_cache' * value_cache
        Jc .*= b
        add_vector!(M, X, p, Jc, vgf.jacobian_type.basis)
    end
    return X
end
function _get_gradient!(
        M, X, vgf::AbstractVectorGradientFunction, cr::ComponentwiseRobustifierFunction, p;
        value_cache = get_value(M, vgf, p), jacobian_cache = nothing,
    )
    # get gradients for every component
    len = length(vgf)
    r = cr.robustifier
    zero_vector!(M, X, p)
    Y = copy(M, p, X)
    for j in 1:len
        get_gradient!(M, Y, vgf, p, j) # gradient of f_{i,j}
        (_, b, _) = get_robustifier_values(r, abs(value_cache[j])^2)
        # compute robustifier derivative
        X .+= (b * value_cache[j]) .* Y
    end
    return X
end

# --- Residuals
_doc_get_residuals_nlso = """
    get_residuals(M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p)
    get_residuals!(M::AbstractManifold, v, nlso::NonlinearLeastSquaresObjective, p)

Compute the vector of residuals ``F(p) ∈ ℝ^n``, ``n = $(_tex(:sum, "1", "m")) n_i``.
In other words this is the concatenation of the residual vectors ``F_i(p)``, ``i=1,…,m``
of the components of the the [`NonlinearLeastSquaresObjective`](@ref) `nlso`
at the current point ``p`` on `M`.

This can be computed in-place of `v`.
"""

@doc "$(_doc_get_residuals_nlso)"
function get_residuals(
        M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...
    )
    v = zeros(residuals_count(nlso))
    return get_residuals!(M, v, nlso, p; kwargs...)
end

@doc "$(_doc_get_residuals_nlso)"
function get_residuals!(
        M::AbstractManifold, v, nlso::NonlinearLeastSquaresObjective, p; kwargs...,
    )
    start = 0
    for o in nlso.objective # for every block
        len = length(o)
        view_v = view(v, (start + 1):(start + len))
        get_value!(M, view_v, o, p)
        start += len
    end
    return v
end

#
#
# The solver state

# TODO: Update keywords in docs
@doc """
    LevenbergMarquardtState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields

A default value is given in brackets if a parameter can be left out in initialization.

$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:retraction_method))
* `residual_values`:      value of ``F`` calculated in the solver setup or the previous iteration
$(_fields(:stopping_criterion; name = "stop"))
* `X`:             the current gradient of ``F``
* `direction`:     the current search direction, which is the solution of the linearized
  subproblem in each iteration.
* `jacobian_f`:          the current Jacobian of ``F``. Either a matrix in coordinates or
  `nothing` if the Jacobian is used only as a linear operator.
* `step_vector`:          the tangent vector at `x` that is used to move to the next point
* `last_stepsize`:        length of `step_vector`
* `η`:                    Scaling factor for the sufficient cost decrease threshold required
  to accept new proposal points. Allowed range: `0 < η < 1`.
* `damping_term`:         current value of the damping term
* `damping_term_min`:     initial (and also minimal) value of the damping term
* `β`:                    parameter by which the damping term is multiplied when the current
  new point is rejected
* `expect_zero_residual`: if true, the algorithm expects that the value of
  the residual (objective) at minimum is equal to 0.
* `minimum_acceptable_model_improvement`: the minimum improvement in the model function that
  is required to accept a new point; if this is not met, the new point is rejected and
  the damping term is increased.
* `sub_problem`: the linear subproblem solver to use to solve the linearized
  subproblem in each iteration.
* `sub_state`: the state to use for the linear subproblem solver.

# Constructor

    LevenbergMarquardtState(M, initial_residual_values, initial_jacobian; kwargs...)

Generate the Levenberg-Marquardt solver state.

# Keyword arguments

The following fields are keyword arguments

* `β = 5.0`
* `damping_term_min = 0.1`
* `damping_term = damping_term_min`
* `η = 0.2`,
* `expect_zero_residual = false`
* `initial_gradient = `$(_link(:zero_vector))
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-12)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-12)"))
* `minimum_acceptable_model_improvement::Real = eps(number_eltype(p))`

# See also

[`gradient_descent`](@ref), [`LevenbergMarquardt`](@ref)
"""
mutable struct LevenbergMarquardtState{
        P,
        TStop <: StoppingCriterion,
        TRTM <: AbstractRetractionMethod,
        Tresidual_values,
        TGrad,
        TJac,
        Tparams <: Real,
        Pr,
        St,
    } <: AbstractGradientSolverState
    p::P
    stop::TStop
    retraction_method::TRTM
    residual_values::Tresidual_values
    jacobian_f::TJac
    direction::TGrad
    X::TGrad
    η::Tparams
    damping_term::Tparams
    damping_term_min::Tparams
    β::Tparams
    expect_zero_residual::Bool
    minimum_acceptable_model_improvement::Tparams
    sub_problem::Pr
    sub_state::St
    function LevenbergMarquardtState(
            M::AbstractManifold,
            initial_residual_values::Tresidual_values;
            initial_jacobian_f = nothing,
            p::P = rand(M),
            X::TGrad = zero_vector(M, p),
            direction::TGrad = zero_vector(M, p),
            stopping_criterion::StoppingCriterion = StopAfterIteration(200) |
                StopWhenGradientNormLess(1.0e-12) |
                StopWhenStepsizeLess(1.0e-12),
            retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
            η::Real = 0.2,
            damping_term_min::Real = 0.1,
            damping_term::Real = damping_term_min,
            β::Real = 5.0,
            expect_zero_residual::Bool = false,
            minimum_acceptable_model_improvement::Real = eps(number_eltype(p)),
            sub_problem::Pr = nothing,
            sub_state::St = nothing,
        ) where {P, Tresidual_values, TGrad, Pr, St}
        # TODO: what if initial:Jacobian_f is still nothing? Fill it?
        # We could try checking if the provided sub_state actually needs `jacobian_f` or not but it's just about having a nicer error message.
        if isnothing(sub_problem) || isnothing(sub_state)
            s = "You have to specify a solver for the sub problem, that is, both a `sub_problem` to be solved"
            s = "$s and a `sub_state` specifying the solver."
            isnothing(sub_problem) && (s = "$s\n The `sub_problem` was not specified.")
            isnothing(sub_problem) && (s = "$s\n The `sub_state` was not specified.")
            throw(ArgumentError(s))
        end
        if η <= 0 || η >= 1
            throw(ArgumentError("Value of η must be strictly between 0 and 1, received $η"))
        end
        if damping_term_min <= 0
            throw(
                ArgumentError("Value of damping_term_min must be strictly above 0, received $damping_term_min"),
            )
        end
        (β <= 1) && throw(ArgumentError("Value of β must be strictly above 1, received $β"))
        _sub_state = maybe_wrap_evaluation_type(sub_state)
        Tparams = promote_type(typeof(η), typeof(damping_term_min), typeof(β))
        SC = typeof(stopping_criterion)
        RM = typeof(retraction_method)
        return new{
            P, SC, RM, Tresidual_values, TGrad, typeof(initial_jacobian_f), Tparams, Pr, typeof(_sub_state),
        }(
            p,
            stopping_criterion, retraction_method,
            initial_residual_values, initial_jacobian_f,
            direction, X, η,
            damping_term, damping_term_min,
            β,
            expect_zero_residual,
            minimum_acceptable_model_improvement,
            sub_problem, sub_state,
        )
    end
end

function show(io::IO, lms::LevenbergMarquardtState)
    i = get_count(lms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(lms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Levenberg Marquardt Algorithm
    $Iter
    ## Parameters
    * β: $(lms.β)
    * damping term_ $(lms.damping_term) (min: $(lms.damping_term_min))
    * η: $(lms.η)
    * expect zero residual: $(lms.expect_zero_residual)
    * retraction method: $(lms.retraction_method)

    ## Stopping criterion

    $(status_summary(lms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

#
#
# --- Subproblems ----

#
# ----- A cost/grad objective to e.g. do CG, or Gauß-Newton ----

@doc """
    LevenbergMarquardtLinearSurrogateObjective{E<:AbstractEvaluationType, VF<:AbstractManifoldFirstOrderObjective{E}, R} <: AbstractManifoldFirstOrderObjective{E, VF}

Given an [`NonlinearLeastSquaresObjective`](@ref) `objective` and a damping term `damping_term`,
this objective represents the penalized objective for the sub-problem to solve within every step
of the Levenberg-Marquardt algorithm given by

```math
μ_p(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, _tex(:Cal, "L") * "(X) + y"; index = "2"))^2
  + $(_tex(:frac, "λ", "2"))$(_tex(:norm, "X"; index = "p"))^2,
  $(_tex(:qquad))$(_tex(:text, " for "))X ∈ $(_math(:TangentSpace)), λ ≥ 0,
```

where ``X ∈ $(_math(:TangentSpace))`` and ``λ ≥ 0`` is the damping or penalty term.

In order to build a surrogate also for the robutsified Levenberg-Marquardt, introduce
``α = 1 - $(_tex(:sqrt, "1 + 2 $(_tex(:frac, "ρ''(p)", "ρ'(p)"))$(_tex(:norm, "F(p)"; index = "2"))^2"))``
and set ``y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p)`` and ``$(_tex(:Cal, "L"))(X) = CJ_F^*(p)[F(p)]``
with

```math
C = $(_tex(:sqrt, "ρ'(p)"))(I-αP), $(_tex(:qquad)) P = $(_tex(:frac, "F(p)F(p)^" * _tex(:rm, "T"), _tex(:norm, "F(p)"; index = "2") * "^2")),
```

where ``F(p) ∈ ℝ^n`` is the vector of residuals at point ``p ∈ M`` and ``J_F^*(p): ℝ^n → $(_math(:TangentSpace))```
is the adjoint Jacobian.

## Fields

* `objective`:     the [`NonlinearLeastSquaresObjective`](@ref) to penalize
* `penalty::Real`: the damping term ``λ``
* `ε::Real`:       stabilization for ``α ≤ 1-ε`` in the rescaling of the Jacobian, that
* `mode::Symbol`:  which ode to use to stabilize α, see the internal helper [`get_LevenbergMarquardt_scaling`](@ref)
* `value_cache`:   a vector to store the residuals ``F(p)`` at the current point `p` internally to avoid recomputations

## Constructor

    LevenbergMarquardtLinearSurrogateObjective(objective; penalty::Real = 1e-6, ε::Real = 1e-4, mode::Symbol = :Default )

"""
mutable struct LevenbergMarquardtLinearSurrogateObjective{
        E <: AbstractEvaluationType, R <: Real, TO <: NonlinearLeastSquaresObjective{E}, TVC <: AbstractVector{R},
    } <: AbstractLinearSurrogateObjective{E, NonlinearLeastSquaresObjective{E}}
    objective::TO
    penalty::R
    ε::R
    mode::Symbol
    value_cache::TVC
    function LevenbergMarquardtLinearSurrogateObjective(
            objective::NonlinearLeastSquaresObjective{E};
            penalty::R = 1.0e-6, ε::R = 1.0e-4, mode::Symbol = :Default,
            residuals::TVC = zeros(residuals_count(get_objective(objective))),
        ) where {E, R <: Real, TVC <: AbstractVector}
        return new{E, R, typeof(objective), TVC}(objective, penalty, ε, mode, residuals)
    end
end

function show(io::IO, o::LevenbergMarquardtLinearSurrogateObjective{E}) where {E}
    return print(io, "LevenbergMarquardtLinearSurrogateObjective{$E}($(o.objective); penalty=$(o.penalty), ε=$(o.ε), mode=:$(o.mode))")
end

"""
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime::Real, ρ_double_prime::Real, FSq::Real, ε::Real, mode::Symbol)

Compute the scaling ``$(_tex(:frac, _tex(:sqrt, "ρ'"), "1 - α"))`` for the residula ``y`` and
the scaling ``$(_tex(:frac, "α", _tex(:norm, "F"; index = "2") * "^2"))`` that are required for the robust
rescaling within [`LevenbergMarquardt`](@ref)s [`get_vector_field`](@ref) and [`get_linear_operator`](@ref),
respectively. The value for ``α`` is given by

```math
    α = 1-$(_tex(:sqrt, "1 + 2$(_tex(:frac, "ρ_k''", "ρ_k'"))$(_tex(:norm, "F_k"; index = "2"))"))
```

where
* ``ρ'`` is the first derivative of the [`AbstractRobustifierFunction`](@ref) at ``$(_tex(:norm, "F"; index = "2"))``
* ``ρ''`` is the second derivative of the [`AbstractRobustifierFunction`](@ref) at ``$(_tex(:norm, "k"; index = "2"))``
* `FSq` is the value ``$(_tex(:norm, "F"; index = "2"))``

## Numerical stability

For a unique solution that is a minimizer in a Levenberg-Marquardt step,
we require `α < 1` and [TriggsMcLauchlanHartleyFitzgibbon:2000](@cite) recommends to bound this even by ``1-ε``.

Furthermore if ``ρ´_k + 2ρ''_k $(_tex(:norm, "F"; index = "2")) ≤ 0`` the Hessian is also indefinite.
This can be caught by making sure the argument of the ``√`` is ensured to be nonnegative.

The [Ceres solver](http://ceres-solver.org/nnls_modeling.html#theory) even omits the second term
in the square root already if ``ρ_k'' < 0`` for stability reason, which means setting ``α = 0``.
In the case ``$(_tex(:norm, "F"; index = "2"))`` we also set the operator scaling ``α / FkSq = 0``.

## Additional arguments

* `ε::Real`: the stability for ``α`` to not be too close to one.
* `mode::Symbol` specify the mode of calculation
  - `:Default` keeps negative ``ρ''_k < 0`` but makes sure the square root is well-defined.
  - `:Strict` set ``α = 0`` when ``ρ''_k < 0`` like Ceres does
"""
function get_LevenbergMarquardt_scaling(
        ρ_prime::Real, ρ_double_prime::Real, FkSq::Real,
        ε::Real, mode::Symbol
    )
    # second derivative existent and negative: In strict mode (motivated by ceres) -> return sqrt(ρ_prime), 0
    (ismissing(ρ_double_prime) || (ρ_double_prime < 0 && mode == :Strict)) && return (sqrt(ρ_prime), 0.0)
    (iszero(FkSq) && mode == :Strict) && return (sqrt(ρ_prime), 0.0)
    α = 1 - sqrt(max(1 + 2 * (ρ_double_prime / ρ_prime) * FkSq, 0.0))
    α = min(α, 1 - ε)
    residual_scaling = sqrt(ρ_prime) / (1 - α)
    operator_scaling = ifelse(iszero(FkSq), 0.0, α / FkSq)
    return residual_scaling, operator_scaling
end
function set_parameter!(lmlso::LevenbergMarquardtLinearSurrogateObjective, ::Val{:Penalty}, penalty::Real)
    lmlso.penalty = penalty
    return lmlso
end

get_objective(lmsco::LevenbergMarquardtLinearSurrogateObjective) = lmsco.objective
"""
    get_cost(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )

Compute the surrogate cost. Let ``F`` denote the vector of residuals (of a block),
``ρ, ρ'``, ``ρ''`` the value, first, and second derivative of the [`AbstractRobustifierFunction`](@ref)
of the inner [`NonlinearLeastSquaresObjective`](@ref)

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
"""
    get_cost(TpM::TangentSpace, slso::SymmetricLInearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, X)

Compute the surrogate cost when solving its normal equation, see also
[`get_cost(::AbstractManifold, ::LevenbergMarquardtLinearSurrogateObjective, p, X)`](@ref),
[`get_linear_operator`](@ref), and [`get_vector_field`](@ref) for more details.
"""
function get_cost(
        TpM::TangentSpace, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, X
    ) where {E <: AbstractEvaluationType}
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_cost(M, slso.objective, p, X)
end

# TODO: Write docs.
function get_gradient(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    Y = zero_vector(M, p)
    return get_gradient!(M, Y, lmsco, p, X)
end
function get_gradient!(
        M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    value_cache = lmsco.value_cache
    nlso = lmsco.objective
    # For every block
    zero_vector!(M, Y, p)
    Z = copy(M, p, Y)
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len_o = length(o)
        _get_gradient!(
            M, Z, o, r, p, X;
            value_cache = value_cache[(start + 1):(start + len_o)], ε = lmsco.ε, mode = lmsco.mode,
        )
        Y .+= Z
        start += len_o
    end
    # add penalty term
    Y .+= lmsco.penalty .* X
    return Y
end
function _get_gradient!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol,
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
    b = zero(a)
    get_jacobian!(M, b, o, p, X)
    # Compute C^TCa = C^2 a (inplace of a)
    b .= ρ_prime .* (I - operator_scaling * (a * a'))^2 * b
    # add C^T y = C^T (sqrt(ρ(p)) / (1 - α) F(p)) (which overall has a ρ_prime upfront)
    b .+= residual_scaling .* sqrt(ρ_prime) .* (I - operator_scaling * (a * a')) * a
    # apply the adjoint
    zero_vector!(M, Y, p)
    add_adjoint_jacobian!(M, Y, o, p, b)
    return Y
end
# Componentwise
function _get_gradient!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, cr::ComponentwiseRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol,
    )
    # per single component a for-loop similar to the one for the blocks
    r = cr.robustifier
    zero_vector!(M, Y, p)
    b = zero(value_cache)
    get_jacobian!(M, b, o, p, X)
    # Componentwise a few things decouple
    for (i, ai) in enumerate(value_cache)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, ε, mode)
        # get the “Jacobian” of the ith component, i.e. its
        # Compute C^TCa = C^2 a (inplace of a)
        b[i] = ρ_prime * (1 - operator_scaling * ai_sq)^2 * b[i]
        # add C^T y = C^T (sqrt(ρ(p)) / (1 - α) F(p)) (which overall has a ρ_prime upfront)
        b[i] += residual_scaling * sqrt(ρ_prime) * (1 - operator_scaling * ai_sq) * ai
    end
    # apply the adjoint
    zero_vector!(M, Y, p)
    add_adjoint_jacobian!(M, Y, o, p, b)
    return Y
end

"""
    default_lm_lin_solve!(sk, JJ::AbstractMatrix, grad_f_c)

Solve the system `JJ \\ grad_f_c` where JJ is (mathematically) a symmetric positive
definite matrix and save the result to `sk`. In case of numerical errors the
`PosDefException` is caught and the default symmetric solver `(Symmetric(JJ) \\ grad_f_c)`
is used.

The function is intended to be used with [`LevenbergMarquardt`](@ref).
"""
function default_lm_lin_solve!(sk, JJ::AbstractMatrix, grad_f_c)
    try
        ldiv!(sk, cholesky(Symmetric(JJ)), grad_f_c)
    catch e
        if e isa PosDefException
            sk .= Symmetric(JJ) \ grad_f_c
        else
            rethrow()
        end
    end
    return sk
end

#
#
# We can do a closed form solution of the Surrogate using \ as soon as we have a basis
# We can model that as a state of a solver for ease of use
# TODO / temp remark: move to the coordinates file?
"""
    CoordinatesNormalSystemState <: AbstractManoptSolverState

A solver state indicating that we solve the [`LevenbergMarquardtLinearSurrogateObjective`](@ref)
using a linear system in coordinates of the tangent space at the current iterate

# Fields
* `A` an ``n×n`` matrix to store the normal operator in coordinates, where `n` is the number of coordinates
* `b` a ``n`` vector storing the right hand side of the system of normal equations
* `basis::`[`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`)
* `linsolve` a functor `(A,b) -> c` to solve the linear system or `(c, A, b) -> c` depending on the evaluation type specified in `solve!`
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
function solve!(dmp::DefaultManoptProblem{<:TangentSpace}, cnss::CoordinatesNormalSystemState{AllocatingEvaluation})
    # Update A and b
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    o = get_objective(dmp)
    get_linear_operator!(M, cnss.A, o, p, cnss.basis)
    get_vector_field!(M, cnss.b, o, p, cnss.basis)
    cnss.c = cnss.linsolve!!(cnss.A, -cnss.b)
    X = get_vector(M, p, cnss.c, cnss.basis)
    # TODO: Remove when we are sure everything is correct, but for now check that the cost is not much worse than the zero step, which should never happen
    if get_cost(dmp, 0 * X) < get_cost(dmp, -X) - sqrt(eps()) * (1 + sqrt(eps()) * get_cost(dmp, 0 * X))
        @show get_cost(dmp, 0 * X)
        @show get_cost(dmp, -X)
        error("model cost is much worse than zero step, something is wrong")
    end
    return cnss
end
function solve!(dmp::DefaultManoptProblem{<:TangentSpace}, cnss::CoordinatesNormalSystemState{InplaceEvaluation})
    # Update A and b
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    o = get_objective(dmp) # implicit: SymmetricSystem ...
    get_linear_operator!(M, cnss.A, o, p, cnss.basis)
    get_vector_field!(M, cnss.b, o, p, cnss.basis)
    cnss.b .*= -1
    cnss.linsolve!!(cnss.c, cnss.A, cnss.b)
    return cnss
end
# Maybe a bit too precise, but in this case we get a coefficient vector and we want a tangent vector
function get_solver_result(
        dmp::DefaultManoptProblem{<:TangentSpace, <:SymmetricLinearSystem{<:AbstractEvaluationType, <:LevenbergMarquardtLinearSurrogateObjective}},
        cnss::CoordinatesNormalSystemState
    )
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_vector(M, p, cnss.c, cnss.basis)
end
#
#
#
"""
    get_linear_normal_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    get_linear_normal_operator(M::AbstractManifold, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    get_linear_normal_operator!(M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    get_linear_normal_operator!(M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X; ε = lmsco.ε, mode = lmsco.mod)

    get_linear_normal_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p[, c], B::AbstractBasis; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    get_linear_normal_operator(M::AbstractManifold, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p[, c], B::AbstractBasis; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    get_linear_normal_operator!(M::AbstractManifold, [A | b], lmsco::LevenbergMarquardtLinearSurrogateObjective, p[, c], B::AbstractBasis; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    get_linear_normal_operator!(M::AbstractManifold, [A | b], o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p[, c], B::AbstractBasis; ε = lmsco.ε, mode = lmsco.mode)

Compute the linear operator ``$(_tex(:Cal, "A"))`` corresponding to the optimality conditions of the
modified Levenberg-Marquardt surrogate objective, i.e. the normal conditions

```math
$(_tex(:Cal, "A"))(X) = $(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))(X) + λX
= J_F^*(p)$(_tex(:bigl))[ C^T C J_F(p)[X] $(_tex(:bigr))] + λX,
```

where ``λ = ```penalty` is a damping parameter.
While this can be set as a keyword, e.g. when computing the cost, its default is the internally
stored `penalty` from the surrogate objective ``μ_k``.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

If you provide an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `B` ``Z_1,…,Z_d`` instead, the operator is returned in Matrix form `A`,
with respect to these coordinates, i.e. if you provide furthermore `X`
in coordinates `c` as ``X = $(_tex(:displaystyle))$(_tex(:sum, "i=0", "d")) c_iZ_i``,
then the result ``Y = $(_tex(:displaystyle))$(_tex(:sum, "i=0", "d")) b_iZ_i`` with ``b = Ac`` is computed.

See also [`get_normal_vector_field`](@ref) for evaluating the corresponding vector field.
"""
function get_linear_normal_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        penalty::Real = lmsco.penalty,
    )
    Y = zero_vector(M, p)
    return get_linear_normal_operator!(M, Y, lmsco, p, X; penalty = penalty)
end
function get_linear_normal_operator!(
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
        add_linear_normal_operator!(M, Y, o, r, p, X; ε = lmsco.ε, mode = lmsco.mode, value_cache = value_cache, Y_cache = Y_cache)
        start += len
    end
    # Finally add the damping term
    (penalty != 0) && (Y .+= penalty .* X)
    return Y
end
# for a single block – the actual formula - but never with penalty
function add_linear_normal_operator!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol, Y_cache = zero_vector(M, p)
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
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
function _get_linear_normal_operator!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, cr::ComponentwiseRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol, Y_cache = nothing,
    )
    b = zero(value_cache)
    get_jacobian!(M, b, o, p, X)
    r = cr.robustifier
    for (i, ai) in enumerate(value_cache)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, ε, mode)
        # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
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
function get_linear_normal_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, c, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    d = zero(c)
    return get_linear_normal_operator!(M, d, lmsco, p, c, B; penalty = penalty)
end
function get_linear_normal_operator!(
        M::AbstractManifold, d, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, c, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(d, 0)
    e = copy(d)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        _get_linear_normal_operator!(M, e, o, r, p, c, B; ε = lmsco.ε, mode = lmsco.mode)
        d .+= e
    end
    # Finally add the damping term
    (penalty != 0) && (d .+= penalty * c)
    return d
end
function _get_linear_normal_operator!(M::AbstractManifold, d, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, c, B::AbstractBasis; kwargs...)
    A = get_linear_normal_operator(M, o, r, p, B; kwargs...)
    d .= A * c
    return d
end
#
# Basis case: (b) no coordinates -> compute a matrix representation
function get_linear_normal_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    d = number_of_coordinates(M, B)
    A = zeros(number_eltype(p), d, d)
    return get_linear_normal_operator!(M, A, lmsco, p, B; penalty = penalty)
end
function get_linear_normal_operator!(
        M::AbstractManifold, A::AbstractMatrix, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(A, 0)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        add_linear_normal_operator!(M, A, o, r, p, B; ε = lmsco.ε, mode = lmsco.mode)
    end
    # Finally add the damping term
    (penalty != 0) && (LinearAlgebra.diagview(A) .+= penalty)
    return A
end
"""
    add_linear_normal_operator!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractVectorGradientFunction,
        r::AbstractRobustifierFunction, p, basis::AbstractBasis;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol
    )

Add the contribution of a single block (vectorial function with its robustifier) to
the linear normal operator, i.e. compute ``A += J_F^*(p)[C^T C J_F(p)[X]]`` in-place of `A`
for the given block.
"""
function add_linear_normal_operator!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractVectorGradientFunction,
        r::AbstractRobustifierFunction, p, basis::AbstractBasis;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # to Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
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
function add_linear_normal_operator!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractVectorGradientFunction,
        cr::ComponentwiseRobustifierFunction, p, basis::AbstractBasis;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol
    )
    a = value_cache # evaluate residuals F(p)
    b = zero(a)
    r = cr.robustifier
    for (i, ai) in enumerate(a)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, ε, mode)
        # to Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
        # (a) J_F is n-by-d so we have to allocate – where could we maybe store something like that and pass it down?
        b[i] = ρ_prime * (1 - operator_scaling * ai_sq)^2
    end
    JF = get_jacobian(M, o, p; basis = basis)
    # compute A' C^TC A (C^TC = C^2 here) inplace of A
    A .+= JF' * Diagonal(b) * JF
    # damping term is added once after summing up all blocks, so we do not add it here
    return A
end

"""
    get_linear_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X)
    get_linear_operator(M::AbstractManifold, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X)
    get_linear_operator!(M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X)
    get_linear_operator!(M::AbstractManifold, y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X)

Compute the linear operator ``$(_tex(:Cal, "L"))`` corresponding to the optimality conditions of the
Levenberg-Marquardt surrogate objective, i.e. the normal conditions

```math
$(_tex(:Cal, "L"))(X) = C J_F(p)[X] $(_tex(:bigr))],
```

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.
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
    # TODO: use the actual basis? store it in the VGF instead maybe?
    c_cache = allocate_result(M, get_coordinates, p, X, DefaultOrthonormalBasis())
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        _get_linear_operator!(
            M, view(y, (start + 1):(start + len)), o, r, p, X, value_cache;
            ε = lmsco.ε, mode = lmsco.mode, Y_cache = Y_cache, c_cache = c_cache,
        )
        start += len
    end
    return y
end
# for a single block – the actual formula
function _get_linear_operator!(
        M::AbstractManifold, y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X,
        value_cache = get_value(M, o, p); ε::Real, mode::Symbol, Y_cache, c_cache
    )
    F_sq = sum(abs2, value_cache)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    get_jacobian!(M, y, o, p, X; Y_cache = Y_cache, c_cache = c_cache)
    # Compute C y
    α = sqrt(ρ_prime)
    t = dot(value_cache, y)
    @. y = α * (y - operator_scaling * t * value_cache)
    return y
end
# Componenwise: Decouple
function _get_linear_operator!(
        M::AbstractManifold, y, o::AbstractVectorGradientFunction, cr::ComponentwiseRobustifierFunction, p, X,
        value_cache = get_value(M, o, p); ε::Real, mode::Symbol, Y_cache, c_cache
    )
    a = value_cache
    r = cr.robustifier
    get_jacobian!(M, y, o, p, X)
    for (i, ai) in enumerate(value_cache)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, ε, mode)
        # get the “Jacobian” of the ith component, i.e. y[i]
        # C is justr a diagonal matrix here
        y[i] = sqrt(ρ_prime) * (1 - operator_scaling * ai_sq) * y[i]
    end
    return y
end

_doc_get_normal_vector_field = """
    get_normal_vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    get_normal_vector_field!(M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    get_normal_vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis)
    get_normal_vector_field!(M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis)

Compute the normal linear operator tangent vector ``X`` corresponding to the optimality conditions of the
Levenberg-Marquardt surrogate objective, i.e.,

```math
X = J_F^*(p)[ C^T y], $(_tex(:quad)) y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p).
```

If you provide an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `B` ``=$(_tex(:set, "Z_1,…,Z_d"))`` additionally,
the result will be given in coordinates `c`, i.e. such that ``X = $(_tex(:sum, "i=1", "d")) c_iZ_i``.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`get_linear_normal_operator`](@ref) for evaluating the corresponding linear operator of the (normal) linear system,
and [`get_LevenbergMarquardt_scaling`](@ref) for details on the scaling and computation of ``C``.
"""

_doc_add_normal_vector_field = """
    add_normal_vector_field!(M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p)
    add_normal_vector_field!(M::AbstractManifold, c, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, B::AbstractBasis)

Add the contribution of `o` / `r` to the normal linear operator tangent vector in `X` or `c`.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`get_linear_normal_operator`](@ref) for evaluating the corresponding linear operator of the (normal) linear system,
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
            ε = lmsco.ε, mode = lmsco.mode, value_cache = value_cache, Y_cache = Y_cache,
        )
        start += len
        X .+= Z
    end
    return X
end
# for a single block – the actual formula
function _get_normal_vector_field!(
        M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol, Y_cache = zero_vector(M, p),
    )
    y = copy(value_cache)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
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
        M::AbstractManifold, X, o::AbstractVectorGradientFunction, cr::ComponentwiseRobustifierFunction, p;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol, Y_cache = nothing,
    )
    y = copy(value_cache)
    r = cr.robustifier
    for (i, ai) in enumerate(y)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, ε, mode)
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
        add_normal_vector_field!(M, c, o, r, p, B; ε = lmsco.ε, mode = lmsco.mode)
    end
    return c
end

# for a single block – the actual formula
@doc "$(_doc_add_normal_vector_field)"
function add_normal_vector_field!(
        M::AbstractManifold, c, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, B::AbstractBasis;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol,
    )
    y = copy(value_cache) # evaluate residuals F(p)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # Compute y = ρ'(p) / (1-α)) F(p) and ...
    y .= residual_scaling .* sqrt(ρ_prime) * (I - operator_scaling * (y * y')) * y
    # ...apply the adjoint, i.e. compute  J_F^*(p)[C^T y] (inplace of y)
    add_adjoint_jacobian!(M, c, o, p, y, B)
    return c
end
# Compponentwise: decouple, C is a diagonalmatrix
function add_normal_vector_field!(
        M::AbstractManifold, c, o::AbstractVectorGradientFunction, cr::ComponentwiseRobustifierFunction, p, B::AbstractBasis;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol, Y_cache = nothing,
    )
    y = copy(value_cache) # evaluate residuals F(p)
    r = cr.robustifier
    for (i, ai) in enumerate(y)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, ε, mode)
        # Compute y = ρ'(p) / (1-α)) F(p) and ...
        y[i] = residual_scaling * sqrt(ρ_prime) * (1 - operator_scaling * ai_sq) * ai
    end
    # ...apply the adjoint, i.e. compute  J_F^*(p)[C^T y] (inplace of y)
    add_adjoint_jacobian!(M, c, o, p, y, B)
    return c
end

"""
    get_vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    get_vector_field!(M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    get_vector_field!(M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p)

Compute the vector field ``y`` corresponding to the Levenberg-Marquardt surrogate objective, i.e.,

```math
y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p).
```

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also
* [`get_LevenbergMarquardt_scaling`](@ref) for details on the scaling
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
        M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    # Init to zero
    fill!(y, 0)
    start = 0
    # For every block
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        _get_vector_field!(M, view(y, (start + 1):(start + length(o))), o, r, p; ε = lmsco.ε, mode = lmsco.mode)
        start += length(o)
    end
    return y
end
# for a single block – the actual formula
function _get_vector_field!(
        M::AbstractManifold, y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
        ε::Real, mode::Symbol,
    )
    get_value!(M, y, o, p) # evaluate residuals F(p)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, _ = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # Compute y = sqrt(ρ(p)) / (1-α) * F(p)
    y .*= residual_scaling
    return y
end
# Componentwise, it decouples, C is diagonal
function _get_vector_field!(
        M::AbstractManifold, y, o::AbstractVectorGradientFunction, cr::ComponentwiseRobustifierFunction, p;
        ε::Real, mode::Symbol,
    )
    get_value!(M, y, o, p) # evaluate residuals F(p)
    r = cr.robustifier
    for (i, ai) in enumerate(y)
        ai_sq = abs(ai)^2
        (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, ai_sq)
        residual_scaling, _ = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, ai_sq, ε, mode)
        # Compute y = sqrt(ρ(p)) / (1-α) * F(p)
        y[i] *= residual_scaling
    end
    return y
end

#
# The Symmetric Linear System (e.g. in CGRes) for the LM Surrogate is its normal equations and vector.
# (a) a vector X or a basis B
function get_linear_operator(
        M::AbstractManifold, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, XB
    ) where {E <: AbstractEvaluationType}
    return get_linear_normal_operator(M, slso.objective, p, XB)
end
function get_linear_operator!(
        M::AbstractManifold, YA, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, XB
    ) where {E <: AbstractEvaluationType}
    return get_linear_normal_operator!(M, YA, slso.objective, p, XB)
end
# (b) coefficients in a basis
function get_linear_operator(
        M::AbstractManifold, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, c, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return get_linear_normal_operator(M, slso.objective, p, c, B)
end
function get_linear_operator!(
        M::AbstractManifold, Y, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, c, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return get_linear_normal_operator!(M, Y, slso.objective, p, c, B)
end
# RHS as a tangent vector
function get_vector_field(
        M::AbstractManifold, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p
    ) where {E <: AbstractEvaluationType}
    return -get_normal_vector_field(M, slso.objective, p)
end
function get_vector_field!(
        M::AbstractManifold, Y, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p
    ) where {E <: AbstractEvaluationType}
    get_normal_vector_field!(M, Y, slso.objective, p)
    Y .*= -1
    return Y
end
# RHS in coordinates
function get_vector_field(
        M::AbstractManifold, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return -get_normal_vector_field(M, slso.objective, p, B)
end
function get_vector_field!(
        M::AbstractManifold, c, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    get_normal_vector_field!(M, c, slso.objective, p, B)
    c .*= -1
    return c
end
