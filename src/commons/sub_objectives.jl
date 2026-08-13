"""
    AbstractLevenbergMarquardtLinearSurrogateObjective

Abstract supertype for Levenberg-Marquardt surrogates like
[`LevenbergMarquardtLinearSurrogateObjective`](@ref) and
[`LevenbergMarquardtLinearSurrogateCoordinatesObjective`](@ref).
"""
abstract type AbstractLevenbergMarquardtLinearSurrogateObjective <: AbstractLinearSurrogateObjective{ManifoldNonlinearLeastSquaresObjective} end

_doc_AL_Cost(iter) = "$(_tex(:Cal, "L"))_{ρ^{($iter)}}(p, μ^{($iter)}, λ^{($iter)})"
_doc_AL_Cost_long = """
```math
$(_tex(:Cal, "L"))_ρ(p, μ, λ)
= f(x) + $(_tex(:frac, "ρ", "2"))$(_tex(:biggl))(
  $(_tex(:sum, "j=1", "n"))$(_tex(:Bigl))(
    h_j(p) + $(_tex(:frac, "λ_j", "ρ"))
  $(_tex(:Bigr)))^2
  +
  $(_tex(:sum, "i=1", "m"))$(_tex(:max))$(_tex(:set, "0, $(_tex(:frac, "μ_i", "ρ")) + g_i(p)"))^2
$(_tex(:biggr)))
```
"""

@doc """
    AugmentedLagrangianCost{CO,R,T} <: AbstractConstrainedFunction{CO}

Stores the parameters ``ρ ∈ ℝ``, ``μ ∈ ℝ^m``, ``λ ∈ ℝ^n``
of the augmented Lagrangian associated to the [`ConstrainedManifoldObjective`](@ref) `co`.

This struct is also a functor `(M,p) -> v` that can be used as a cost function within a solver,
based on the internal [`ConstrainedManifoldObjective`](@ref) it computes

$_doc_AL_Cost_long

## Fields

* `co::CO`, `ρ::R`, `μ::T`, `λ::T` as mentioned in the formula, where ``R`` should be the
number type used and ``T`` the vector type.

# Constructor

    AugmentedLagrangianCost(co, ρ, μ, λ)
"""
mutable struct AugmentedLagrangianCost{CO, R, T} <: AbstractConstrainedFunction{CO}
    co::CO
    ρ::R
    μ::T
    λ::T
end
function set_parameter!(alc::AugmentedLagrangianCost, ::Val{:ρ}, ρ)
    alc.ρ = ρ
    return alc
end
get_parameter(alc::AugmentedLagrangianCost, ::Val{:ρ}) = alc.ρ
# μ & λ already set through the abstract constrained function
function (L::AugmentedLagrangianCost)(M::AbstractManifold, p)
    gp = get_inequality_constraint(M, L.co, p, :)
    hp = get_equality_constraint(M, L.co, p, :)
    m = length(gp)
    n = length(hp)
    c = get_cost(M, L.co, p)
    d = 0.0
    (m > 0) && (d += sum(max.(zeros(m), L.μ ./ L.ρ .+ gp) .^ 2))
    (n > 0) && (d += sum((hp .+ L.λ ./ L.ρ) .^ 2))
    return c + (L.ρ / 2) * d
end

@doc """
    AugmentedLagrangianGrad{CO,R,T} <: AbstractConstrainedFunction{T}

Stores the parameters ``ρ ∈ ℝ``, ``μ ∈ ℝ^m``, ``λ ∈ ℝ^n``
of the augmented Lagrangian associated to the [`ConstrainedManifoldObjective`](@ref) `co`.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

additionally this gradient does accept a positional last argument to specify the `range`
for the internal gradient call of the constrained objective.

based on the internal [`ConstrainedManifoldObjective`](@ref) and computes the gradient
`$(_tex(:grad))$(_tex(:Cal, "L"))_{ρ}(p, μ, λ)``, see also [`AugmentedLagrangianCost`](@ref).

## Fields

* `co::CO`, `ρ::R`, `μ::T`, `λ::T` as mentioned in the formula, where ``R`` should be the
number type used and ``T`` the vector type.

# Constructor

    AugmentedLagrangianGrad(co, ρ, μ, λ)

"""
mutable struct AugmentedLagrangianGrad{CO, R, T} <: AbstractConstrainedFunction{T}
    co::CO
    ρ::R
    μ::T
    λ::T
end
function (LG::AugmentedLagrangianGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return LG(M, X, p)
end
function set_parameter!(alg::AugmentedLagrangianGrad, ::Val{:ρ}, ρ)
    alg.ρ = ρ
    return alg
end
get_parameter(alg::AugmentedLagrangianGrad, ::Val{:ρ}) = alg.ρ
# μ & λ already set through the abstract constrained function

# default, that is especially when the `grad_g` and `grad_h` are functions.
function (LG::AugmentedLagrangianGrad)(
        M::AbstractManifold, X, p, range = NestedPowerRepresentation()
    )
    gp = get_inequality_constraint(M, LG.co, p, :)
    hp = get_equality_constraint(M, LG.co, p, :)
    m = length(gp)
    n = length(hp)
    get_gradient!(M, X, LG.co, p)
    if m > 0
        indices = (gp .+ LG.μ ./ LG.ρ) .> 0
        if sum(indices) > 0
            weights = (gp .* LG.ρ .+ LG.μ)[indices]
            X .+= sum(
                weights .* get_grad_inequality_constraint(M, LG.co, p, indices, range)
            )
        end
    end
    if n > 0
        X .+= sum(
            (hp .* LG.ρ .+ LG.λ) .* get_grad_equality_constraint(M, LG.co, p, :, range)
        )
    end
    return X
end

#
#
# ---
@doc """
    LevenbergMarquardtLinearSurrogateCoordinatesObjective{VF<:AbstractManifoldFirstOrderObjective, R} <: AbstractLevenbergMarquardtLinearSurrogateObjective{E}

A subobjective similar to `LevenbergMarquardtLinearSurrogateObjective` but which uses
coordinate-based Jacobians in a single, selected basis instead of being centered around
linear operators.
## Fields

* `objective`:     the [`ManifoldNonlinearLeastSquaresObjective`](@ref) to penalize
* `penalty::Real`: the damping term ``λ``
* `threshold::Real`: stabilization ``ε`` for ``α ≤ 1-ε`` in the rescaling of the residual and jacobian, see [`get_LevenbergMarquardt_scaling`](@ref)
* `mode::Symbol`:  which mode to use to stabilize α, see the internal helper [`get_LevenbergMarquardt_scaling`](@ref)
* `value_cache`:   a vector to store the residuals ``F(p)`` at the current point `p` internally to avoid recomputations
* `jacobian_cache`: a vector to store the coordinate-based Jacobian of the residuals at the
  current point `p` internally to avoid recomputations. If the Jacobian is used as a linear
  operator, this is just a vector of `nothing`s.

## Constructor

    LevenbergMarquardtLinearSurrogateCoordinatesObjective(objective; penalty::Real = 1e-6, threshold::Real = 1e-4, mode::Symbol = :Strict)
"""
mutable struct LevenbergMarquardtLinearSurrogateCoordinatesObjective{
        R <: Real, TO <: ManifoldNonlinearLeastSquaresObjective, TVC <: AbstractVector{R}, TJC <: AbstractVector, TB <: AbstractBasis,
    } <: AbstractLevenbergMarquardtLinearSurrogateObjective
    objective::TO
    penalty::R
    threshold::R
    mode::Symbol
    value_cache::TVC
    jacobian_cache::TJC
    basis::TB
    function LevenbergMarquardtLinearSurrogateCoordinatesObjective(
            objective::ManifoldNonlinearLeastSquaresObjective;
            penalty::R = 1.0e-6, threshold::R = 1.0e-4, mode::Symbol = :Strict,
            residuals::TVC = zeros(residuals_count(get_objective(objective))),
            jacobian_cache::TJC = fill(nothing, length(get_objective(objective).objective)),
            basis::TB = DefaultOrthonormalBasis(),
        ) where {R <: Real, TVC <: AbstractVector, TJC <: AbstractVector, TB <: AbstractBasis}
        return new{R, typeof(objective), TVC, TJC, TB}(objective, penalty, threshold, mode, residuals, jacobian_cache, basis)
    end
end

function get_normal_linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco.objective)
    # For every block
    fill!(A, 0)
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len_o = length(o)
        add_normal_linear_operator_coord!(
            M, A, o, r, p, B; value_cache = view(lmsco.value_cache, (start + 1):(start + len_o)), jacobian_cache = jc,
            threshold = lmsco.threshold, mode = lmsco.mode
        )
        start += len_o
    end
    # Finally add the damping term
    (penalty != 0) && (_diagview(A) .+= penalty)
    return A
end
function add_normal_linear_operator_coord!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractVectorGradientFunction,
        r::AbstractRobustifierFunction, p, basis::AbstractBasis;
        value_cache, jacobian_cache, threshold::Real, mode::Symbol
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # to Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared indirectly
    # (a) J_F is n-by-d so we have to allocate – where could we maybe store something like that and pass it down?
    # (I - s*a*a')^2 = I + (-2s + s^2*||a||^2) * a*a'
    # so JF' * (ρ' * (I - s*a*a')^2) * JF
    #   = ρ' * (JF'JF) + ρ' * (-2s + s^2*||a||^2) * (JF'a) * (JF'a)'
    rank1_scaling = ρ_prime * (-2 * operator_scaling + operator_scaling^2 * F_sq)
    mul!(A, jacobian_cache', jacobian_cache, ρ_prime, true)
    if !iszero(rank1_scaling)
        JFa = jacobian_cache' * a
        mul!(A, JFa, JFa', rank1_scaling, true)
    end
    # damping term is added once after summing up all blocks, so we do not add it here
    return A
end

function add_normal_linear_operator_coord!(
        M::AbstractManifold, c::AbstractVector,
        lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p, cX::AbstractVector;
        penalty::Real = lmsco.penalty,
    )
    nlso = get_objective(lmsco)
    # For every block
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        add_normal_linear_operator_coord!(
            M, c, o, r, p, cX;
            threshold = lmsco.threshold, mode = lmsco.mode, value_cache = value_cache, jacobian_cache = jc
        )
        start += len
    end
    # Finally add the damping term
    (penalty != 0) && (c .+= penalty .* cX)
    return c
end
function add_normal_linear_operator_coord!(
        M::AbstractManifold, c::AbstractVector, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, cX::AbstractVector;
        value_cache, jacobian_cache, threshold::Real, mode::Symbol
    )
    a = value_cache # residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared indirectly
    b = convert(Vector, jacobian_cache * cX)
    # Compute C^TCb = C^2 b
    # The code below is mathematically equivalent to the following, but avoids allocating
    # the outer product a * a' and the matrix-vector product (a * a') * b
    # b .= ρ_prime .* (I - operator_scaling * (a * a'))^2 * b
    t = dot(a, b)
    aa = dot(a, a)
    coef = operator_scaling * t * (operator_scaling * aa - 2)

    @. b = ρ_prime * (b + coef * a)

    # Now apply the adjoint
    mul!(c, jacobian_cache', b, true, true)
    # penalty is added once after summing up all blocks, so we do not add it here
    return c
end
"""
    add_linear_operator_coord!(
        M::AbstractManifold, y::AbstractVector, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p, cX::AbstractVector
    )

Add the (Triggs correction, residual-like) linear operator corresponding to the `lmsco`
surrogate to vector `y`. It is assumed that `lmsco.value_cache` has been filled in
`step_solver!` of [`LevenbergMarquardt`](@ref), so we can just use it here.
"""
function add_linear_operator_coord!(
        M::AbstractManifold, y::AbstractVector, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p, cX::AbstractVector
    )
    nlso = get_objective(lmsco)
    # Init to zero
    start = 0
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        _add_linear_operator_coord!(
            M, view(y, (start + 1):(start + len)), o, r, p, cX, value_cache, jc;
            threshold = lmsco.threshold, mode = lmsco.mode
        )
        start += len
    end
    return y
end
function _add_linear_operator_coord!(
        M::AbstractManifold, y::AbstractVector, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, cX::AbstractVector,
        value_cache, jacobian_cache; threshold::Real, mode::Symbol
    )
    F_sq = sum(abs2, value_cache)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    y_cache = jacobian_cache * cX
    # Compute C y
    α = sqrt(ρ_prime)
    t = dot(value_cache, y_cache)
    @. y += α * (y_cache - operator_scaling * t * value_cache)
    return y
end

function get_normal_vector_field_coord!(
        M::AbstractManifold, c::AbstractVector, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p,
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(c, 0)
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len_o = length(o)
        add_normal_vector_field_coord!(
            M, c, o, r, p;
            value_cache = view(lmsco.value_cache, (start + 1):(start + len_o)),
            jacobian_cache = jc, threshold = lmsco.threshold, mode = lmsco.mode
        )
        start += len_o
    end
    return c
end

# for a single block – the actual formula cf. nls_general 1348
function add_normal_vector_field_coord!(
        M::AbstractManifold, c::AbstractVector, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
        value_cache, jacobian_cache, threshold::Real, mode::Symbol,
    )
    y = copy(value_cache) # evaluate residuals F(p)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, threshold, mode)
    # Compute y = ρ'(p) / (1-α)) F(p) and ...
    y .= residual_scaling .* sqrt(ρ_prime) * (I - operator_scaling * (y * y')) * y
    # ...apply the adjoint, i.e. compute  J_F^*(p)[C^T y] (inplace of y)
    mul!(c, jacobian_cache', y, true, true)
    return c
end

function set_parameter!(lmlso::LevenbergMarquardtLinearSurrogateCoordinatesObjective, ::Val{:Penalty}, penalty::Real)
    lmlso.penalty = penalty
    return lmlso
end

function show(io::IO, lmlsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective)
    print(io, "LevenbergMarquardtLinearSurrogateCoordinatesObjective(", lmlsco.objective, "; ")
    print(io, "penalty=", lmlsco.penalty, ", threshold=", lmlsco.threshold, ", mode=:", lmlsco.mode)
    print(io, ", basis = ", lmlsco.basis)
    print(io, ", residuals=", lmlsco.value_cache, ", jacoian_cache=", lmlsco.jacobian_cache)
    return print(io, ")")
end

function status_summary(lmlsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective; context::Symbol = :default)
    (context === :short) && (return repr(lmlsco))
    (context === :inline) && (return "A linear surrogate objective in coordinates for the Levenberg Marquardt algorithm based on $(status_summary(lmlsco.objective; context = context)) with penalty $(lmlsco.penalty)")
    return """
    A linear surrogate objective in coordinates for the Levenberg Marquardt Algorithm

    ## Objective
    $(_in_str(status_summary(lmlsco.objective, context = context); indent = 1))

    ## Parameters
    * basis:     $(_MANOPT_INDENT)$(lmlsco.basis)
    * mode:      $(_MANOPT_INDENT)$(lmlsco.mode)
    * penalty:   $(_MANOPT_INDENT)$(lmlsco.penalty)
    * threshold: $(_MANOPT_INDENT)$(lmlsco.threshold)
    """
end

#
#
# ---
@doc """
    LevenbergMarquardtLinearSurrogateObjective{VF<:AbstractManifoldFirstOrderObjective{E}, R} <: AbstractLevenbergMarquardtLinearSurrogateObjective

The linear surrogate objective for a [`ManifoldNonlinearLeastSquaresObjective`](@ref).

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
and set ``y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p)`` and ``$(_tex(:Cal, "L"))(X) = CJ_F(p)[X]``
with

```math
C = $(_tex(:sqrt, "ρ'(p)"))(I-αP), $(_tex(:qquad)) P = $(_tex(:frac, "F(p)F(p)^" * _tex(:rm, "T"), _tex(:norm, "F(p)"; index = "2") * "^2")),
```

where ``F(p) ∈ ℝ^n`` is the vector of residuals at point ``p ∈ M`` and ``J_F(p): $(_math(:TangentSpace)) → ℝ^n``
is the Jacobian.
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

    LevenbergMarquardtLinearSurrogateObjective(objective; penalty::Real = 1e-6, threshold::Real = 1e-4, mode::Symbol = :Strict)
"""
mutable struct LevenbergMarquardtLinearSurrogateObjective{
        R <: Real, TO <: ManifoldNonlinearLeastSquaresObjective, TVC <: AbstractVector{R},
    } <: AbstractLevenbergMarquardtLinearSurrogateObjective
    objective::TO
    penalty::R
    threshold::R
    mode::Symbol
    value_cache::TVC
    function LevenbergMarquardtLinearSurrogateObjective(
            objective::ManifoldNonlinearLeastSquaresObjective;
            penalty::R = 1.0e-6, threshold::R = 1.0e-4, mode::Symbol = :Strict,
            residuals::TVC = zeros(residuals_count(get_objective(objective))),
        ) where {R <: Real, TVC <: AbstractVector}
        return new{R, typeof(objective), TVC}(objective, penalty, threshold, mode, residuals)
    end
end

"""
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime::Real, ρ_double_prime::Real, FSq::Real, threshold::Real=1.0e-5, mode::Symbol=:Strict)

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
- `:Normal` keeps negative ``ρ''(s) < 0`` but makes sure the square root is well-defined.
- `:Strict` (default) set ``α = 0`` when ``ρ''(s) < 0`` or when ``s = 0``
"""
function get_LevenbergMarquardt_scaling(
        ρ_prime::Real, ρ_double_prime::Real, FkSq::Real,
        threshold::Real = 1.0e-5, mode::Symbol = :Strict
    )
    # second derivative existent and negative: In strict mode (motivated by ceres) -> return sqrt(ρ_prime), 0
    ((ρ_double_prime < 0 && mode == :Strict)) && return (sqrt(ρ_prime), 0.0)
    (iszero(FkSq) && mode == :Strict) && return (sqrt(ρ_prime), 0.0)
    α = 1 - sqrt(max(1 + 2 * (ρ_double_prime / ρ_prime) * FkSq, 0.0))
    α = min(α, 1 - threshold)
    residual_scaling = sqrt(ρ_prime) / (1 - α)
    operator_scaling = ifelse(iszero(FkSq), 0.0, α / FkSq)
    return residual_scaling, operator_scaling
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
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, ::ZeroVector
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
    (penalty != 0) && (_diagview(A) .+= penalty)
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

function set_parameter!(lmlso::LevenbergMarquardtLinearSurrogateObjective, ::Val{:Penalty}, penalty::Real)
    lmlso.penalty = penalty
    return lmlso
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

#
#
# ---
"""
    NormalEquationsObjective{O <: AbstractLinearSurrogateObjective} <: AbstractSymmetricLinearSystemObjective

A [`AbstractLinearSurrogateObjective`](@ref) might be overdetermined, and it usually is overdetermined,
e.g. for the case of the [`LevenbergMarquardt`](@ref) algorithm.
For this case, one considers the [normal equations](https://en.wikipedia.org/wiki/Non-linear_least_squares).

This wrapper provides the same three functions as the wrapped surrogate

* [`get_linear_operator`](@ref) to compute/evaluate the linear operator ``$(_tex(:Cal, "L"))``
* [`get_vector_field`](@ref) to compute/evaluate the vector ``y``
* [`get_objective`](@ref) to provide access to the underlying surrogate

so that we obtain a symmetric linear system of equations, that can be
* solved with an iterative method like [`conjugate_gradient_descent`](@ref) or [`conjugate_residual`](@ref)
* solved as a linear system in a basis of the corresponding tangent space.
"""
struct NormalEquationsObjective{O <: AbstractLinearSurrogateObjective} <: AbstractSymmetricLinearSystemObjective
    objective::O
end

"""
    get_cost(TpM::TangentSpace, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, X)

Compute the surrogate cost when solving its normal equation, see also
[`get_cost(::AbstractManifold, ::LevenbergMarquardtLinearSurrogateObjective, p, X)`](@ref),
[`get_linear_operator`](@ref), and [`get_vector_field`](@ref) for more details.
"""
function get_cost(
        TpM::TangentSpace, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, X
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_cost(M, neo.objective, p, X)
end
function get_cost(
        TpM::TangentSpace, lnsco::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateCoordinatesObjective},
        ::ZeroVector
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    n = residuals_count(lnsco.objective.objective)
    vf = zeros(number_eltype(p), n)
    get_vector_field!(M, vf, lnsco.objective, p)
    return 0.5 * norm(vf)^2
end
function get_cost(
        TpM::TangentSpace, lnsco::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateCoordinatesObjective},
        X,
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    cX = get_coordinates(M, p, X)
    n = residuals_count(lnsco.objective.objective)
    vf = zeros(number_eltype(p), n)
    get_vector_field!(M, vf, lnsco.objective, p)
    add_linear_operator_coord!(TpM, vf, lnsco.objective, p, cX)
    cost = 0.5 * norm(vf)^2
    cost += (lnsco.objective.penalty / 2) * norm(M, p, X)^2
    return cost
end

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
        M::AbstractManifold, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, p, XB
    )
    return get_normal_linear_operator(M, neo.objective, p, XB)
end
@doc "$(_doc_linOp_NEO)"
function get_linear_operator!(
        M::AbstractManifold, YA, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, p, XB
    )
    return get_normal_linear_operator!(M, YA, neo.objective, p, XB)
end
# (b) coefficients in a basis
function get_linear_operator(
        M::AbstractManifold, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, p, c, B::AbstractBasis
    )
    return get_normal_linear_operator(M, neo.objective, p, c, B)
end
function get_linear_operator!(
        M::AbstractManifold, Y, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, p, c, B::AbstractBasis
    )
    return get_normal_linear_operator!(M, Y, neo.objective, p, c, B)
end
function get_linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateCoordinatesObjective}, p, B::AbstractBasis;
        penalty::Real = neo.objective.penalty,
    )
    return get_normal_linear_operator!(M, A, neo.objective, p, B; penalty = penalty)
end

function get_vector_field!(
        M::AbstractManifold, c, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateCoordinatesObjective}, p, B::AbstractBasis
    )
    get_normal_vector_field_coord!(M, c, neo.objective, p)
    c .*= -1
    return c
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
        M::AbstractManifold, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, p
    )
    return -get_normal_vector_field(M, neo.objective, p)
end
@doc "$(_doc_vecField_NEO)"
function get_vector_field!(
        M::AbstractManifold, Y, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, p
    )
    get_normal_vector_field!(M, Y, neo.objective, p)
    Y .*= -1
    return Y
end
# RHS in coordinates
function get_vector_field(
        M::AbstractManifold, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, p, B::AbstractBasis
    )
    return -get_normal_vector_field(M, neo.objective, p, B)
end
function get_vector_field!(
        M::AbstractManifold, c, neo::NormalEquationsObjective{<:LevenbergMarquardtLinearSurrogateObjective}, p, B::AbstractBasis
    )
    get_normal_vector_field!(M, c, neo.objective, p, B)
    c .*= -1
    return c
end

function show(io::IO, neo::NormalEquationsObjective)
    print(io, "NormalEquationsObjective(")
    print(io, neo.objective)
    return print(io, ")")
end

function status_summary(neo::NormalEquationsObjective; context::Symbol = :default)
    (context === :short) && return repr(neo)
    (context === :inline) && return "Normal equation objective for the objective $(status_summary(neo.objective; context = context))"
    return """
    A Normal equation objective to be used within Levenberg Marquardt to solve the surrogate

    ## Objective
    $(_in_str(status_summary(neo.objective; context = context); headers = 1, indent = 1))"""
end

get_objective(slsmo::NormalEquationsObjective) = slsmo.objective

function set_parameter!(neo::NormalEquationsObjective, e::Val, value)
    set_parameter!(neo.objective, e, value)
    return neo
end


@doc """
    TrustRegionModelObjective{O<:AbstractManifoldHessianObjective} <: AbstractManifoldSubObjective{O}

A trust region model of the form

```math
    m(X) = f(p) + ⟨$(_tex(:grad)) f(p), X⟩_p + $(_tex(:frac, "1", "2")) ⟨$(_tex(:Hess)) f(p)[X], X⟩_p
```

# Fields

* `objective`: an [`AbstractManifoldHessianObjective`](@ref) proving ``f``, its gradient and Hessian

# Constructors

    TrustRegionModelObjective(objective)

with either an [`AbstractManifoldHessianObjective`](@ref) `objective` or an decorator containing such an objective
"""
struct TrustRegionModelObjective{
        O <: Union{ManifoldHessianObjective, AbstractDecoratedManifoldObjective},
    } <: AbstractManifoldSubObjective{O}
    objective::O
end
get_objective(trmo::TrustRegionModelObjective) = trmo.objective

@doc """
    get_cost(TpM, trmo::TrustRegionModelObjective, X)

Evaluate the tangent space [`TrustRegionModelObjective`](@ref)

```math
m(X) = f(p) + ⟨$(_tex(:grad)) f(p), X ⟩_p + $(_tex(:frac, "1", "2")) ⟨$(_tex(:Hess)) f(p)[X], X⟩_p.
```
"""
function get_cost(TpM::TangentSpace, trmo::TrustRegionModelObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    c = get_objective_cost(M, trmo, p)
    G = get_objective_gradient(M, trmo, p)
    Y = get_objective_hessian(M, trmo, p, X)
    return c + inner(M, p, G, X) + 1 / 2 * inner(M, p, Y, X)
end
@doc """
    get_gradient(TpM, trmo::TrustRegionModelObjective, X)

Evaluate the gradient of the [`TrustRegionModelObjective`](@ref)

```math
$(_tex(:grad)) m(X) = $(_tex(:grad)) f(p) + $(_tex(:Hess)) f(p)[X].
```
"""
function get_gradient(TpM::TangentSpace, trmo::TrustRegionModelObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_gradient(M, trmo, p) + get_objective_hessian(M, trmo, p, X)
end
function get_gradient!(TpM::TangentSpace, Y, trmo::TrustRegionModelObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    get_objective_hessian!(M, Y, trmo, p, X)
    Y .+= get_objective_gradient(M, trmo, p)
    return Y
end
@doc """
    get_hessian(TpM, trmo::TrustRegionModelObjective, X)

Evaluate the Hessian of the [`TrustRegionModelObjective`](@ref)

```math
$(_tex(:Hess)) m(X)[Y] = $(_tex(:Hess)) f(p)[Y].
```
"""
function get_hessian(TpM::TangentSpace, trmo::TrustRegionModelObjective, X, V)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_hessian(M, trmo, p, V)
end
function get_hessian!(TpM::TangentSpace, W, trmo::TrustRegionModelObjective, X, V)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_hessian!(M, W, trmo, p, V)
end

function Base.show(io::IO, trmo::TrustRegionModelObjective)
    print(io, "TrustRegionModelObjective(")
    print(io, trmo.objective)
    return print(io, ")")
end
function status_summary(trmo::TrustRegionModelObjective; context::Symbol = :default)
    (context === :short) && return repr(trmo)
    (context === :inline) && return "The (tangent space) model for the trust region solver for the objective $(status_summary(trmo.objective; context = context))"
    return """
    The trust region model for the sub problem in the tangent space

    ## Objective
    $(_in_str(status_summary(trmo.objective)))"""
end
