"""
    AbstractRobustifierFunction <: Function

An abstract type to represent robustifiers, i.e., functions ``ρ: ℝ → ℝ``,
currently mainly used within [Levenberg-Marquardt](@ref).

Usually these should be twice continuously differentiable functions with

* ``ρ(0) = 0`` and ``ρ'(0) = 1``, to mimic the classical least squares behavior around zero residuals
* ``ρ'(x) < 1`` in outlier regions
* ``ρ''(x) < 0`` in outlier regions

Note that the robustifier is applied to the squared residuals within the
nonlinear least squares framework, i.e., ``ρ(f_i(p)^2)``.
"""
abstract type AbstractRobustifierFunction <: Function end

function get_robustifier_values end

"""
    (a, b, c) = get_robustifier_values(ρ::AbstractRobustifierFunction, x::Real)

Evaluate the robustifier ``ρ`` and its first two derivatives at `x`.

# Returns

A tuple `(a, b, c)` with
* `a = ρ(x)`
* `b = ρ'(x)`
* `c = ρ''(x)`
"""
get_robustifier_values(ρ::AbstractRobustifierFunction, x::Real)
