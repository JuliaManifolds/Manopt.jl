abstract type AbstractInitialLinesearchGuess end

struct ArmijoInitialGuess <: AbstractInitialLinesearchGuess end

"""
    (::ArmijoInitialGuess)(mp::AbstractManoptProblem, s::AbstractManoptSolverState, k, l, η; kwargs...)

# Input

* `mp`: the [`AbstractManoptProblem`](@ref) we are aiming to minimize
* `s`:  the [`AbstractManoptSolverState`](@ref) for the current solver
* `k`:  the current iteration
* `l`:  the last step size computed in the previous iteration.
* `η`:  the search direction

Return an initial guess for the [`ArmijoLinesearchStepsize`](@ref).

The default provided is based on the [`max_stepsize`](@ref)`(M)`, which we denote by ``m``.
Let further ``X`` be the current descent direction with norm ``n=$(_tex(:norm, "X"; index = "p"))`` its length.
Then this (default) initial guess returns

* ``l`` if ``m`` is not finite
* ``$(_tex(:min))(l, $(_tex(:frac, "m", "n")))`` otherwise

This ensures that the initial guess does not yield to large (initial) steps.
"""
function (::ArmijoInitialGuess)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, ::Int, l::Real, η; kwargs...
    )
    M = get_manifold(mp)
    X = get_gradient(s)
    p = get_iterate(s)
    grad_norm = norm(M, p, X)
    max_step = max_stepsize(M, p)
    return ifelse(isfinite(max_step), min(l, max_step / grad_norm), l)
end

_doc_stepsize_initial_guess(default = "") = """
* `initial_guess`$(length(default) > 0 ? " = $(default)" : ""): a function to provide an initial guess for the step size,
  it maps `(problem, state, k, last_stepsize, η) -> α_0` based on
  * a [`AbstractManoptProblem`](@ref) `problem`
  * a [`AbstractManoptSolverState`](@ref) `state`
  * the current iterate `k`
  * the last step size `last_stepsize`
  * the search direction `η`
  and should at least accept the keywords
  * `lf0 = `[`get_cost`](@ref)`(problem, get_iterate(state))` the current cost at ^p` here interpreted as the initial point of `f` along the line search direction`
  * `Dlf0 = `[`get_differential`](@ref)`(problem, get_iterate(state), η)` the directional derivative at point `p` in direction `η`

"""

default_point_norm(::AbstractManifold, p) = one(number_eltype(p))

default_vector_norm(::AbstractManifold, p, X) = norm(M, p, X)

"""
"""
Base.@kwdef mutable struct HagerZhangInitialGuess{TF <: Real, TPN, TVN} <: AbstractInitialLinesearchGuess
    ψ0::TF = 0.01
    ψ1::TF = 0.01
    ψ2::TF = 2.0
    constant_guess::TF = NaN
    quadstep::Bool = true
    point_norm::TPN = default_point_norm
    vector_norm::TVN = default_vector_norm
    zero_abstol::TF = eps(TF)
    alphamax::TF = Inf
end


"""
"""
function (hzi::HagerZhangInitialGuess{TF})(mp::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int, last_stepsize::Real, η; lf0, Dlf0) where {TF <: Real}
    M = get_manifold(mp)
    p = get_iterate(s)
    abs_lf0 = abs(lf0)

    alphamax = min(hzi.alphamax, max_stepsize(M, p))

    if k == 1
        pn = hzi.point_norm(M, p)
        ηn = hzi.vector_norm(M, p, η)
        # Step I0
        if isnan(hzi.constant_guess)
            if pn > hzi.zero_abstol
                # I0.(a)
                return min(hzi.ψ0 * pn / ηn, alphamax)
            elseif abs_lf0 > hzi.zero_abstol
                # I0.(b)
                return min(hzi.ψ0 * abs_lf0 / norm(M, p, η)^2, alphamax)
            else
                # I0.(c)
                return one(TF)
            end
        else
            return hzi.constant_guess
        end
    else
        if hzi.quadstep
            # attempt step I1
            step_R = hzi.ψ1 * last_stepsize
            f_R = get_cost(mp, p, step_R * η)
            # solving quadratic fit to the line given lf0, Dlf0 and cost at f_R
            q_b = Dlf0
            q_a = (f_R - q_b * step_R - lf0) / step_R^2

            if f_R ≤ lf0 && isfinite(q_a) && q_a > hzi.zero_abstol
                # if condition is false, we go to step I2
                a_min = -q_b / (2 * q_a)
                return min(a_min, alphamax)
            end
        end
        # step I2
        return min(hzi.ψ2 * last_stepsize, alphamax)
    end
end
