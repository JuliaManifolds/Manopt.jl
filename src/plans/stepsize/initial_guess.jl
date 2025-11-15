"""
    AbstractInitialLinesearchGuess

An abstract type for initial line search guess strategies. These are functors that map
`(problem, state, k, last_stepsize, η) -> α_0`, where `α_0` is the initial step size,
based on

* an [`AbstractManoptProblem`](@ref) `problem`
* an [`AbstractManoptSolverState`](@ref) `state`
* the current iterate `k`
* the last step size `last_stepsize`
* the search direction `η`
"""
abstract type AbstractInitialLinesearchGuess end

"""
    ConstantInitialGuess{TF} <: AbstractInitialLinesearchGuess

Implement a constant initial guess for line searches.

# Constructor

    ConstantInitialGuess(α::TF)

where `α` is the constant initial step size.
"""
struct ConstantInitialGuess{TF} <: AbstractInitialLinesearchGuess
    α::TF
end
ConstantInitialGuess() = ConstantInitialGuess(1.0)

function (cig::ConstantInitialGuess)(
        ::AbstractManoptProblem, ::AbstractManoptSolverState, ::Int, ::Real, η; kwargs...
    )
    return cig.α
end

"""
    ArmijoInitialGuess <: AbstractInitialLinesearchGuess

Implement the initial guess for an Armijo line search.

The initial step size is chosen as `min(l, max_stepsize(M, p) / norm(M, p, η))`,
where `l` is the last step size used, `p` the current point and `η` the search direction.

The default provided is based on the [`max_stepsize`](@ref)`(M)`.

# Constructor

    ArmijoInitialGuess()
"""
struct ArmijoInitialGuess <: AbstractInitialLinesearchGuess end

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

_doc_stepsize_initial_guess_field = """
* `initial_guess`: a function to provide an initial guess for the step size,
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

"""
    default_point_distance(::AbstractManifold, p)

The default Hager-Zhang guess for distance between `p` the solution to the optimization
problem. The default is 0, which deactivates heuristic I0 (a).
On each manifold with `default_point_distance`, you need to also implement `default_vector_norm`.
"""
default_point_distance(::AbstractManifold, p) = zero(number_eltype(p))

"""
    default_point_distance(::DefaultManifold, p)

Following [HagerZhang:2006:2](@cite), the expected distance to the optimal solution from `p`
on `DefaultManifold` is the `Inf` norm of `p`.
"""
default_point_distance(::DefaultManifold, p) = norm(p, Inf)

"""
    default_point_distance(::AbstractManifold, p)

The default Hager-Zhang guess for distance between `p` the solution to the optimization
problem along the descent direction. There is no default implementation because it is only
needed for manifolds with a specific `default_point_distance` method.
"""
default_vector_norm(M::AbstractManifold, p, X)
default_vector_norm(::DefaultManifold, p, X) = norm(p, Inf)


"""
    HagerZhangInitialGuess{TF <: Real, TPN, TVN} <: AbstractInitialLinesearchGuess

Initial line search guess from the paper [HagerZhang:2006:2](@cite), following the procedure
`initial`. The line search was adapted to the Riemannian setting by introducing
customizable norms for point and tangent vectors and maximum stepsize `alphamax`.
"""
struct HagerZhangInitialGuess{TF <: Real, TPN, TVN} <: AbstractInitialLinesearchGuess
    ψ0::TF
    ψ1::TF
    ψ2::TF
    constant_guess::TF
    quadstep::Bool
    point_distance::TPN
    vector_norm::TVN
    zero_abstol::TF
    alphamax::TF
end

HagerZhangInitialGuess() = HagerZhangInitialGuess{Float64}()
function HagerZhangInitialGuess{TF}(;
        ψ0::TF = 0.01,
        ψ1::TF = 0.01,
        ψ2::TF = 2.0,
        constant_guess::TF = NaN,
        quadstep::Bool = true,
        point_distance::TPN = default_point_distance,
        vector_norm::TVN = default_vector_norm,
        zero_abstol::TF = eps(TF),
        alphamax::TF = Inf,
    ) where {TF <: Real, TPN, TVN}
    return HagerZhangInitialGuess{TF, TPN, TVN}(
        ψ0, ψ1, ψ2, constant_guess, quadstep,
        point_distance, vector_norm, zero_abstol, alphamax
    )
end

function (hzi::HagerZhangInitialGuess{TF})(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState,
        k::Int, last_stepsize::Real, η;
        lf0 = get_cost(mp, get_iterate(s)),
        Dlf0 = get_differential(mp, get_iterate(s), η),
    ) where {TF <: Real}
    M = get_manifold(mp)
    p = get_iterate(s)
    abs_lf0 = abs(lf0)

    alphamax = min(hzi.alphamax, max_stepsize(M, p))

    if k == 1
        pn = hzi.point_distance(M, p)
        # Step I0
        if isnan(hzi.constant_guess)
            if pn > hzi.zero_abstol
                # I0.(a)
                ηn = hzi.vector_norm(M, p, η)
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
            f_R = get_cost(mp, ManifoldsBase.retract_fused(M, p, η, step_R, default_retraction_method(M, typeof(p))))
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
