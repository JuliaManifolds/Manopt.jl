mutable struct LevenbergMarquardtBoxSubsolver{TSt <: AbstractManoptSolverState, F <: Real} <: AbstractManoptSolverState
    internal_state::TSt
    last_gcd_result::Symbol
    last_gcd_stepsize::F
end

"""
    hessian_value_diag(ha::LevenbergMarquardtBoxSubsolver, M, p, X::UnitVector)

Evaluate the quadratic form associated with the stored Hessian approximation.
"""
function hessian_value_diag(ha::LevenbergMarquardtBoxSubsolver, M::AbstractManifold, p, X)
    return hessian_value_diag(ha.internal_state, M, p, X)
end
"""
    hessian_value_diag(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X::UnitVector)

Evaluate the quadratic form associated with the Hessian approximation [`CoordinatesNormalSystemState`].
Returns the scalar ``c^{\top} A c`` where ``c`` are the coordinates of the
[`UnitVector`](@ref) `X` at `p` (in the basis `ha.basis`) and ``B`` is `ha.A`.
"""
function hessian_value_diag(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X::UnitVector)
    b = to_coordinate_index(M, X, ha.basis)
    return ha.A[b, b]
end
"""
    hessian_value_diag(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X)

Evaluate the quadratic form associated with the Hessian approximation [`CoordinatesNormalSystemState`].
Returns the scalar ``c^{\top} A c`` where ``c`` are the coordinates of `X` at `p`
(in the basis `ha.basis`) and ``A`` is `ha.A`.
"""
function hessian_value_diag(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X)
    c = get_coordinates(M, p, X, ha.basis)
    return dot(c, ha.A, c)
end

"""
    hessian_value(ha::LevenbergMarquardtBoxSubsolver, M, p, X::UnitVector, Y)

Evaluate the quadratic form associated with the stored Hessian approximation.
"""
function hessian_value(ha::LevenbergMarquardtBoxSubsolver, M::AbstractManifold, p, X::UnitVector, Y)
    return hessian_value(ha.internal_state, M, p, X, Y)
end

"""
    hessian_value(ha::CoordinatesNormalSystemState, M, p, X::UnitVector, Y)

Evaluate the quadratic form associated with the stored Hessian approximation.
Returns the scalar ``c_b^{\top} B c`` where ``c_b`` are the coordinates of the
[`UnitVector`](@ref) `X` at `p` (assumed to correspond to the basis `ha.basis`),
``c`` are the coordinates of the tangent vector `Y` at `p` (in the basis `ha.basis`)
and ``B`` is `ha.A`.
"""
function hessian_value(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X::UnitVector, Y)
    b = to_coordinate_index(M, X, ha.basis)
    return dot(view(ha.A, b, :), get_coordinates(M, p, Y, ha.basis))
end

function initialize_solver!(amp::AbstractManoptProblem, dss::LevenbergMarquardtBoxSubsolver)
    initialize_solver!(amp, dss.internal_state)
    return dss
end
function stop_solver!(amp::AbstractManoptProblem, ams::LevenbergMarquardtBoxSubsolver, k)
    return stop_solver!(amp, ams.internal_state, k)
end


function LevenbergMarquardtBoxSubsolver(::AbstractManifold, sub_state_::AbstractManoptSolverState, p)
    return LevenbergMarquardtBoxSubsolver{typeof(sub_state_), number_eltype(p)}(
        sub_state_,
        :not_searched,
        NaN,
    )
end

function solve_LM_subproblem!(
        M::AbstractManifold, X, p, problem::AbstractManoptProblem,
        state::LevenbergMarquardtBoxSubsolver, grad_Y,
    )
    solve!(problem, state.internal_state)
    copyto!(M, X, p, get_solver_result(problem, state.internal_state))
    X .*= -1
    # trim to box using GCD
    gcd = GeneralizedCauchyDirectionSubsolver(M, p, state)
    state.last_gcd_result, state.last_gcd_stepsize = find_generalized_cauchy_direction!(M, gcd, X, p, X, grad_Y)
    X .*= state.last_gcd_stepsize
    return X
end
