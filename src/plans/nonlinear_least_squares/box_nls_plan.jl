mutable struct LevenbergMarquardtBoxSubsolver{TSt <: AbstractManoptSolverState, F <: Real} <: AbstractManoptSolverState
    internal_state::TSt
    last_gcd_result::Symbol
    last_gcd_stepsize::F
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
    gcd = GeneralizedCauchyDirectionSubsolver(M, p, X)
    state.last_gcd_result, state.last_gcd_stepsize = find_generalized_cauchy_direction!(M, gcd, X, p, X, grad_Y)
    return X
end
