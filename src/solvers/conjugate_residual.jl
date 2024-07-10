@doc raw"""
    ConjugateResidualState{T,R,TStop<:StoppingCriterion} <: AbstractManoptSolverState


"""
mutable struct ConjugateResidualState{T,R,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    X::T
    r::T
    d::T
    Ar::T
    Ad::T
    α::R
    β::R
    stop::TStop
    function ConjugateResidualState(
        TpM::TangentSpace,
        slso::SymmetricLinearSystemObjective;
        X::T=rand(TpM),
        r::T=-get_gradient(TpM, slso, X), # fix
        d::T=r,
        Ar::T=get_hessian(TpM, slso, X, r), # fix
        Ad::T=Ar,
        α::R=0.0,
        β::R=0.0,
        stop::SC=StopAfterIteration(5) | StopWhenGradientNormLess(1e-8),
        kwargs...,
    ) where {T,R,SC<:StoppingCriterion}
        crs = new{T,R,SC}()
        crs.X = X
        crs.r = r
        crs.d = d
        crs.Ar = Ar
        crs.Ad = Ad
        crs.α = α
        crs.β = β
        crs.stop = stop
        return crs
    end
end

get_iterate(crs::ConjugateResidualState) = crs.X
function set_iterate!(crs::ConjugateResidualState, ::AbstractManifold, X)
    crs.X = X
    return crs
end

get_gradient(crs::ConjugateResidualState) = -crs.r
function set_gradient!(crs::ConjugateResidualState, ::AbstractManifold, r)
    crs.r = -r
    return crs
end

function get_message(crs::ConjugateResidualState)
    return get_message(crs.α)
end

function show(io::IO, crs::ConjugateResidualState)
    i = get_count(crs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(crs.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Conjugate Residual Method
    $Iter
    ## Parameters
    * α: $(crs.α)
    * β: $(crs.β)

    ## Stopping criterion
    $(status_summary(crs.stop))

    This indicates convergence: $Conv
    """
    return print(io, s)
end

@doc raw"""
    conjugate_residual(TpM::TangentSpace, A, b, p=rand(M))
    conjugate_residual(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, p=rand(M))
    conjugate_residual!(TpM::TangentSpace, A, b, p)
    conjugate_residual!(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, p)

Compute the solution of ``\mathcal A[X] = b``, where

* ``\mathcal A`` is a linear operator on ``T_p\mathcal M``
* ``X, b ∈ T_p\mathcal M`` are tangent vectors.

This implementation follows Algorithm 3 in [LaiYoshise:2024](@cite).

"""
function conjugate_residual(
    TpM::TangentSpace,
    slso::SymmetricLinearSystemObjective,
    x0;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    y0 = copy(TpM, x0)
    return conjugate_residual!(TpM, slso, y0; evaluation=evaluation, kwargs...)
end

function conjugate_residual!(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective, x0; kwargs...
)
    crs = ConjugateResidualState(TpM, slso; kwargs...)
    dslso = decorate_objective!(TpM, slso; kwargs...)
    dmp = DefaultManoptProblem(TpM, dslso)
    crs = decorate_state!(crs; kwargs...)
    solve!(dmp, crs)
    return get_solver_return(get_objective(dmp), crs)
end

function initialize_solver!(
    amp::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState
)
    p = get_manifold(amp).point
    crs.X = rand(get_manifold(amp))
    crs.r = -get_gradient(amp, crs.X)
    crs.d = crs.r
    crs.Ar = get_hessian(amp, crs.X, crs.r)
    crs.Ad = crs.Ar
    crs.α = 0.0
    crs.β = 0.0

    return crs
end

function step_solver!(
    amp::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState, i
)
    TpM = get_manifold(amp)
    M = base_manifold(TpM)
    p = TpM.point
    # store current values (RB:) These are just references, nothing is copied here.
    # ...so we could also just write crs. upfront in the following formulae
    r = crs.r
    d = crs.d
    Ar = crs.Ar
    Ad = crs.Ad

    crs.α = inner(M, p, r, Ar) / inner(M, p, Ad, Ad)

    crs.X += crs.α * d

    crs.r -= crs.α * Ad
    crs.Ar = get_hessian(amp, crs.X, crs.r)
    crs.β = inner(M, p, crs.r, crs.Ar) / inner(M, p, r, Ar)
    crs.d = crs.r + crs.β * d
    crs.Ad = crs.Ar + crs.β * Ad
    return crs
end

get_solver_result(crs::ConjugateResidualState) = crs.X
