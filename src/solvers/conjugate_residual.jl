@doc raw"""
    ConjugateResidualState{T,R,TStop<:StoppingCriterion} <: AbstractManoptSolverState


"""
mutable struct ConjugateResidualState{T,R,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    X::T
    r::T
    r_old::T
    d::T
    Ar::T
    Ar_old::T
    Ad::T
    α::R
    β::R
    stop::TStop
    function ConjugateResidualState(
        TpM::TangentSpace,
        slso::SymmetricLinearSystemObjective;
        X::T=rand(TpM),
        r::T=get_gradient(TpM, slso, X), # fix
        d::T=r,
        Ar::T=get_hessian(TpM, slso, X, r), # fix
        Ad::T=Ar,
        α::R=0.0,
        β::R=0.0,
        stop::SC=StopAfterIteration(5) | StopWhenGradientNormLess(1e-8),
        kwargs...,
    ) where {T,R,SC<:StoppingCriterion}
        M = base_manifold(TpM)
        p = base_point(TpM)
        crs = new{T,R,SC}()
        crs.X = X
        crs.r = r
        crs.r_old = copy(M, p, r)
        crs.d = d
        crs.Ar = Ar
        crs.Ar_old = copy(M, p, Ar)
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

Compute the solution of ``\mathcal A[X] = -b``, where

* ``\mathcal A`` is a linear operator on ``T_p\mathcal M``
* ``X, b ∈ T_p\mathcal M`` are tangent vectors.

This implementation follows Algorithm 3 in [LaiYoshise:2024](@cite) and
is initalised with ``X^{(0)}`` as

* the initial residual ``r^{(0)} = X^{(0)} + \mathcal A[X^{(0)}]``
* the initial conjugate direction ``d^{(0)} = r^{(0)}``
* initialize ``Y^{(0)} = \mathcal A[X^{(0)}]

performed
the following steps at iteration ``k=0,…``:

1. compute a step size ``α_k = \frac{⟨ r^{(k)}, \mathcal A[r^{(k)}] ⟩_p}{\lVert \mathcal A[d^{(0)}] \rVert_p}
2. do a step ``X^{(k+1)} = X^{(k)} + α_kd^{(k)}
2. update the residual ``r^{(k+1)} = r^{(k)} + α_k Y^{(k)}``
4. compute ``Z = \mathcal A[r^{(k+1)}]``
5. Update the conjugate coefficient ``β_k = \frac{⟨ r^{(k+1)}, \mathcal A[r^{(k+1)}] ⟩_p}{⟨ r^{(k)}, \mathcal A[r^{(k)}] ⟩_p}
6. Update the conjugate direction ``d^{(k+1)} = -r^{(k+1)} + β_kd^{(k)}``
7. Update  ``Y^{(0)} = -Z + β_k Y^{(k})`` the evaluated ``\mamthcal A[d^{(k)]``
8. increase ``k`` to ``k+1``.
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
    M = base_manifold(get_manifold(amp))
    p = base_point(get_manifold(amp))
    crs.X = rand(get_manifold(amp))
    get_gradient!(amp, crs.r, crs.X)
    copyto!(M, p, crs.d, crs.r)
    copyto!(M, p, crs.r_old, crs.r)
    copyto!(M, p, crs.Ar_old, crs.Ar)
    crs.Ar = get_hessian(amp, crs.X, crs.r)
    copyto!(M, p, crs.Ad, crs.Ar)
    crs.α = 0.0
    crs.β = 0.0
    return crs
end

function step_solver!(
    amp::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState, i
)
    TpM = get_manifold(amp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    den = inner(M, p, crs.Ad, crs.Ad)
    crs.α = inner(M, p, crs.r, crs.Ar) / (den == 0 ? 1.0 : den)
    println("α:", crs.α, "(", den, ")")
    crs.X += crs.α * crs.d
    copyto!(M, crs.r_old, p, crs.r)
    crs.r -= crs.α * crs.Ad
    copyto!(M, crs.Ar_old, p, crs.Ar)
    crs.Ar = get_hessian(amp, crs.X, crs.r)
    den = inner(M, p, crs.r_old, crs.Ar_old)
    crs.β = inner(M, p, crs.r, crs.Ar) / (den == 0 ? 1.0 : den)
    crs.d = -crs.r + crs.β * crs.d
    crs.Ad = -crs.Ar + crs.β * crs.Ad
    return crs
end

get_solver_result(crs::ConjugateResidualState) = crs.X
