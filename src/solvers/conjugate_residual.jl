@doc raw"""
    ConjugateResidualState{T,R,TStop<:StoppingCriterion} <: AbstractManoptSolverState

A state for the [`conjugate_residual`](@ref) solver.

# Fields

* `X::T`: the iterate
* `r::T`: the residual ``r = -b(p) - \mathcal A(p)[X]``
* `d::T`: the conjugate direction
* `Ar::T`, `Ad::T`: storages for ``\mathcal A``
* `rAr::R`: internal field for storing ``⟨ r, \mathcal A(p)[r] ⟩``
* `α::R`: a step length
* `β::R`: the conjugate coefficient
* `stop::TStop`: a [`StoppingCriterion`] for the solver

# Constructor

        function ConjugateResidualState(
            TpM::TangentSpace,
            slso::SymmetricLinearSystemObjective;
            X=rand(TpM),
            r=-get_gradient(TpM, slso, X),
            d=copy(TpM, r),
            Ar=get_hessian(TpM, slso, X, r),
            Ad=copy(TpM, Ar),
            α::R=0.0,
            β::R=0.0,
            stopping_criterion=StopAfterIteration(manifold_dimension(TpM)) |
                               StopWhenGradientNormLess(1e-8),
            kwargs...,
    )

    Initialise the state with default values.
"""
mutable struct ConjugateResidualState{T,R,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    X::T
    r::T
    d::T
    Ar::T
    Ad::T
    rAr::R
    α::R
    β::R
    stop::TStop
    function ConjugateResidualState(
        TpM::TangentSpace,
        slso::SymmetricLinearSystemObjective;
        X::T=rand(TpM),
        r::T=-get_gradient(TpM, slso, X),
        d::T=copy(TpM, r),
        Ar::T=get_hessian(TpM, slso, X, r),
        Ad::T=copy(TpM, Ar),
        α::R=0.0,
        β::R=0.0,
        stopping_criterion::SC=StopAfterIteration(manifold_dimension(TpM)) |
                               StopWhenGradientNormLess(1e-8),
        kwargs...,
    ) where {T,R,SC<:StoppingCriterion}
        M = base_manifold(TpM)
        p = base_point(TpM)
        crs = new{T,R,SC}()
        crs.X = X
        crs.r = r
        crs.d = d
        crs.Ar = Ar
        crs.Ad = Ad
        crs.α = α
        crs.β = β
        crs.rAr = zero(R)
        crs.stop = stopping_criterion
        return crs
    end
end

get_iterate(crs::ConjugateResidualState) = crs.X
function set_iterate!(crs::ConjugateResidualState, ::AbstractManifold, X)
    crs.X = X
    return crs
end

get_gradient(crs::ConjugateResidualState) = crs.r
function set_gradient!(crs::ConjugateResidualState, ::AbstractManifold, r)
    crs.r = r
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
    conjugate_residual(TpM::TangentSpace, A, b, p=rand(TpM))
    conjugate_residual(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, p=rand(TpM))
    conjugate_residual!(TpM::TangentSpace, A, b, p)
    conjugate_residual!(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, p)

Compute the solution of ``\mathcal A(p)[X] = -b(p)``, where

* ``\mathcal A`` is a linear operator on ``T_p\mathcal M``
* ``b`` is a vector field on the manifold
* ``X ∈ T_p\mathcal M`` are tangent vectors.

This implementation follows Algorithm 3 in [LaiYoshise:2024](@cite) and
is initalised with ``X^{(0)}`` as

* the initial residual ``r^{(0)} = -b(p) - \mathcal A(p)[X^{(0)}]``
* the initial conjugate direction ``d^{(0)} = r^{(0)}``
* initialize ``Y^{(0)} = \mathcal A(p)[X^{(0)}]``

performed the following steps at iteration ``k=0,…`` until the `stopping_criterion=` is fulfilled.

1. compute a step size ``α_k = \displaystyle\frac{\langle r^{(k)}, \mathcal A(p)[r^{(k)}] \rangle_p}{\langle \mathcal A(p)[d^{(k)}], \mathcal A(p)[d^{(k)}] \rangle_p}``
2. do a step ``X^{(k+1)} = X^{(k)} + α_kd^{(k)}``
2. update the residual ``r^{(k+1)} = r^{(k)} + α_k Y^{(k)}``
4. compute ``Z = \mathcal A(p)[r^{(k+1)}]``
5. Update the conjugate coefficient ``β_k = \displaystyle\frac{\langle r^{(k+1)}, \mathcal A(p)[r^{(k+1)}] \rangle_p}{\langle r^{(k)}, \mathcal A(p)[r^{(k)}] \rangle_p}``
6. Update the conjugate direction ``d^{(k+1)} = r^{(k+1)} + β_kd^{(k)}``
7. Update  ``Y^{(0)} = -Z + β_k Y^{(k})``, the evaluated ``\mamthcal A[d^{(k)]``

# Input

* `TpM` the [`TangentSpace`](@extref `ManifoldsBase.TangentSpace`) as the domain
* `A` a symmetric linear operator on the tangent space `(M, p, X) -> Y`
* `b` a vector field on the tangent space `(M, p) -> X`


# Keyword arguments

* `evaluation=`[`AllocatingEvaluation`](@ref) specify whether `A` and `b` are implemented allocating or in-place
* `stopping_criterion::`[`StoppingCriterion`](@ref)`=`[`StopAfterIteration`](@ref)`(`[`manifold_dimension`](@extref ManifoldsBase.manifold_dimension-Tuple{AbstractManifold})`(TpM))`[` | `](@ref StopWhenAny)[`StopWhenGradientNormLess`](@ref)`(1e-8)`

# Output

the obtained (approximate) minimizer ``X^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details.
"""
conjugate_residual(TpM::TangentSpace, args...; kwargs...)

function conjugate_residual(
    TpM::TangentSpace,
    A,
    b,
    X=zero_vector(TpM);
    evaluation::AbstractEvaluationType=AllocatingEvaluation,
    kwargs...,
)
    slso = SymmetricLinearSystemObjective(A, b; evaluation=evaluation, kwargs...)
    return conjugate_residual(TpM, slso, X; evaluation=evaluation, kwargs...)
end
function conjugate_residual(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective, X=zero_vector(TpM); kwargs...
)
    Y = copy(TpM, X)
    return conjugate_residual!(TpM, slso, Y; kwargs...)
end

function conjugate_residual!(
    TpM::TangentSpace,
    slso::SymmetricLinearSystemObjective,
    x0;
    stopping_criterion::SC=StopAfterIteration(manifold_dimension(TpM)) |
                           StopWhenGradientNormLess(1e-8),
    kwargs...,
) where {SC<:StoppingCriterion}
    crs = ConjugateResidualState(
        TpM, slso; stopping_criterion=stopping_criterion, kwargs...
    )
    dslso = decorate_objective!(TpM, slso; kwargs...)
    dmp = DefaultManoptProblem(TpM, dslso)
    crs = decorate_state!(crs; kwargs...)
    solve!(dmp, crs)
    return get_solver_return(get_objective(dmp), crs)
end

function initialize_solver!(
    amp::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState
)
    TpM = get_manifold(amp)
    get_hessian!(TpM, crs.r, get_objective(amp), base_point(TpM), crs.X)
    crs.r *= -1
    crs.r .-= get_b(TpM, get_objective(amp), crs.X)
    copyto!(TpM, crs.d, crs.r)
    get_hessian!(amp, crs.Ar, crs.X, crs.r)
    copyto!(TpM, crs.Ad, crs.Ar)
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
    crs.α = inner(M, p, crs.r, crs.Ar) / inner(M, p, crs.Ad, crs.Ad)
    crs.X += crs.α * crs.d
    crs.rAr = inner(M, p, crs.r, crs.Ar)
    crs.r -= crs.α * crs.Ad
    get_hessian!(amp, crs.Ar, crs.X, crs.r)
    crs.β = inner(M, p, crs.r, crs.Ar) / crs.rAr
    crs.d = crs.r + crs.β * crs.d
    crs.Ad = crs.Ar + crs.β * crs.Ad
    return crs
end

get_solver_result(crs::ConjugateResidualState) = crs.X
