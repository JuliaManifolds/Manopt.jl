@doc raw"""
    conjugate_residual(TpM::TangentSpace, A, b, p=rand(TpM))
    conjugate_residual(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, p=rand(TpM))
    conjugate_residual!(TpM::TangentSpace, A, b, p)
    conjugate_residual!(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, p)

Compute the solution of ``\mathcal A(p)[X] + b(p) = 0_p ``, where

* ``\mathcal A`` is a linear, symmetric operator on ``T_p\mathcal M``
* ``b`` is a vector field on the manifold
* ``X ∈ T_p\mathcal M`` is a tangent vector
* ``0_p`` is the zero vector ``T_p\mathcal M``.

This implementation follows Algorithm 3 in [LaiYoshise:2024](@cite) and
is initalised with ``X^{(0)}`` as the zero vector and

* the initial residual ``r^{(0)} = -b(p) - \mathcal A(p)[X^{(0)}]``
* the initial conjugate direction ``d^{(0)} = r^{(0)}``
* initialize ``Y^{(0)} = \mathcal A(p)[X^{(0)}]``

performed the following steps at iteration ``k=0,…`` until the `stopping_criterion` is fulfilled.

1. compute a step size ``α_k = \displaystyle\frac{\langle r^{(k)}, \mathcal A(p)[r^{(k)}] \rangle_p}{\langle \mathcal A(p)[d^{(k)}], \mathcal A(p)[d^{(k)}] \rangle_p}``
2. do a step ``X^{(k+1)} = X^{(k)} + α_kd^{(k)}``
2. update the residual ``r^{(k+1)} = r^{(k)} + α_k Y^{(k)}``
4. compute ``Z = \mathcal A(p)[r^{(k+1)}]``
5. Update the conjugate coefficient ``β_k = \displaystyle\frac{\langle r^{(k+1)}, \mathcal A(p)[r^{(k+1)}] \rangle_p}{\langle r^{(k)}, \mathcal A(p)[r^{(k)}] \rangle_p}``
6. Update the conjugate direction ``d^{(k+1)} = r^{(k+1)} + β_kd^{(k)}``
7. Update  ``Y^{(k+1)} = -Z + β_k Y^{(k)}``

Note that the right hand side of Step 7 is the same as evaluating ``\mathcal A[d^{(k+1)]``, but avoids the actual evaluation

# Input

* `TpM` the [`TangentSpace`](@extref `ManifoldsBase.TangentSpace`) as the domain
* `A` a symmetric linear operator on the tangent space `(M, p, X) -> Y`
* `b` a vector field on the tangent space `(M, p) -> X`


# Keyword arguments

* `evaluation=`[`AllocatingEvaluation`](@ref) specify whether `A` and `b` are implemented allocating or in-place
* `stopping_criterion::`[`StoppingCriterion`](@ref)`=`[`StopAfterIteration`](@ref)`(`[`manifold_dimension`](@extref ManifoldsBase.manifold_dimension-Tuple{AbstractManifold})`(TpM))`[` | `](@ref StopWhenAny)[`StopWhenRelativeResidualLess`](@ref)`(c,1e-8)`,
  where `c` is the norm of ``\lVert b \rVert``.

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
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
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
    X;
    stopping_criterion::SC=StopAfterIteration(manifold_dimension(TpM)) |
                           StopWhenRelativeResidualLess(
        norm(base_manifold(TpM), base_point(TpM), get_b(TpM, slso, X)), 1e-8
    ),
    kwargs...,
) where {SC<:StoppingCriterion}
    crs = ConjugateResidualState(
        TpM, slso; stopping_criterion=stopping_criterion, kwargs...
    )
    dslso = decorate_objective!(TpM, slso; kwargs...)
    dmp = DefaultManoptProblem(TpM, dslso)
    dcrs = decorate_state!(crs; kwargs...)
    solve!(dmp, dcrs)
    return get_solver_return(get_objective(dmp), dcrs)
end

function initialize_solver!(
    amp::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState
)
    TpM = get_manifold(amp)
    get_hessian!(TpM, crs.r, get_objective(amp), base_point(TpM), crs.X)
    crs.r .*= -1
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
    crs.X .+= crs.α .* crs.d
    crs.rAr = inner(M, p, crs.r, crs.Ar)
    crs.r .-= crs.α .* crs.Ad
    get_hessian!(amp, crs.Ar, crs.X, crs.r)
    crs.β = inner(M, p, crs.r, crs.Ar) / crs.rAr
    crs.d .= crs.r .+ crs.β .* crs.d
    crs.Ad .= crs.Ar .+ crs.β .* crs.Ad
    return crs
end

get_solver_result(crs::ConjugateResidualState) = crs.X
