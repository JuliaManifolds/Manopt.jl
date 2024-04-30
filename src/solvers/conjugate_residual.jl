mutable struct ConjugateResidualState{T,R,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    x::T
    r::T
    d::T
    Ar::T
    Ad::T
    α::R
    β::R
    stop::TStop
    function ConjugateResidualState(
        TpM::TangentSpace,
        mho::ManifoldHessianObjective,
        x::T=rand(TpM);
        r::T=-get_gradient(TpM, mho, x),
        d::T=r,
        Ar::T=get_hessian(TpM, mho, x, r),
        Ad::T=Ar,
        α::R=0.0,
        β::R=0.0,
        stop::StoppingCriterion=StopAfterIteration(200) | StopWhenGradientNormLess(1e-8),
        kwargs...,
    ) where {T,R}
        crs = new{T,R,typeof(stop)}()
        crs.x = x
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

get_iterate(crs::ConjugateResidualState) = crs.x
function set_iterate!(crs::ConjugateResidualState, ::AbstractManifold, x)
    crs.x = x
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

function conjugate_residual(
    TpM::TangentSpace,
    A,
    b,
    x0;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    M = base_manifold(TpM)
    p = TpM.point
    mho = ManifoldHessianObjective(
        (TpM, x) -> 0.5 * inner(M, p, x, A * x) - inner(M, p, b, x),
        (TpM, x) -> A * x - b,
        (TpM, x, y) -> A * y,
    )

    return conjugate_residual(TpM, mho, x0; evaluation=evaluation, kwargs...)
end

function conjugate_residual(
    TpM::TangentSpace,
    mho::ManifoldHessianObjective,
    x0;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    y0 = copy(TpM, x0)
    return conjugate_residual!(TpM, mho, y0; evaluation=evaluation, kwargs...)
end

function conjugate_residual!(
    TpM::TangentSpace, mho::ManifoldHessianObjective, x0; kwargs...
)
    crs = ConjugateResidualState(TpM, mho, x0; kwargs...)
    dmho = decorate_objective!(TpM, mho; kwargs...)
    dmp = DefaultManoptProblem(TpM, dmho)
    crs = decorate_state!(crs; kwargs...)
    solve!(dmp, crs)
    return get_solver_return(get_objective(dmp), crs)
end

function initialize_solver!(
    ::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState
)
    # (RB:) Reset / update A, r, D, Ar, Ad α β
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

    # update iterate and residual
    crs.x += crs.α * d
    crs.r -= crs.α * Ad

    # this is the only evaluation of A
    crs.Ar = get_hessian(amp, crs.x, crs.r)

    # update d and Ad
    crs.β = inner(M, p, crs.r, crs.Ar) / inner(M, p, r, Ar)
    crs.d = crs.r + crs.β * d
    crs.Ad = crs.Ar + crs.β * Ad

    return crs
end

get_solver_result(crs::ConjugateResidualState) = crs.x
