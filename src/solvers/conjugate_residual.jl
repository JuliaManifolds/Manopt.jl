mutable struct ConjugateResidualState{
    T,
    R,
    TStop<:StoppingCriterion
} <: AbstractManoptSolverState
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
        x::T;
        r::T = -get_gradient(TpM, mho, x),
        d::T = r,
        Ar::T = get_hessian(TpM, mho, x, r),
        Ad::T = Ar,
        α::R = inner(TpM, TpM.point, r, Ar) / inner(TpM, TpM.point, Ad, Ad),
        β::R = 0.0,
        stop::StoppingCriterion = StopAfterIteration(200) | StopWhenGradientNormLess(1e-8),
        kwargs...,
    ) where{T,R}
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
function set_iterate!(crs::ConjugateResidualState, ::AbstractManifold, p)
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
    * x: $(crs.x)
    * r: $(crs.r)
    * d: $(crs.d)
    * Ar: $(crs.Ar)
    * Ad: $(crs.Ad)
    * α: $(crs.α)
    * β: $(crs.β)

    ## Stopping criterion
    $(status_summary(crs.stop))

    ## Stepsize
    $(crs.α)

    This indicates convergence: $Conv
    """

    return print(io, s)
end

function conjugate_residual!(
    TpM::TangentSpace,
    mho::ManifoldHessianObjective,
    x;
    kwargs...
)
    crs = ConjugateResidualState(TpM, mho, x)
    dmho = decorate_objective!(TpM, mho; kwargs...)
    dmp = DefaultManoptProblem(TpM, dmho)
    crs = decorate_state!(crs; kwargs...)
    solve!(dmp, crs)
    return get_solver_return(get_objective(dmp), crs)
end

function initialize_solver!(::AbstractManoptProblem, crs::ConjugateResidualState)
    return crs
end

function step_solver!(amp::AbstractManoptProblem, crs::ConjugateResidualState, i)

    # I would propose to use something like
    # TpM = get_manifold(amp)
    # p = TpM.point
    # and juts the inner call below in the 2 cases
    #
    # this (a) just calls get_manifold once (and not 4 times)
    # and avoids a function definition in every step
    #
    # Besides that this whole file looks very good in style already!

    metric = (X, Y) -> inner(
        get_manifold(amp), get_manifold(amp).point, X, Y
    )

    A = x -> get_hessian(amp, crs.x, x)

    # store current values
    r = crs.r
    d = crs.d
    Ar = crs.Ar
    Ad = crs.Ad

    # update iterate and residual
    crs.x += crs.α * d
    crs.r -= crs.α * Ad

    # this is the only evaluation of A
    crs.Ar = A(crs.r)

    # update d and Ad
    crs.β = metric(crs.r, crs.Ar) / metric(r, Ar)
    crs.d = crs.r + crs.β * d
    crs.Ad = crs.Ar + crs.β * Ad

    return crs
end

get_solver_result(crs::ConjugateResidualState) = crs.x
