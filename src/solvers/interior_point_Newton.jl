# struct for state of interior point algorithm
mutable struct InteriorPointState{
    P,
    Pr<:AbstractManoptProblem,
    St<:AbstractManoptSolverState,
    T,
    R,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    TStepsize<:Stepsize,
} <: AbstractGradientSolverState
    p::P
    sub_problem::Pr
    sub_state::St
    X::T # not sure if needed?
    μ::T
    λ::T
    s::T
    ρ::R
    σ::R
    stop::TStop
    retraction_method::TRTM
    stepsize::TStepsize
    function InteriorPointState(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        p::P,
        sub_problem::Pr,
        sub_state::St;
        X::T=get_gradient(M, cmo, p), # not sure if needed?
        μ::T=ones(length(get_inequality_constraints(M, cmo, p))),
        λ::T=zeros(length(get_equality_constraints(M, cmo, p))),
        s::T=ones(length(get_inequality_constraints(M, cmo, p))),
        ρ::R=μ's / length(get_inequality_constraints(M, cmo, p)),
        σ::R=calculate_σ(M, cmo, p, μ, λ, s),
        stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-8),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        stepsize::Stepsize=ArmijoLinesearch(
            M; retraction_method=retraction_method, initial_stepsize=1.0),
        kwargs...,
    ) where {P,Pr,St,T,R}
        ips = new{P,typeof(sub_problem),typeof(sub_state),T,R,typeof(stop),typeof(retraction_method),typeof(stepsize)}()
        ips.p = p
        ips.sub_problem = sub_problem
        ips.sub_state = sub_state
        ips.X = X
        ips.μ = μ
        ips.λ = λ
        ips.s = s
        ips.ρ = ρ
        ips.σ = σ
        ips.stop = stop
        ips.retraction_method = retraction_method
        ips.stepsize = stepsize
        return ips
    end
end

# get & set iterate
get_iterate(ips::InteriorPointState) = ips.p
function set_iterate!(ips::InteriorPointState, ::AbstractManifold, p)
    ips.p = p
    return ips
end
# get & set gradient (not sure if needed?)
get_gradient(ips::InteriorPointState) = ips.X
function set_gradient!(ips::InteriorPointState, ::AbstractManifold, X)
    ips.X = X
    return ips
end
# only message on stepsize for now
function get_message(ips::InteriorPointState)
    return get_message(ips.stepsize)
end
# pretty print state info
function show(io::IO, ips::InteriorPointState)
    i = get_count(ips, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ips.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Interior Point Newton Method
    $Iter
    ## Parameters
    * ρ: $(ips.ρ)
    * σ: $(ips.σ)
    ## Stopping criterion
    $(status_summary(ips.stop))
    * retraction method: $(ips.retraction_method)
    ## Stepsize
    $(ips.stepsize)
    This indicates convergence: $Conv
    """
    return print(io, s)
end

# not-in-place,
# takes M, f, grad_f, Hess_f and possibly constraint functions and their graidents
function interior_point_Newton(
    M::AbstractManifold,
    f, grad_f, Hess_f, p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing, h=nothing, grad_g=nothing, grad_h=nothing,
    kwargs...,)
    q = copy(M, p)
    mho = ManifoldHessianObjective(f, grad_f, Hess_f)
    cmo = ConstrainedManifoldObjective(mho, g, grad_g, h, grad_h; evaluation=evaluation)
    return interior_point_Newton!(M, cmo, q; evaluation=evaluation, kwargs...)
end

# not-in-place
# case where dim(M) = 1 and in particular p is a number
function interior_point_Newton(
    M::AbstractManifold,
    f, grad_f, Hess_f,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing, grad_g=nothing, grad_h=nothing, h=nothing,
    kwargs...,)

    q = [p]
    f_(M, p) = f(M, p[])

    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    Hess_f_ = _to_mutating_gradient(Hess_f, evaluation)

    g_ = isnothing(g) ? nothing : (M, p) -> g(M, p[])
    grad_g_ = isnothing(grad_g) ? nothing : _to_mutating_gradient(grad_g, evaluation)
    h_ = isnothing(h) ? nothing : (M, p) -> h(M, p[])
    grad_h_ = isnothing(grad_h) ? nothing : _to_mutating_gradient(grad_h, evaluation)

    mho = ManifoldHessianObjective(f_, grad_f_, Hess_f_)
    cmo = ConstrainedManifoldObjective(mho, g_, grad_g_, h_, grad_h_; evaluation=evaluation)

    rs = interior_point_Newton(M, cmo, q; evaluation=evaluation, kwargs...)

    return (typeof(q) == typeof(rs)) ? rs[] : rs
end

# not-in-place
# takes only manifold, constrained objetive and initial point
function interior_point_Newton(
    M::AbstractManifold, cmo::O, p; kwargs...
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return interior_point_Newton!(M, cmo, q; kwargs...)
end

# in-place
# takes M, f, grad_f, Hess_f and possibly constreint functions and their graidents
function interior_point_Newton!(
    M::AbstractManifold, f, grad_f, Hess_f, p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing, h=nothing, grad_g=nothing, grad_h=nothing,
    kwargs...,)

    mho = ManifoldHessianObjective(f, grad_f, Hess_f)
    cmo = ConstrainedManifoldObjective(mho, g, grad_g, h, grad_h; evaluation=evaluation)
    dcmo = decorate_objective!(M, cmo; kwargs...)

    return interior_point_Newton!(M, dcmo, p; evaluation=evaluation, kwargs...)
end

# MAIN SOLVER
function interior_point_Newton!(
    M::AbstractManifold,
    cmo::O,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    X = get_gradient(M, cmo, p),
    μ = ones(length(get_inequality_constraints(M, cmo, p))),
    λ = zeros(length(get_equality_constraints(M, cmo, p))),
    s = μ,
    ρ = μ's / length(get_inequality_constraints(M, cmo, p)),
    σ = calculate_σ(M, cmo, p, μ, λ, s),
    stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-5),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stepsize::Stepsize=ArmijoLinesearch(
        M; retraction_method=retraction_method, initial_stepsize=1.0),
    sub_kwargs=(;),
    sub_objective = decorate_objective!(
        TangentSpace(M, p) × ℝ^length(λ),
        SymmetricLinearSystemObjective(
            ReducedLagrangianHess(cmo, μ, λ, s),
            NegativeReducedLagrangianGrad(cmo, μ, λ, s, ρ*σ),
        ),
        sub_kwargs...,),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                              StopWhenGradientNormLess(1e-5),
    sub_state::AbstractManoptSolverState=decorate_state!(
        ConjugateResidualState(
            TangentSpace(M, p),
            sub_objective;
            stop = sub_stopping_criterion,
            sub_kwargs...,);
        sub_kwargs...,),
    sub_problem::AbstractManoptProblem=DefaultManoptProblem(
        TangentSpace(M, p), sub_objective),
    kwargs...,
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    !is_feasible(M, cmo, p) && throw(ErrorException("Starting point p must be feasible."))
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = InteriorPointState(
        M, cmo, p,
        sub_problem, sub_state;
        X=X, μ=μ, λ=λ, s=s,
        stop=stop,
        retraction_method=retraction_method,
        stepsize=stepsize,
        kwargs...)
    ips = decorate_state!(ips; kwargs...)
    solve!(dmp, ips)
    return get_solver_return(get_objective(dmp), ips)
end

# inititializer, might add more here
function initialize_solver!(::AbstractManoptProblem, ips::InteriorPointState)
    return ips
end

# step solver
function step_solver!(amp::AbstractManoptProblem, ips::InteriorPointState, i)
    M = get_manifold(amp)
    cmo = get_objective(amp)

    m, n = length(ips.μ), length(ips.λ)
    TpM = TangentSpace(M, ips.p)
    # TqN = TangentSpace(M × ℝ^n, ArrayPartition(ips.p, ips.λ))

    g = get_inequality_constraints(amp, ips.p)
    Jg = get_grad_inequality_constraints(amp, ips.p)

    set_manopt_parameter!(ips.sub_problem, :Manifold, :Basepoint, ips.p)

    # make deterministic instead of random?
    set_iterate!(ips.sub_state, TpM, rand(TpM))

    set_manopt_parameter!(ips.sub_problem, :Objective, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :s, ips.s)
    set_manopt_parameter!(ips.sub_problem, :Objective, :barrier_param, ips.ρ*ips.σ)

    Xp, Xλ = get_solver_result(solve!(ips.sub_problem, ips.sub_state)), zeros(n)

    Xμ = (ips.μ .* (Jg * Xp .+ g)) ./ ips.s
    Xs = (ips.ρ*ips.σ) ./ ips.μ - ips.s - ips.s .* Xμ ./ ips.μ

    α = get_stepsize(amp, ips, i)

    # update params
    retract!(M, ips.p, ips.p, α*Xp, ips.retraction_method)
    ips.μ += α*Xμ
    ips.λ += α*Xλ
    ips.s += α*Xs
    ips.ρ = ips.μ'ips.s / m
    ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s)

    return ips
end

get_solver_result(ips::InteriorPointState) = ips.p