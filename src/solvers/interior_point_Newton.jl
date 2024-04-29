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
    * p: $(ips.p)
    * μ: $(ips.μ)
    * λ: $(ips.λ)
    * s: $(ips.s)
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
    ρ = μ's / length(μ),
    σ = calculate_σ(M, cmo, p, μ, λ, s),
    stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-5),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stepsize::Stepsize=ArmijoLinesearch(
        M; retraction_method=retraction_method, initial_stepsize=1.0),
    TpM = TangentSpace(M, p),
    sub_kwargs=(;),
    sub_cost = ConjugateResidualCost(
        ReducedLagrangianHess(cmo, μ, λ, s),
        ReducedLagrangianGrad(cmo, μ, λ, s, ρ*σ)(M, p)),
    sub_grad = ConjugateResidualGrad(
        ReducedLagrangianHess(cmo, μ, λ, s),
        ReducedLagrangianGrad(cmo, μ, λ, s, ρ*σ)(M, p)),
    sub_Hess = ConjugateResidualHess(
        ReducedLagrangianHess(cmo, μ, λ, s)),
    sub_objective = decorate_objective!(
        TpM,
        ManifoldHessianObjective(
            sub_cost, sub_grad, sub_Hess;
            sub_kwargs...);
        evaluation=evaluation),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                              StopWhenGradientNormLess(1e-5),
    sub_state::AbstractManoptSolverState=decorate_state!(
        ConjugateResidualState(
            TpM,
            sub_objective;
            stop = sub_stopping_criterion,
            sub_kwargs...,
        );
        sub_kwargs...,
    ),
    sub_problem::AbstractManoptProblem=DefaultManoptProblem(TpM, sub_objective),
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
    TpM = TangentSpace(M, ips.p)

    m, n = length(ips.μ), length(ips.λ)

    # (RB:) This can not work currently, because in the constructor (line 226, the `sub_problem`)
    # (RB:) You set the sub problems manifold to just be TpM
    # (RB:) Can we maybe trick this to be a ℝ^0 for the case of no inequality constraints?
    # if n > 0
    #     set_manopt_parameter!(
    #         ips.sub_problem, :Manifold, :Basepoint, ArrayPartition(ips.p, ips.λ))
    #     set_iterate!(ips.sub_state, TpM × ℝ^n, rand(TpM × ℝ^n))
    # else
    #     set_manopt_parameter!(
    #         ips.sub_problem, :Manifold, :Basepoint, ips.p)
    #     set_iterate!(ips.sub_state, TpM, rand(TpM))
    # end

    # (RB:) I think these are wrong you set the correctly in the lines below – do not replace the function but update the parameters therein.
    # set_manopt_parameter!(
    #     ips.sub_problem, :Objective, :A,
    #     ReducedLagrangianHess(cmo, ips.μ, ips.λ, ips.s))
    # set_manopt_parameter!(
    #     ips.sub_problem, :Objective, :b,
    #     -ReducedLagrangianGrad(cmo, ips.μ, ips.λ, ips.s, ips.ρ*ips.σ)(M, ips.p))

    # This is also a bit more complicated than you think since the Manifold might be a product
    # set_manopt_parameter!(ips.sub_problem, :Manifold, :Basepoint, ips.p)
    # Sure we want to start at a random point? I would prefer a deterministic start if possible.
    set_iterate!(ips.sub_state, TpM, rand(TpM))

    set_manopt_parameter!(ips.sub_problem, :Objective, :Cost, :A, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Cost, :A, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Cost, :A, :s, ips.s)

    set_manopt_parameter!(ips.sub_problem, :Objective, :Cost, :b, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Cost, :b, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Cost, :b, :s, ips.s)
    set_manopt_parameter!(
        ips.sub_problem, :Objective, :Cost, :b, :barrier_param, ips.ρ*ips.σ)

    set_manopt_parameter!(ips.sub_problem, :Objective, :Gradient, :A, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Gradient, :A, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Gradient, :A, :s, ips.s)
    # (RB:) Why twice?
    set_manopt_parameter!(ips.sub_problem, :Objective, :Gradient, :b, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Gradient, :b, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Gradient, :b, :s, ips.s)
    # (RB:) Why again? See 295
    set_manopt_parameter!(
        ips.sub_problem, :Objective, :Cost, :b, :barrier_param, ips.ρ*ips.σ)

    set_manopt_parameter!(ips.sub_problem, :Objective, :Hessian, :A, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Hessian, :A, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :Hessian, :A, :s, ips.s)

    # X = get_solver_result(solve!(ips.sub_problem, ips.sub_state))

    lhs = X -> ReducedLagrangianHess(cmo, ips.μ, ips.λ, ips.s)(M, ips.p, X)
    rhs = -ReducedLagrangianGrad(cmo, ips.μ, ips.λ, ips.s, ips.ρ*ips.σ)(M, ips.p)

    inner_ = (X; Y) -> inner(M, p, X, Y)

    mho = ManifoldHessianObjective(
        (TpM, X) -> 0.5 * inner_(X, lhs(X)) - inner_(rhs, X),
        (TpM, X) -> lhs(X) - rhs,
        (TpM, X, Y) -> lhs(Y)
    )

    X = conjugate_residual(TpM, mho, rand(TpM))

    # get either one or two tangent vectors depending on if equality constrains are present
    # hm this also seems a bit “hacked” – see above maybe an ℝ^0 would be a doable solution?
    (n > 0) ? (Xp, Xλ = X) : (Xp = X)

    α = get_stepsize(amp, ips, i)

    # update p
    retract!(M, ips.p, ips.p, α*Xp, ips.retraction_method)

    # update μ, s and aux parameters if m > 0
    if m > 0
        g = get_inequality_constraints(amp, ips.p)
        Jg = get_grad_inequality_constraints(amp, ips.p)

        b = ips.ρ*ips.σ

        Xμ = (ips.μ .* (Jg * Xp .+ g)) ./ ips.s
        Xs = b ./ ips.μ - ips.s - ips.s .* Xμ ./ ips.μ

        ips.μ += α*Xμ
        ips.s += α*Xs

        ips.ρ = ips.μ'ips.s / m
        ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s)
    end

    # update λ if n > 0
    (n > 0) && (ips.λ += α*Xλ)

    return ips
end

get_solver_result(ips::InteriorPointState) = ips.p