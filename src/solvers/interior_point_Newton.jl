# not-in-place,
# takes M, f, grad_f, Hess_f and possibly constraint functions and their graidents
function interior_point_Newton(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    kwargs...,
)
    q = copy(M, p)
    mho = ManifoldHessianObjective(f, grad_f, Hess_f)
    cmo = ConstrainedManifoldObjective(mho, g, grad_g, h, grad_h; evaluation=evaluation)
    return interior_point_Newton!(
        M, cmo, Hess_g, Hess_h, q; evaluation=evaluation, kwargs...
    )
end

# not-in-place
# case where dim(M) = 1 and in particular p is a number
function interior_point_Newton(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    kwargs...,
)
    q = [p]
    f_(M, p) = f(M, p[])

    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    Hess_f_ = _to_mutating_gradient(Hess_f, evaluation)

    g_ = isnothing(g) ? nothing : (M, p) -> g(M, p[])
    grad_g_ = isnothing(grad_g) ? nothing : _to_mutating_gradient(grad_g, evaluation)
    Hess_g_ = isnothing(Hess_g) ? nothing : _to_mutating_gradient(Hess_g, evaluation)
    h_ = isnothing(h) ? nothing : (M, p) -> h(M, p[])
    grad_h_ = isnothing(grad_h) ? nothing : _to_mutating_gradient(grad_h, evaluation)
    Hess_h_ = isnothing(Hess_h) ? nothing : _to_mutating_gradient(Hess_h, evaluation)

    mho = ManifoldHessianObjective(f_, grad_f_, Hess_f_)
    cmo = ConstrainedManifoldObjective(mho, g_, grad_g_, h_, grad_h_; evaluation=evaluation)

    rs = interior_point_Newton(
        M, cmo, Hess_g_, Hess_h_, q; evaluation=evaluation, kwargs...
    )

    return (typeof(q) == typeof(rs)) ? rs[] : rs
end

# not-in-place
# takes only manifold, constrained objetive and initial point
function interior_point_Newton(
    M::AbstractManifold, 
    cmo::O,
    Hess_g,
    Hess_h,
    p; 
    kwargs...
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return interior_point_Newton!(M, cmo, Hess_g, Hess_h, q; kwargs...)
end

# in-place
# takes M, f, grad_f, Hess_f and possibly constreint functions and their graidents
function interior_point_Newton!(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    kwargs...,
)
    mho = ManifoldHessianObjective(f, grad_f, Hess_f)
    cmo = ConstrainedManifoldObjective(mho, g, grad_g, h, grad_h; evaluation=evaluation)
    dcmo = decorate_objective!(M, cmo; kwargs...)

    return interior_point_Newton!(M, dcmo, p, Hess_g=Hess_g, Hess_h=Hess_h, evaluation=evaluation, kwargs...)
end

# MAIN SOLVER
function interior_point_Newton!(
    M::AbstractManifold,
    cmo::O,
    Hess_g,
    Hess_h,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    X=get_gradient(M, cmo, p),
    μ=ones(length(get_inequality_constraints(M, cmo, p))),
    λ=zeros(length(get_equality_constraints(M, cmo, p))),
    s=μ,
    ρ=μ's / length(get_inequality_constraints(M, cmo, p)),
    σ=calculate_σ(M, cmo, p, μ, λ, s),
    stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-5),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    _N=M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s),
    stepsize::Stepsize=InteriorPointLinesearch(
        _N;
        retraction_method=default_retraction_method(_N),
        additional_decrease_condition=ConstraintLineSearchCheckFunction(
            cmo,
            length(μ) * minimum(μ .* s) / sum(μ .* s),
            sum(μ .* s) / sqrt(MeritFunction(_N, cmo, p, μ, λ, s)),
            0.5,
        ),
        initial_stepsize=1.0,
    ),
    sub_kwargs=(;),
    sub_objective=decorate_objective!(
        TangentSpace(M × ℝ^length(λ), rand(M × ℝ^length(λ))),
        SymmetricLinearSystemObjective(
            ReducedLagrangianHess(cmo, Hess_g, Hess_h, μ, λ, s),
            NegativeReducedLagrangianGrad(cmo, μ, λ, s, ρ * σ),
        ),
        sub_kwargs...,
    ),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(20) |
                                              StopWhenGradientNormLess(1e-5),
    sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState}=decorate_state!(
        ConjugateResidualState(
            TangentSpace(M × ℝ^length(λ), rand(M × ℝ^length(λ))),
            sub_objective;
            stop=sub_stopping_criterion,
            sub_kwargs...,
        );
        sub_kwargs...,
    ),
    sub_problem::Union{F,AbstractManoptProblem}=DefaultManoptProblem(
        TangentSpace(M × ℝ^length(λ), rand(M × ℝ^length(λ))), sub_objective
    ),
    kwargs...,
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective},F}
    !is_feasible(M, cmo, p) && throw(ErrorException("Starting point p must be feasible."))
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = InteriorPointState(
        M,
        cmo,
        Hess_g,
        Hess_h,
        p,
        sub_problem,
        sub_state;
        X=X,
        μ=μ,
        λ=λ,
        s=s,
        stop=stop,
        retraction_method=retraction_method,
        stepsize=stepsize,
        kwargs...,
    )
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
    g = get_inequality_constraints(amp, ips.p)
    Jg = get_grad_inequality_constraints(amp, ips.p)

    N = M × ℝ^n
    q = rand(N)
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.λ)
    TqN = TangentSpace(N, q)

    # make deterministic as opposed to random?
    set_iterate!(ips.sub_state, get_manifold(ips.sub_problem), rand(TqN))

    set_manopt_parameter!(ips.sub_problem, :Manifold, :Basepoint, q)

    set_manopt_parameter!(ips.sub_problem, :Objective, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :s, ips.s)
    set_manopt_parameter!(ips.sub_problem, :Objective, :barrier_param, ips.ρ * ips.σ)

    # product manifold on which to perform linesearch
    K = M × ℝ^m × ℝ^n × ℝ^m
    X = allocate_result(K, rand)

    Xp, Xλ = get_solver_result(solve!(ips.sub_problem, ips.sub_state)).x

    if m > 0
        Xμ = (ips.μ .* ([inner(M, ips.p, Jg[i], Xp) for i in 1:m])) ./ ips.s
        Xs = (ips.ρ * ips.σ) ./ ips.μ - ips.s - ips.s .* Xμ ./ ips.μ
    end

    copyto!(K[1], X[N, 1], Xp)
    (m > 0) && (copyto!(K[2], X[K, 2], Xμ))
    (n > 0) && (copyto!(K[3], X[K, 3], Xλ))
    (m > 0) && (copyto!(K[4], X[K, 4], Xs))

    α = ips.stepsize(amp, ips, i, X)#*0.1

    # update params
    retract!(M, ips.p, ips.p, α * Xp, ips.retraction_method)
    if m > 0
        ips.μ += α * Xμ
        ips.s += α * Xs
        ips.ρ = ips.μ'ips.s / m
        ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s)
    end
    (n > 0) && (ips.λ += α * Xλ)

    get_gradient!(M, ips.X, cmo, ips.p)

    return ips
end

get_solver_result(ips::InteriorPointState) = ips.p
